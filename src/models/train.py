"""Обучение нескольких моделей, логирование в MLflow и сохранение лучшей.

Лучший run выбирается по среднему ROC-AUC на кросс-валидации train.
Test считается только для отчёта и не участвует в выборе.
"""

from __future__ import annotations

from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.config import (
    PROJECT_ROOT,
    artifacts_dir,
    dump_json,
    load_params,
    model_path,
    processed_dir,
)
from src.models.evaluate import build_evaluation_figures, compute_metrics
from src.models.pipeline import create_pipeline
from src.models.predict import positive_class_probability
from src.plot_utils import plt
from src.schema import detect_target


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_estimator(algorithm: str, random_state: int):
    if algorithm == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        )
    if algorithm == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
    if algorithm == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Неизвестный алгоритм: {algorithm}")


def algorithm_label(algorithm: str) -> str:
    labels = {
        "logistic_regression": "LogisticRegression",
        "random_forest": "RandomForestClassifier",
        "gradient_boosting": "GradientBoostingClassifier",
    }
    return labels[algorithm]


def class_weight_label(estimator) -> str:
    value = getattr(estimator, "class_weight", None)
    if value is None:
        return "not_supported"
    return str(value)


def run_name(algorithm: str, params: dict[str, Any], index: int) -> str:
    parts = [algorithm_label(algorithm), f"{index:02d}"]
    for key, value in sorted(params.items()):
        short = key.split("__")[-1]
        parts.append(f"{short}={value}")
    return "_".join(parts)


def make_search(
    pipeline, experiment: dict[str, Any], cv, n_jobs: int, random_state: int
):
    method = experiment["search"]
    if method == "grid":
        return (
            GridSearchCV(
                estimator=pipeline,
                param_grid=experiment["param_grid"],
                scoring="roc_auc",
                cv=cv,
                n_jobs=n_jobs,
                refit=True,
                error_score="raise",
            ),
            "GridSearchCV",
        )
    if method == "random":
        return (
            RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=experiment["param_distributions"],
                n_iter=int(experiment["n_iter"]),
                scoring="roc_auc",
                cv=cv,
                n_jobs=n_jobs,
                refit=True,
                random_state=random_state,
                error_score="raise",
            ),
            "RandomizedSearchCV",
        )
    raise ValueError(f"Неизвестный способ подбора: {method}")


def log_run(
    model,
    run_title: str,
    algorithm: str,
    search_method: str,
    params: dict[str, Any],
    cv_score: float,
    y_test,
    probability,
    threshold: float,
) -> str:
    metrics = compute_metrics(y_test, probability, threshold=threshold)
    with mlflow.start_run(run_name=run_title):
        active = mlflow.active_run()
        run_id = active.info.run_id
        mlflow.set_tag("algorithm", algorithm_label(algorithm))
        mlflow.set_tag("search_method", search_method)
        mlflow.log_param("algorithm", algorithm_label(algorithm))
        mlflow.log_param("search_method", search_method)
        mlflow.log_param(
            "class_weight",
            class_weight_label(model.named_steps["model"]),
        )
        mlflow.log_params(
            {key.split("__")[-1]: _plain(value) for key, value in params.items()}
        )
        mlflow.log_metric("cv_roc_auc", float(cv_score))
        for name in ("roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"):
            mlflow.log_metric(name, metrics[name])
        for figure_name, figure in build_evaluation_figures(
            y_test, probability, threshold
        ):
            mlflow.log_figure(figure, f"plots/{figure_name}.png")
            plt.close(figure)
        mlflow.sklearn.log_model(model, artifact_path="model")
    return run_id


def train() -> dict[str, Any]:
    params = load_params()
    prepare_params = params["prepare"]
    train_params = params["train"]
    random_state = int(train_params["random_state"])
    threshold = float(train_params["threshold"])
    cv_folds = int(train_params["cv"])
    n_jobs = int(train_params["n_jobs"])
    experiment_name = train_params["experiment_name"]

    train_path = processed_dir() / "train.csv"
    test_path = processed_dir() / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "Нет подготовленных данных. "
            "Сначала выполните python -m src.data.make_dataset"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    target = detect_target(train_df.columns, preferred=prepare_params["target_column"])
    x_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    x_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    tracking_uri = (PROJECT_ROOT / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    run_index = 1

    for experiment in train_params["experiments"]:
        algorithm = experiment["algorithm"]
        estimator = build_estimator(algorithm, random_state)
        pipeline = create_pipeline(estimator)
        search, search_method = make_search(
            pipeline,
            experiment,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        print(f"Подбор {algorithm_label(algorithm)} через {search_method}")
        search.fit(x_train, y_train)

        for candidate_index, candidate_params in enumerate(
            search.cv_results_["params"]
        ):
            cv_score = float(search.cv_results_["mean_test_score"][candidate_index])
            if candidate_index == int(search.best_index_):
                model = search.best_estimator_
            else:
                model = create_pipeline(build_estimator(algorithm, random_state))
                model.set_params(**candidate_params)
                model.fit(x_train, y_train)

            probability = positive_class_probability(model, x_test)
            title = run_name(algorithm, candidate_params, run_index)
            run_id = log_run(
                model=model,
                run_title=title,
                algorithm=algorithm,
                search_method=search_method,
                params=candidate_params,
                cv_score=cv_score,
                y_test=y_test,
                probability=probability,
                threshold=threshold,
            )
            metrics = compute_metrics(y_test, probability, threshold=threshold)
            record = {
                "run_name": title,
                "run_id": run_id,
                "algorithm": algorithm_label(algorithm),
                "search_method": search_method,
                "params": {
                    key: _plain(value) for key, value in candidate_params.items()
                },
                "cv_roc_auc": cv_score,
                "test_metrics": metrics,
            }
            results.append(record)
            print(
                f"  {title}: CV ROC-AUC={cv_score:.4f}, "
                f"test ROC-AUC={metrics['roc_auc']:.4f}"
            )
            if best is None or cv_score > best["cv_roc_auc"]:
                best = {**record, "model": model}
            run_index += 1

    if best is None or len(results) < 5:
        raise RuntimeError(
            f"Ожидалось минимум 5 запусков MLflow, получено {len(results)}"
        )

    destination = model_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], destination)

    summary_runs = [
        {key: value for key, value in item.items() if key != "model"}
        for item in results
    ]
    best_public = {key: value for key, value in best.items() if key != "model"}
    summary = {
        "experiment": experiment_name,
        "tracking_uri": tracking_uri,
        "selection_metric": "cv_roc_auc",
        "selection_note": (
            "Модель выбрана по CV ROC-AUC на train. "
            "Test использован только для отчёта."
        ),
        "n_runs": len(results),
        "model_path": str(destination),
        "best": best_public,
        "runs": summary_runs,
    }
    dump_json(artifacts_dir() / "training_summary.json", summary)
    print(f"Запусков MLflow: {len(results)}")
    print(f"Лучший алгоритм по CV ROC-AUC: {best_public['algorithm']}")
    print(f"CV ROC-AUC: {best_public['cv_roc_auc']:.4f}")
    print(f"Test ROC-AUC лучшей модели: {best_public['test_metrics']['roc_auc']:.4f}")
    print(f"Модель сохранена: {destination}")
    return summary


def main() -> None:
    train()


if __name__ == "__main__":
    main()
