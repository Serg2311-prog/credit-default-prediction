"""Метрики и графики финальной оценки. Test не используется для выбора модели."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from src.config import artifacts_dir, dump_json, load_params, model_path, processed_dir
from src.models.predict import load_model, positive_class_probability
from src.plot_utils import configure_plots, plt
from src.schema import detect_target


def compute_metrics(
    y_true,
    probability,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "tn": int(true_negative),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "tp": int(true_positive),
    }


def build_evaluation_figures(y_true, probability, threshold: float = 0.5):
    configure_plots()
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    figures = []

    figure, axis = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, probability, ax=axis)
    axis.set_title("ROC-кривая")
    figures.append(("roc_curve", figure))

    figure, axis = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, probability, ax=axis)
    axis.set_title("Precision-Recall кривая")
    figures.append(("pr_curve", figure))

    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, prediction, ax=axis)
    axis.set_title("Матрица ошибок")
    figures.append(("confusion_matrix", figure))
    return figures


def save_evaluation_plots(
    y_true,
    probability,
    output_dir: Path,
    threshold: float = 0.5,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, figure in build_evaluation_figures(y_true, probability, threshold):
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)
    return saved


def evaluate_model(
    model_file: Path | None = None,
    test_file: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, float]:
    params = load_params()
    threshold = float(params["train"]["threshold"])
    model = load_model(model_file or model_path())
    test_path = test_file or (processed_dir() / "test.csv")
    if not test_path.is_file():
        raise FileNotFoundError(
            f"Не найден test.csv: {test_path}. Сначала выполните подготовку данных."
        )

    test_df = pd.read_csv(test_path)
    target = detect_target(
        test_df.columns, preferred=params["prepare"]["target_column"]
    )
    features = test_df.drop(columns=[target])
    probability = positive_class_probability(model, features)
    metrics = compute_metrics(test_df[target], probability, threshold=threshold)
    metrics_payload = {
        "threshold": threshold,
        "rows": int(len(test_df)),
        "target": target,
        "metrics": metrics,
        "note": "Метрики test посчитаны после выбора модели по CV ROC-AUC.",
    }
    destination = output_dir or artifacts_dir()
    dump_json(destination / "test_metrics.json", metrics_payload)
    paths = save_evaluation_plots(
        test_df[target],
        probability,
        destination,
        threshold=threshold,
    )
    print("Метрики на test:")
    for name in ("roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"):
        print(f"  {name}: {metrics[name]:.4f}")
    for path in paths:
        print(f"График: {path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Финальная оценка сохранённой модели на test"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    evaluate_model(
        model_file=Path(args.model) if args.model else None,
        test_file=Path(args.test) if args.test else None,
        output_dir=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
