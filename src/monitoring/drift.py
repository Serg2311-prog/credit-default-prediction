"""Population Stability Index между train и новой порцией наблюдений."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import artifacts_dir, dump_json, load_params, model_path, processed_dir
from src.features.build_features import add_features
from src.models.predict import load_model, positive_class_probability
from src.schema import detect_target

PSI_EPS = 1e-6
NUMERIC_PSI_COLUMNS = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "BILL_AMT1",
    "PAY_AMT1",
    "utilisation",
]
CATEGORICAL_PSI_COLUMNS = ["EDUCATION", "SEX", "MARRIAGE"]


def interpret_psi(value: float) -> str:
    if value < 0.1:
        return "значимого drift нет"
    if value < 0.25:
        return "умеренный drift"
    return "существенный drift"


def _value_edges(unique_values: np.ndarray) -> np.ndarray:
    """Отдельный бин на каждое редкое значение, чтобы 0/1 не схлопывались в один."""
    if len(unique_values) == 1:
        center = float(unique_values[0])
        return np.array([center - 0.5, center + 0.5], dtype=float)
    mids = (unique_values[:-1] + unique_values[1:]) / 2.0
    left = float(unique_values[0] - (mids[0] - unique_values[0]))
    right = float(unique_values[-1] + (unique_values[-1] - mids[-1]))
    return np.concatenate(([left], mids, [right])).astype(float)


def _bin_edges(expected: np.ndarray, bins: int) -> np.ndarray:
    unique_values = np.unique(expected.astype(float))
    if len(unique_values) <= max(bins, 15):
        return _value_edges(unique_values)

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles)).astype(float)
    if len(edges) >= 3:
        return edges

    low = float(np.min(expected))
    high = float(np.max(expected))
    if high <= low:
        return np.array([low - 0.5, low + 0.5], dtype=float)
    return np.linspace(low, high, bins + 1)


def _shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    shares = counts.astype(float)
    total = shares.sum()
    if total <= 0:
        shares = np.ones_like(shares, dtype=float)
        total = float(shares.sum())
    shares = shares / total
    shares = np.clip(shares, PSI_EPS, None)
    return shares / shares.sum()


def population_stability_index(
    expected,
    actual,
    bins: int = 10,
) -> float:
    """PSI по квантилям baseline. Нулевые доли заменяются малым числом."""
    baseline = np.asarray(expected, dtype=float)
    current = np.asarray(actual, dtype=float)
    baseline = baseline[np.isfinite(baseline)]
    current = current[np.isfinite(current)]
    if len(baseline) == 0 or len(current) == 0:
        raise ValueError("Для PSI нужны непустые числовые выборки")

    edges = _bin_edges(baseline, bins)
    # Правый край включаем, иначе максимум baseline может выпасть из последнего бина.
    edges = edges.copy()
    span = max(abs(edges[-1] - edges[0]), 1.0)
    edges[0] = min(edges[0], float(np.min(current))) - span * PSI_EPS
    edges[-1] = max(edges[-1], float(np.max(current))) + span * PSI_EPS
    if np.any(np.diff(edges) <= 0):
        edges = np.array(
            [
                min(baseline.min(), current.min()) - 0.5,
                max(baseline.max(), current.max()) + 0.5,
            ]
        )

    expected_share = _shares(baseline, edges)
    actual_share = _shares(current, edges)
    psi = np.sum(
        (actual_share - expected_share) * np.log(actual_share / expected_share)
    )
    if not np.isfinite(psi):
        raise ValueError("PSI получился нечисловым")
    if abs(psi) < 1e-12:
        return 0.0
    return float(max(psi, 0.0))


def population_stability_index_categorical(expected, actual) -> float:
    baseline = pd.Series(expected).dropna().astype(str)
    current = pd.Series(actual).dropna().astype(str)
    if baseline.empty or current.empty:
        raise ValueError("Для категориального PSI нужны непустые выборки")

    categories = sorted(set(baseline.unique()) | set(current.unique()))
    expected_counts = baseline.value_counts(normalize=True)
    actual_counts = current.value_counts(normalize=True)
    expected_share = np.array(
        [expected_counts.get(item, 0.0) for item in categories], dtype=float
    )
    actual_share = np.array(
        [actual_counts.get(item, 0.0) for item in categories], dtype=float
    )
    expected_share = np.clip(expected_share, PSI_EPS, None)
    actual_share = np.clip(actual_share, PSI_EPS, None)
    expected_share = expected_share / expected_share.sum()
    actual_share = actual_share / actual_share.sum()
    psi = np.sum(
        (actual_share - expected_share) * np.log(actual_share / expected_share)
    )
    if abs(psi) < 1e-12:
        return 0.0
    return float(max(psi, 0.0))


def _monitoring_slice(
    test_df: pd.DataFrame, fraction: float, random_state: int
) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("sample_fraction должен быть в интервале (0, 1]")
    if fraction == 1:
        return test_df.reset_index(drop=True)
    return test_df.sample(frac=fraction, random_state=random_state).reset_index(
        drop=True
    )


def calculate_drift() -> dict[str, Any]:
    params = load_params()
    monitor = params.get("monitor", {})
    fraction = float(monitor.get("sample_fraction", 0.5))
    bins = int(monitor.get("bins", 10))
    random_state = int(params["prepare"]["random_state"])
    target_hint = params["prepare"]["target_column"]

    train_path = processed_dir() / "train.csv"
    test_path = processed_dir() / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Нет train/test. Сначала выполните подготовку данных.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    target = detect_target(train_df.columns, preferred=target_hint)
    new_df = _monitoring_slice(test_df, fraction, random_state)

    train_features = add_features(train_df.drop(columns=[target]))
    new_features = add_features(new_df.drop(columns=[target]))
    rows = []

    for column in NUMERIC_PSI_COLUMNS:
        value = population_stability_index(
            train_features[column],
            new_features[column],
            bins=bins,
        )
        rows.append(
            {
                "feature": column,
                "kind": "numeric",
                "psi": value,
                "interpretation": interpret_psi(value),
            }
        )

    for column in CATEGORICAL_PSI_COLUMNS:
        value = population_stability_index_categorical(
            train_features[column],
            new_features[column],
        )
        rows.append(
            {
                "feature": column,
                "kind": "categorical",
                "psi": value,
                "interpretation": interpret_psi(value),
            }
        )

    model_file = model_path()
    if model_file.is_file():
        model = load_model(model_file)
        baseline_probability = positive_class_probability(
            model,
            train_df.drop(columns=[target]),
        )
        new_probability = positive_class_probability(
            model, new_df.drop(columns=[target])
        )
        value = population_stability_index(
            baseline_probability, new_probability, bins=bins
        )
        rows.append(
            {
                "feature": "predicted_probability",
                "kind": "prediction",
                "psi": value,
                "interpretation": interpret_psi(value),
            }
        )
    else:
        print(f"Модель не найдена ({model_file}), PSI вероятности пропущен.")

    report = {
        "baseline": "train",
        "current": f"test sample_fraction={fraction}",
        "baseline_rows": int(len(train_df)),
        "current_rows": int(len(new_df)),
        "rules": {
            "psi < 0.1": "значимого drift нет",
            "0.1 <= psi < 0.25": "умеренный drift",
            "psi >= 0.25": "существенный drift",
        },
        "features": rows,
    }
    output = artifacts_dir() / "psi_report.json"
    dump_json(output, report)
    print(f"Baseline: train ({len(train_df)} строк)")
    print(f"Новые данные: часть test ({len(new_df)} строк, доля {fraction})")
    print(f"{'Признак':<28} {'PSI':>10}  Интерпретация")
    for row in rows:
        print(f"{row['feature']:<28} {row['psi']:>10.4f}  {row['interpretation']}")
    print(f"Отчёт: {output}")
    return report


def main() -> None:
    calculate_drift()


if __name__ == "__main__":
    main()
