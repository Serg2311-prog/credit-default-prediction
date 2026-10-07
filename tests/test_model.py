"""Тесты pipeline, метрик, предсказания и PSI."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.build_features import add_features
from src.models.evaluate import compute_metrics, save_evaluation_plots
from src.models.pipeline import create_pipeline
from src.models.predict import positive_class_probability, predict_frame
from src.monitoring.drift import (
    interpret_psi,
    population_stability_index,
    population_stability_index_categorical,
)
from src.schema import NUMERIC_FEATURES, RAW_FEATURE_COLUMNS, TARGET_COLUMN


def _fit_logreg(frame: pd.DataFrame):
    pipeline = create_pipeline(LogisticRegression(max_iter=300, solver="lbfgs"))
    pipeline.fit(frame[RAW_FEATURE_COLUMNS], frame[TARGET_COLUMN])
    return pipeline


def test_pipeline_accepts_raw_columns(raw_frame):
    features = raw_frame[RAW_FEATURE_COLUMNS]
    assert "bill_amt_mean" not in features.columns
    pipeline = _fit_logreg(raw_frame)
    prediction = pipeline.predict(features.iloc[:5])
    probability = positive_class_probability(pipeline, features.iloc[:5])
    assert len(prediction) == 5
    assert probability.shape == (5,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()


def test_pipeline_imputes_engineered_missing_values(raw_frame):
    frame = raw_frame.copy()
    frame.loc[0, "LIMIT_BAL"] = 0
    pipeline = _fit_logreg(frame)
    probability = positive_class_probability(pipeline, frame[RAW_FEATURE_COLUMNS])
    assert np.isfinite(probability).all()


def test_scaler_is_fit_on_train_only(raw_frame):
    train = raw_frame.iloc[:60].copy()
    test = raw_frame.iloc[60:].copy()
    test["LIMIT_BAL"] = 1_000_000
    pipeline = _fit_logreg(train)
    scaler = (
        pipeline.named_steps["preprocess"]
        .named_transformers_["num"]
        .named_steps["scaler"]
    )
    limit_index = NUMERIC_FEATURES.index("LIMIT_BAL")
    transformed_train = add_features(train)
    assert np.isclose(scaler.mean_[limit_index], transformed_train["LIMIT_BAL"].mean())
    assert not np.isclose(scaler.mean_[limit_index], test["LIMIT_BAL"].mean())


def test_unknown_category_is_ignored(raw_frame):
    pipeline = _fit_logreg(raw_frame)
    changed = raw_frame.iloc[:8].copy()
    changed["EDUCATION"] = 6
    probability = positive_class_probability(pipeline, changed[RAW_FEATURE_COLUMNS])
    assert np.isfinite(probability).all()


def test_predict_frame_adds_score_columns(raw_frame):
    pipeline = _fit_logreg(raw_frame)
    predicted = predict_frame(pipeline, raw_frame[RAW_FEATURE_COLUMNS], threshold=0.5)
    assert {"prediction", "default_probability"}.issubset(predicted.columns)
    assert set(predicted["prediction"].unique()).issubset({0, 1})


def test_compute_metrics_and_plots(tmp_path: Path):
    y_true = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, probability, threshold=0.5)
    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    paths = save_evaluation_plots(y_true, probability, tmp_path)
    assert len(paths) == 3
    assert all(path.is_file() for path in paths)


def test_psi_identical_distribution_is_near_zero():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=400)
    assert population_stability_index(sample, sample.copy()) < 0.02


def test_psi_protects_empty_bins_and_detects_shift():
    baseline = np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=float)
    disjoint = np.array([10, 10, 10, 11, 12], dtype=float)
    value = population_stability_index(baseline, disjoint, bins=5)
    assert np.isfinite(value)

    stable = np.ones(200)
    assert population_stability_index(stable, stable.copy()) < 0.05

    expected = np.concatenate([np.zeros(180), np.ones(20)])
    shifted = np.concatenate([np.zeros(20), np.ones(180)])
    drifted = population_stability_index(expected, shifted, bins=4)
    assert drifted >= 0.25


def test_categorical_psi_and_interpretation():
    assert (
        population_stability_index_categorical(["a", "a", "b"], ["a", "a", "b"]) < 0.05
    )
    assert interpret_psi(0.05) == "значимого drift нет"
    assert interpret_psi(0.2) == "умеренный drift"
    assert interpret_psi(0.3) == "существенный drift"
