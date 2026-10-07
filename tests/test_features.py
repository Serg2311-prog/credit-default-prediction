"""Тесты feature engineering."""

import numpy as np
import pandas as pd

from src.features.build_features import FeatureEngineer, add_features
from src.schema import (
    ID_COLUMN,
    MODEL_FEATURE_COLUMNS,
    PAY_STATUS_COLUMNS,
    TARGET_COLUMN,
)


def test_add_features_excludes_target_and_id(raw_frame):
    frame = raw_frame.copy()
    frame[ID_COLUMN] = np.arange(len(frame))
    features = add_features(frame)
    assert ID_COLUMN not in features.columns
    assert TARGET_COLUMN not in features.columns
    assert list(features.columns) == MODEL_FEATURE_COLUMNS


def test_feature_engineering_is_same_for_train_and_inference(raw_frame):
    with_target = add_features(raw_frame)
    without_target = add_features(raw_frame.drop(columns=[TARGET_COLUMN]))
    pd.testing.assert_frame_equal(with_target, without_target)


def test_delay_and_ratio_values(raw_frame):
    row = raw_frame.iloc[[0]].copy()
    features = add_features(row)
    expected_delays = int((row[PAY_STATUS_COLUMNS] > 0).sum(axis=1).iloc[0])
    assert features.loc[features.index[0], "delay_months_count"] == expected_delays
    assert (
        features.loc[features.index[0], "max_delay"]
        == row[PAY_STATUS_COLUMNS].max(axis=1).iloc[0]
    )
    assert "26-35" in set(features["age_group"]) or features["age_group"].notna().all()


def test_zero_limit_and_negative_bill_do_not_break_ratios(raw_frame):
    frame = raw_frame.copy()
    frame.loc[0, "LIMIT_BAL"] = 0
    frame.loc[0, "BILL_AMT1"] = -50
    features = add_features(frame)
    assert np.isnan(features.loc[0, "utilisation"])
    assert np.isfinite(features.loc[0, "pay_to_bill_ratio"])


def test_transformer_is_stateless(raw_frame):
    transformer = FeatureEngineer()
    first = transformer.fit_transform(raw_frame)
    second = transformer.transform(raw_frame.drop(columns=[TARGET_COLUMN]))
    pd.testing.assert_frame_equal(first, second)
