"""Тесты загрузки, разбиения и валидации данных."""

import numpy as np
import pandas as pd
import pytest

from src.config import find_raw_csv
from src.data.make_dataset import (
    build_modeling_frame,
    load_raw_csv,
    split_train_test,
)
from src.data.validation import DataValidationError, validate_dataframe
from src.schema import ID_COLUMN, RAW_FEATURE_COLUMNS, TARGET_COLUMN, detect_target


def test_detect_target_canonical_name():
    columns = ["ID", "AGE", TARGET_COLUMN]
    assert detect_target(columns) == TARGET_COLUMN


def test_detect_target_alternative_name(raw_frame):
    renamed = raw_frame.rename(columns={TARGET_COLUMN: "default_payment_next_month"})
    assert detect_target(renamed.columns) == "default_payment_next_month"
    validate_dataframe(renamed)


def test_prepare_drops_id_and_reports_quality(raw_frame):
    raw_frame.insert(0, ID_COLUMN, np.arange(len(raw_frame)))
    prepared, report = build_modeling_frame(raw_frame, preferred_target=TARGET_COLUMN)
    assert ID_COLUMN not in prepared.columns
    assert TARGET_COLUMN in prepared.columns
    assert list(prepared.columns) == RAW_FEATURE_COLUMNS + [TARGET_COLUMN]
    assert report["feature_count"] == len(RAW_FEATURE_COLUMNS)
    assert report["nulls_total"] == 0
    assert report["duplicates"] == 0
    assert set(prepared[TARGET_COLUMN].unique()) == {0, 1}


def test_split_is_stratified_and_disjoint(raw_frame):
    frame = raw_frame.copy()
    frame["_row_id"] = np.arange(len(frame))
    train_df, test_df = split_train_test(
        frame,
        target=TARGET_COLUMN,
        test_size=0.25,
        random_state=42,
    )
    assert set(train_df["_row_id"]).isdisjoint(set(test_df["_row_id"]))
    assert len(train_df) + len(test_df) == len(frame)
    assert abs(train_df[TARGET_COLUMN].mean() - test_df[TARGET_COLUMN].mean()) < 0.1


def test_validation_accepts_synthetic_frame(raw_frame):
    checked = validate_dataframe(raw_frame)
    assert isinstance(checked, pd.DataFrame)
    assert len(checked) == len(raw_frame)


def test_validation_rejects_anomalous_age(raw_frame):
    broken = raw_frame.copy()
    broken.loc[0, "AGE"] = 200
    with pytest.raises(DataValidationError, match="Pandera|AGE|проверк"):
        validate_dataframe(broken)


def test_validation_rejects_bad_target_and_missing_column(raw_frame):
    bad_target = raw_frame.copy()
    bad_target.loc[0, TARGET_COLUMN] = 5
    with pytest.raises(DataValidationError):
        validate_dataframe(bad_target)

    missing = raw_frame.drop(columns=["LIMIT_BAL"])
    with pytest.raises(DataValidationError, match="обязательные колонки"):
        validate_dataframe(missing)


def test_validation_rejects_negative_limit(raw_frame):
    broken = raw_frame.copy()
    broken.loc[0, "LIMIT_BAL"] = -10
    with pytest.raises(DataValidationError):
        validate_dataframe(broken)


def test_real_csv_loads_and_passes_validation():
    csv_path = find_raw_csv()
    if csv_path is None:
        pytest.skip("UCI_Credit_Card.csv не найден")
    raw = load_raw_csv(csv_path)
    assert TARGET_COLUMN in raw.columns
    assert "PAY_0" in raw.columns
    assert "BILL_AMT1" in raw.columns
    prepared, report = build_modeling_frame(raw, preferred_target=TARGET_COLUMN)
    assert report["nulls_total"] == 0
    assert report["target"] == TARGET_COLUMN
    assert report["source_rows"] == 30000
    assert len(prepared) == 30000
    assert report["id_duplicates"] == 0
    assert report["duplicates"] == 35
