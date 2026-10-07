"""Одинаковое построение признаков на обучении и на инференсе.

Агрегаты считаются внутри строки. Статистики по всей выборке здесь не оцениваются,
поэтому шаг не подглядывает в test и не требует отдельного fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.schema import (
    BILL_COLUMNS,
    CATEGORICAL_FEATURES,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PAY_AMOUNT_COLUMNS,
    PAY_STATUS_COLUMNS,
    RAW_FEATURE_COLUMNS,
)

AGE_BINS = [0, 25, 35, 45, 60, 200]
AGE_LABELS = ["<=25", "26-35", "36-45", "46-60", "60+"]


def _as_category_string(series: pd.Series) -> pd.Series:
    def convert(value: object) -> object:
        if pd.isna(value):
            return np.nan
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if number.is_integer():
            return str(int(number))
        return str(value)

    return series.map(convert, na_action="ignore")


def _age_group(age: pd.Series) -> pd.Series:
    grouped = pd.cut(
        age,
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,
        include_lowest=True,
    )
    result = pd.Series(grouped.astype("object"), index=age.index)
    result[grouped.isna()] = np.nan
    return result


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет агрегаты по PAY_*, BILL_AMT* и PAY_AMT* и убирает ID/target."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("add_features ожидает pandas.DataFrame с исходными колонками")

    missing = [column for column in RAW_FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Для feature engineering не хватает колонок: " + ", ".join(missing)
        )

    features = df.loc[:, RAW_FEATURE_COLUMNS].copy()
    bills = features[BILL_COLUMNS].astype(float)
    payments = features[PAY_AMOUNT_COLUMNS].astype(float)
    statuses = features[PAY_STATUS_COLUMNS].astype(float)

    features["bill_amt_mean"] = bills.mean(axis=1)
    features["bill_amt_max"] = bills.max(axis=1)
    features["pay_amt_mean"] = payments.mean(axis=1)
    features["pay_amt_max"] = payments.max(axis=1)
    positive_bills = bills.clip(lower=0).sum(axis=1)
    features["pay_to_bill_ratio"] = payments.sum(axis=1) / (positive_bills + 1.0)
    features["delay_months_count"] = (statuses > 0).sum(axis=1).astype(float)
    features["max_delay"] = statuses.max(axis=1)
    features["mean_delay"] = statuses.mean(axis=1)

    limit = features["LIMIT_BAL"].astype(float).replace(0, np.nan)
    features["utilisation"] = features["BILL_AMT1"].astype(float) / limit
    features["age_group"] = _age_group(features["AGE"])

    for column in ("SEX", "EDUCATION", "MARRIAGE"):
        features[column] = _as_category_string(features[column])

    return features.loc[:, MODEL_FEATURE_COLUMNS].copy()


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-шаг: сырые поля клиента -> модельная матрица признаков."""

    def fit(self, X, y=None):  # noqa: N803
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer ожидает pandas.DataFrame")
        missing = [column for column in RAW_FEATURE_COLUMNS if column not in X.columns]
        if missing:
            raise ValueError(
                "Для feature engineering не хватает колонок: " + ", ".join(missing)
            )
        return self

    def transform(self, X):  # noqa: N803
        return add_features(X)


def feature_column_groups() -> tuple[list[str], list[str]]:
    return list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES)
