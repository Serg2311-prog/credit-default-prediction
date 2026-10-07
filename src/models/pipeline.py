"""Sklearn Pipeline: признаки, импутация, масштабирование и модель."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import FeatureEngineer, feature_column_groups


def create_pipeline(estimator) -> Pipeline:
    """Собирает pipeline. Препроцессинг обучается только внутри fit на train."""
    numeric_features, categorical_features = feature_column_groups()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )
