"""Тесты FastAPI."""

import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.api import app as app_module
from src.models.pipeline import create_pipeline
from src.schema import RAW_FEATURE_COLUMNS, TARGET_COLUMN


@pytest.fixture
def fitted_model(raw_frame):
    pipeline = create_pipeline(LogisticRegression(max_iter=300, solver="lbfgs"))
    pipeline.fit(raw_frame[RAW_FEATURE_COLUMNS], raw_frame[TARGET_COLUMN])
    return pipeline


@pytest.fixture
def client(tmp_path, fitted_model, monkeypatch):
    model_file = tmp_path / "credit_default_model.pkl"
    joblib.dump(fitted_model, model_file)
    monkeypatch.setattr(app_module, "MODEL_PATH", model_file)
    monkeypatch.setattr(app_module, "model", None)
    with TestClient(app_module.app) as test_client:
        yield test_client


def _payload(raw_frame) -> dict:
    return json.loads(raw_frame[RAW_FEATURE_COLUMNS].iloc[0].to_json())


def test_root_and_health(client):
    root = client.get("/")
    health = client.get("/health")
    assert root.status_code == 200
    assert root.json()["docs"] == "/docs"
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["model_loaded"] is True


def test_predict_returns_probability(client, raw_frame):
    response = client.post("/predict", json=_payload(raw_frame))
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["default_probability"] <= 1.0


def test_predict_rejects_anomalous_age(client, raw_frame):
    payload = _payload(raw_frame)
    payload["AGE"] = 200
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_requires_raw_fields(client):
    response = client.post("/predict", json={"AGE": 30})
    assert response.status_code == 422


def test_predict_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr(app_module, "model", None)
    with TestClient(app_module.app) as client:
        response = client.post(
            "/predict",
            json={
                "LIMIT_BAL": 20000,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 24,
                "PAY_0": 0,
                "PAY_2": 0,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
                "BILL_AMT1": 1000,
                "BILL_AMT2": 1000,
                "BILL_AMT3": 1000,
                "BILL_AMT4": 1000,
                "BILL_AMT5": 1000,
                "BILL_AMT6": 1000,
                "PAY_AMT1": 100,
                "PAY_AMT2": 100,
                "PAY_AMT3": 100,
                "PAY_AMT4": 100,
                "PAY_AMT5": 100,
                "PAY_AMT6": 100,
            },
        )
    assert response.status_code == 503
