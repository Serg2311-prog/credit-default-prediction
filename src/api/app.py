"""FastAPI: сырые поля клиента -> вероятность дефолта."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import load_params, model_path
from src.features import build_features
from src.models.predict import predict_frame

MODEL_PATH = model_path()
model = None


class ClientFeatures(BaseModel):
    """Сырые поля одного клиента. Агрегаты API считает само."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "LIMIT_BAL": 20000.0,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 24,
                "PAY_0": 2,
                "PAY_2": 2,
                "PAY_3": -1,
                "PAY_4": -1,
                "PAY_5": -2,
                "PAY_6": -2,
                "BILL_AMT1": 3913.0,
                "BILL_AMT2": 3102.0,
                "BILL_AMT3": 689.0,
                "BILL_AMT4": 0.0,
                "BILL_AMT5": 0.0,
                "BILL_AMT6": 0.0,
                "PAY_AMT1": 0.0,
                "PAY_AMT2": 689.0,
                "PAY_AMT3": 0.0,
                "PAY_AMT4": 0.0,
                "PAY_AMT5": 0.0,
                "PAY_AMT6": 0.0,
            }
        }
    )

    LIMIT_BAL: float = Field(..., ge=0)
    SEX: int = Field(..., ge=1, le=2)
    EDUCATION: int = Field(..., ge=0, le=6)
    MARRIAGE: int = Field(..., ge=0, le=3)
    AGE: int = Field(..., ge=18, le=100)
    PAY_0: int = Field(..., ge=-2, le=8)
    PAY_2: int = Field(..., ge=-2, le=8)
    PAY_3: int = Field(..., ge=-2, le=8)
    PAY_4: int = Field(..., ge=-2, le=8)
    PAY_5: int = Field(..., ge=-2, le=8)
    PAY_6: int = Field(..., ge=-2, le=8)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int = Field(..., examples=[0])
    default_probability: float = Field(..., examples=[0.1734])


def decision_threshold() -> float:
    try:
        return float(load_params()["train"]["threshold"])
    except (KeyError, TypeError, ValueError):
        return 0.5


def load_model():
    # Класс нужен joblib, когда pickle восстанавливает шаг FeatureEngineer.
    if build_features.FeatureEngineer is None:  # pragma: no cover
        raise RuntimeError("FeatureEngineer недоступен")
    if not MODEL_PATH.is_file():
        return None
    import joblib

    return joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    yield
    model = None


app = FastAPI(
    title="Credit Default Prediction API",
    description=(
        "Прогноз вероятности дефолта по сырым полям клиента. "
        "Feature engineering и препроцессинг выполняются внутри сохранённой Pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Credit Default Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ClientFeatures) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Модель не загружена. Сначала обучите её и положите "
                "models/credit_default_model.pkl"
            ),
        )
    frame = pd.DataFrame([payload.model_dump()])
    try:
        predicted = predict_frame(model, frame, threshold=decision_threshold())
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось выполнить предсказание: {exc}",
        ) from exc

    probability = round(float(predicted["default_probability"].iloc[0]), 4)
    prediction = int(predicted["prediction"].iloc[0])
    return PredictionResponse(prediction=prediction, default_probability=probability)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
