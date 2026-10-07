"""Инференс сохранённой sklearn Pipeline на сырых полях клиента."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_params, model_path
from src.schema import ID_COLUMN, detect_target


def positive_class_probability(model, frame: pd.DataFrame) -> np.ndarray:
    """Вероятность класса 1 независимо от порядка model.classes_."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Модель не поддерживает predict_proba")
    probabilities = model.predict_proba(frame)
    classes = [int(label) for label in model.classes_]
    if 1 not in classes:
        raise ValueError("В модели нет положительного класса 1")
    return probabilities[:, classes.index(1)]


def model_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Убирает ID и target, если они попали в таблицу вместе с сырыми признаками."""
    columns_to_drop: list[str] = []
    if ID_COLUMN in frame.columns:
        columns_to_drop.append(ID_COLUMN)
    try:
        target = detect_target(frame.columns)
    except ValueError:
        target = None
    if target is not None and target in frame.columns:
        columns_to_drop.append(target)
    if columns_to_drop:
        return frame.drop(columns=columns_to_drop)
    return frame


def predict_frame(
    model,
    frame: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    features = model_input_frame(frame)
    probability = positive_class_probability(model, features)
    prediction = (probability >= threshold).astype(int)
    result = frame.copy()
    result["default_probability"] = probability
    result["prediction"] = prediction
    return result


def load_model(path: Path | None = None):
    model_file = path or model_path()
    if not model_file.is_file():
        raise FileNotFoundError(
            "Файл модели не найден: "
            f"{model_file}. Сначала выполните python -m src.models.train"
        )
    return joblib.load(model_file)


def predict_file(
    input_path: Path, output_path: Path, model_file: Path | None = None
) -> Path:
    params = load_params()
    threshold = float(params["train"]["threshold"])
    model = load_model(model_file)
    frame = pd.read_csv(input_path)
    predicted = predict_frame(model, frame, threshold=threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(output_path, index=False)
    print(f"Предсказания сохранены: {output_path}")
    print(f"Строк: {len(predicted)}")
    print(f"Средняя вероятность дефолта: {predicted['default_probability'].mean():.4f}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Предсказание дефолта для CSV с сырыми полями"
    )
    parser.add_argument("--input", required=True, help="CSV с признаками клиентов")
    parser.add_argument("--output", required=True, help="Куда сохранить предсказания")
    parser.add_argument("--model", default=None, help="Путь к pickle модели")
    args = parser.parse_args()
    model_file = Path(args.model) if args.model else None
    predict_file(Path(args.input), Path(args.output), model_file=model_file)


if __name__ == "__main__":
    main()
