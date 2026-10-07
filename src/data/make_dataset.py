"""Загрузка UCI CSV, очистка и стратифицированное разбиение."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import dump_json, load_params, processed_dir, require_raw_csv
from src.data.validation import validate_dataframe
from src.schema import ID_COLUMN, RAW_FEATURE_COLUMNS, detect_target


def load_raw_csv(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def split_train_test(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит выборку до обучения препроцессинга, со стратификацией по target."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_modeling_frame(
    df: pd.DataFrame,
    preferred_target: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Убирает ID, проверяет качество таблицы и оставляет модельные колонки."""
    frame = df.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    source_rows, source_columns = frame.shape
    target = detect_target(frame.columns, preferred=preferred_target)

    id_duplicates = 0
    if ID_COLUMN in frame.columns:
        id_duplicates = int(frame[ID_COLUMN].duplicated().sum())
        if id_duplicates:
            frame = frame.drop_duplicates(subset=[ID_COLUMN], keep="first")
        frame = frame.drop(columns=[ID_COLUMN])

    # Одинаковые профили с разными ID — это разные клиенты, их не удаляем.
    duplicates = int(frame.duplicated().sum())
    frame = frame.reset_index(drop=True)
    nulls = frame.isnull().sum()
    validated = validate_dataframe(frame, target_column=target)
    modeling = validated[RAW_FEATURE_COLUMNS + [target]].copy()

    class_counts = modeling[target].value_counts().sort_index()
    report: dict[str, Any] = {
        "source_rows": int(source_rows),
        "source_columns": int(source_columns),
        "rows": int(len(modeling)),
        "feature_count": len(RAW_FEATURE_COLUMNS),
        "target": target,
        "duplicates": duplicates,
        "id_duplicates": id_duplicates,
        "nulls_total": int(nulls.sum()),
        "nulls_by_column": {column: int(value) for column, value in nulls.items()},
        "dtypes": {column: str(dtype) for column, dtype in modeling.dtypes.items()},
        "class_counts": {
            str(label): int(count) for label, count in class_counts.items()
        },
        "class_share": {
            str(label): float(count / len(modeling))
            for label, count in class_counts.items()
        },
    }
    return modeling, report


def print_data_report(
    report: dict[str, Any], train_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    target = report["target"]
    print(
        "Размер датасета: "
        f"{report['source_rows']} строк, {report['source_columns']} колонок"
    )
    print(f"Целевая переменная: {target}")
    print(f"Количество признаков без ID и target: {report['feature_count']}")
    print("Баланс классов:")
    for label, count in report["class_counts"].items():
        share = report["class_share"][label] * 100
        print(f"  {label}: {count} ({share:.2f}%)")
    print(f"Количество пропусков: {report['nulls_total']}")
    print("Одинаковые профили без ID, оставлены в выборке: " f"{report['duplicates']}")
    print(f"Дубликаты ID: {report['id_duplicates']}")
    print("Типы колонок после подготовки:")
    for column, dtype in report["dtypes"].items():
        print(f"  {column}: {dtype}")
    print(f"Train: {len(train_df)} строк, test: {len(test_df)} строк")
    print(
        "Доля класса 1 — train: "
        f"{train_df[target].mean():.4f}, test: {test_df[target].mean():.4f}"
    )


def prepare_datasets() -> dict[str, Any]:
    params = load_params()["prepare"]
    raw_path = require_raw_csv()
    raw = load_raw_csv(raw_path)
    modeling, report = build_modeling_frame(
        raw, preferred_target=params["target_column"]
    )
    target = report["target"]
    train_df, test_df = split_train_test(
        modeling,
        target=target,
        test_size=float(params["test_size"]),
        random_state=int(params["random_state"]),
    )
    report["raw_path"] = str(raw_path)
    report["test_size"] = float(params["test_size"])
    report["random_state"] = int(params["random_state"])
    report["train_rows"] = int(len(train_df))
    report["test_rows"] = int(len(test_df))

    output_dir = processed_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    report_path = output_dir / "data_report.json"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    dump_json(report_path, report)
    print_data_report(report, train_df, test_df)
    print(f"Сохранено: {train_path}")
    print(f"Сохранено: {test_path}")
    print(f"Сохранено: {report_path}")
    return report


def main() -> None:
    prepare_datasets()


if __name__ == "__main__":
    main()
