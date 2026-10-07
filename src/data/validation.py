"""Проверка таблицы клиентов схемой Pandera."""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:  # pragma: no cover
    import pandera as pa

from src.schema import (
    BILL_COLUMNS,
    PAY_AMOUNT_COLUMNS,
    PAY_STATUS_COLUMNS,
    RAW_FEATURE_COLUMNS,
    detect_target,
)

# В выгрузке UCI встречаются коды, которых нет в кратком описании:
# EDUCATION: 0, 5, 6; MARRIAGE: 0; PAY_*: от -2 до 8. Их нужно пропускать.
EDUCATION_VALUES = [0, 1, 2, 3, 4, 5, 6]
MARRIAGE_VALUES = [0, 1, 2, 3]
SEX_VALUES = [1, 2]


class DataValidationError(ValueError):
    """Таблица не соответствует ожидаемой схеме."""


def _numeric_column(*checks: Any) -> pa.Column:
    return pa.Column(
        float,
        checks=list(checks),
        nullable=False,
        coerce=True,
    )


def _int_column(*checks: Any) -> pa.Column:
    return pa.Column(
        int,
        checks=list(checks),
        nullable=False,
        coerce=True,
    )


def build_schema(target_column: str) -> pa.DataFrameSchema:
    columns: dict[str, pa.Column] = {
        "LIMIT_BAL": _numeric_column(pa.Check.ge(0)),
        "SEX": _int_column(pa.Check.isin(SEX_VALUES)),
        "EDUCATION": _int_column(pa.Check.isin(EDUCATION_VALUES)),
        "MARRIAGE": _int_column(pa.Check.isin(MARRIAGE_VALUES)),
        "AGE": _int_column(pa.Check.in_range(18, 100)),
    }
    for column in PAY_STATUS_COLUMNS:
        columns[column] = _int_column(pa.Check.in_range(-2, 8))
    for column in BILL_COLUMNS:
        columns[column] = _numeric_column()
    for column in PAY_AMOUNT_COLUMNS:
        columns[column] = _numeric_column(pa.Check.ge(0))
    columns[target_column] = _int_column(pa.Check.isin([0, 1]))
    return pa.DataFrameSchema(columns, strict=False)


def validate_dataframe(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Проверяет обязательные колонки и допустимые значения.

    Возвращает проверенную таблицу. При ошибке выбрасывает DataValidationError.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("На вход ожидается pandas.DataFrame")

    try:
        target = target_column or detect_target(df.columns)
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc

    required = list(RAW_FEATURE_COLUMNS) + [target]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataValidationError(
            "Отсутствуют обязательные колонки: " + ", ".join(missing)
        )

    schema = build_schema(target)
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        details = exc.failure_cases.to_string(index=False)
        if len(details) > 2000:
            details = details[:2000] + "..."
        raise DataValidationError(
            "Данные не прошли проверку Pandera:\n" + details
        ) from exc
    except pa.errors.SchemaError as exc:
        raise DataValidationError(f"Данные не прошли проверку Pandera: {exc}") from exc


def _print_validation_result(title: str) -> None:
    print(title)


def main() -> None:
    """Проверяет синтетику, негативный пример и реальный CSV, если он есть."""
    from src.config import find_raw_csv
    from src.data.make_dataset import load_raw_csv
    from src.data.synthetic import make_synthetic_frame

    sample = make_synthetic_frame(100)
    validate_dataframe(sample)
    _print_validation_result("Синтетическая выборка: валидация пройдена")

    broken = sample.copy()
    broken.loc[0, "AGE"] = 200
    try:
        validate_dataframe(broken)
    except DataValidationError as exc:
        print("Негативная проверка: аномальный AGE отклонён")
        print(str(exc).splitlines()[0])
    else:
        raise SystemExit("Ожидалось исключение для аномального возраста")

    csv_path = find_raw_csv()
    if csv_path is None:
        print("Реальный CSV не найден. Проверена синтетическая схема.")
        return

    raw = load_raw_csv(csv_path)
    if "ID" in raw.columns:
        raw = raw.drop(columns=["ID"])
    validate_dataframe(raw)
    print(f"Реальный CSV прошёл валидацию: {csv_path}")


if __name__ == "__main__":
    main()
