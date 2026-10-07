"""Колонки датасета UCI Default of Credit Card Clients и поиск target."""

from __future__ import annotations

ID_COLUMN = "ID"
TARGET_COLUMN = "default.payment.next.month"

PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{index}" for index in range(1, 7)]
PAY_AMOUNT_COLUMNS = [f"PAY_AMT{index}" for index in range(1, 7)]

RAW_FEATURE_COLUMNS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + PAY_STATUS_COLUMNS
    + BILL_COLUMNS
    + PAY_AMOUNT_COLUMNS
)

ENGINEERED_NUMERIC_COLUMNS = [
    "bill_amt_mean",
    "bill_amt_max",
    "pay_amt_mean",
    "pay_amt_max",
    "pay_to_bill_ratio",
    "delay_months_count",
    "max_delay",
    "mean_delay",
    "utilisation",
]

NUMERIC_FEATURES = (
    ["LIMIT_BAL", "AGE"]
    + PAY_STATUS_COLUMNS
    + BILL_COLUMNS
    + PAY_AMOUNT_COLUMNS
    + ENGINEERED_NUMERIC_COLUMNS
)

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE", "age_group"]
MODEL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def normalize_column_name(name: str) -> str:
    """Оставляет только буквы и цифры, чтобы узнать target при другом написании."""
    return "".join(char for char in str(name).lower() if char.isalnum())


def detect_target(columns, preferred: str | None = None) -> str:
    """Находит целевую колонку, даже если имя немного отличается от канонического."""
    column_list = [str(column).strip() for column in columns]
    if preferred and preferred in column_list:
        return preferred

    normalized = {normalize_column_name(column): column for column in column_list}
    canonical = normalize_column_name(TARGET_COLUMN)
    if canonical in normalized:
        return normalized[canonical]

    for column in column_list:
        if "default" in column.lower():
            return column

    raise ValueError(
        "Не удалось определить целевую колонку. "
        "Ожидалось имя вроде default.payment.next.month."
    )
