"""Небольшой синтетический набор с той же схемой, что и UCI CSV.

Нужен для тестов и шага валидации в CI, когда исходный файл не лежит в репозитории.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import (
    BILL_COLUMNS,
    PAY_AMOUNT_COLUMNS,
    PAY_STATUS_COLUMNS,
    TARGET_COLUMN,
)


def make_synthetic_frame(
    n: int = 80,
    seed: int = 42,
    positive_rate: float = 0.25,
) -> pd.DataFrame:
    if n < 8:
        raise ValueError("Нужно минимум 8 строк, чтобы оба класса были представлены")

    rng = np.random.default_rng(seed)
    n_positive = min(n - 2, max(2, int(round(n * positive_rate))))
    target = np.array([1] * n_positive + [0] * (n - n_positive), dtype=int)
    rng.shuffle(target)

    pay_0 = rng.integers(-2, 3, size=n)
    delayed = rng.integers(1, 4, size=n)
    pay_0 = np.where(target == 1, delayed, pay_0)

    data: dict[str, np.ndarray] = {
        "LIMIT_BAL": rng.uniform(10_000, 300_000, size=n).round(2),
        "SEX": rng.integers(1, 3, size=n),
        "EDUCATION": rng.integers(1, 5, size=n),
        "MARRIAGE": rng.integers(1, 4, size=n),
        "AGE": rng.integers(21, 70, size=n),
        "PAY_0": pay_0.astype(int),
    }
    for column in PAY_STATUS_COLUMNS[1:]:
        data[column] = rng.integers(-2, 3, size=n)
    for column in BILL_COLUMNS:
        data[column] = rng.normal(40_000, 15_000, size=n).round(2)
    for column in PAY_AMOUNT_COLUMNS:
        data[column] = rng.uniform(0, 20_000, size=n).round(2)

    frame = pd.DataFrame(data)
    frame[TARGET_COLUMN] = target
    return frame
