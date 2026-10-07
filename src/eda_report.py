"""Воспроизводимый EDA: таблицы в консоль и графики в artifacts/eda."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import artifacts_dir, find_raw_csv, load_params
from src.data.make_dataset import load_raw_csv
from src.plot_utils import configure_plots, plt
from src.schema import (
    BILL_COLUMNS,
    PAY_AMOUNT_COLUMNS,
    PAY_STATUS_COLUMNS,
    detect_target,
)


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return path


def _bar(series: pd.Series, title: str, xlabel: str, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    series.plot(kind="bar", ax=axis, color="#3b6ea5")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Количество")
    return _save(figure, path)


def _default_rate(df: pd.DataFrame, column: str, target: str, path: Path) -> Path:
    rate = df.groupby(column, observed=False)[target].mean().sort_index()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    rate.plot(kind="bar", ax=axis, color="#b85c38")
    axis.set_title(f"Доля дефолта по {column}")
    axis.set_xlabel(column)
    axis.set_ylabel("Доля класса 1")
    axis.set_ylim(0, 1)
    return _save(figure, path)


def run_eda(csv_path: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    configure_plots()
    source = csv_path or find_raw_csv()
    if source is None:
        raise FileNotFoundError("Не найден UCI_Credit_Card.csv для EDA")

    df = load_raw_csv(source)
    params = load_params()
    target = detect_target(df.columns, preferred=params["prepare"]["target_column"])
    destination = output_dir or (artifacts_dir() / "eda")
    saved: list[Path] = []

    print(f"Файл: {source}")
    print(f"Размер: {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"Target: {target}")
    print(f"Пропуски: {int(df.isnull().sum().sum())}")
    print(f"Дубликаты строк: {int(df.duplicated().sum())}")
    print("Типы признаков:")
    print(df.dtypes.to_string())
    print("Баланс target:")
    print(df[target].value_counts(dropna=False).to_string())

    saved.append(
        _bar(
            df[target].value_counts().sort_index(),
            "Распределение target",
            target,
            destination / "target_distribution.png",
        )
    )

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(df["AGE"], bins=20, color="#3b6ea5", edgecolor="white")
    axis.set_title("Распределение AGE")
    axis.set_xlabel("AGE")
    axis.set_ylabel("Количество")
    saved.append(_save(figure, destination / "age_distribution.png"))

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(df["LIMIT_BAL"], bins=30, color="#3b6ea5", edgecolor="white")
    axis.set_title("Распределение LIMIT_BAL")
    axis.set_xlabel("LIMIT_BAL")
    axis.set_ylabel("Количество")
    saved.append(_save(figure, destination / "limit_bal_distribution.png"))

    saved.append(
        _bar(
            df["EDUCATION"].value_counts().sort_index(),
            "Распределение EDUCATION",
            "EDUCATION",
            destination / "education_distribution.png",
        )
    )
    saved.append(
        _bar(
            df["MARRIAGE"].value_counts().sort_index(),
            "Распределение MARRIAGE",
            "MARRIAGE",
            destination / "marriage_distribution.png",
        )
    )
    saved.append(
        _bar(
            df["PAY_0"].value_counts().sort_index(),
            "Распределение PAY_0",
            "PAY_0",
            destination / "pay_0_distribution.png",
        )
    )

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.boxplot([df[column] for column in BILL_COLUMNS], showfliers=False)
    axis.set_xticks(range(1, len(BILL_COLUMNS) + 1))
    axis.set_xticklabels(BILL_COLUMNS, rotation=30)
    axis.set_title("BILL_AMT1–BILL_AMT6 без крайних выбросов")
    saved.append(_save(figure, destination / "bill_amt_boxplots.png"))

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.boxplot([df[column] for column in PAY_AMOUNT_COLUMNS], showfliers=False)
    axis.set_xticks(range(1, len(PAY_AMOUNT_COLUMNS) + 1))
    axis.set_xticklabels(PAY_AMOUNT_COLUMNS, rotation=30)
    axis.set_title("PAY_AMT1–PAY_AMT6 без крайних выбросов")
    saved.append(_save(figure, destination / "pay_amt_boxplots.png"))

    numeric = df.select_dtypes(include="number")
    correlation = numeric.corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)))
    axis.set_yticks(range(len(correlation.columns)))
    axis.set_xticklabels(correlation.columns, rotation=90, fontsize=7)
    axis.set_yticklabels(correlation.columns, fontsize=7)
    axis.set_title("Корреляции числовых признаков")
    figure.colorbar(image, ax=axis, fraction=0.046)
    saved.append(_save(figure, destination / "correlation_heatmap.png"))

    saved.append(
        _default_rate(
            df, "EDUCATION", target, destination / "default_rate_by_education.png"
        )
    )
    saved.append(
        _default_rate(
            df, "MARRIAGE", target, destination / "default_rate_by_marriage.png"
        )
    )
    saved.append(
        _default_rate(df, "PAY_0", target, destination / "default_rate_by_pay_0.png")
    )

    age_bins = pd.cut(df["AGE"], bins=[20, 25, 35, 45, 60, 80], include_lowest=True)
    rate = df.groupby(age_bins, observed=False)[target].mean()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    rate.plot(kind="bar", ax=axis, color="#b85c38")
    axis.set_title("Доля дефолта по возрастным группам")
    axis.set_xlabel("AGE")
    axis.set_ylabel("Доля класса 1")
    axis.set_ylim(0, 1)
    saved.append(_save(figure, destination / "default_rate_by_age.png"))

    figure, axis = plt.subplots(figsize=(10, 4.5))
    pay_means = df.groupby(target)[PAY_STATUS_COLUMNS].mean().T
    pay_means.plot(kind="bar", ax=axis)
    axis.set_title("Средний статус PAY_* в разрезе target")
    axis.set_xlabel("Месяц")
    axis.set_ylabel("Средний код PAY")
    axis.legend(title=target)
    saved.append(_save(figure, destination / "pay_status_by_target.png"))

    print(f"Сохранено графиков: {len(saved)}")
    for path in saved:
        print(path)
    return saved


def main() -> None:
    run_eda()


if __name__ == "__main__":
    main()
