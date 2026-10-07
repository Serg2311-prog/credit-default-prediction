"""Общие настройки matplotlib для сохранения графиков."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def configure_plots() -> None:
    """Выбирает шрифт с кириллицей, если он установлен в системе."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Segoe UI", "Arial", "Tahoma", "DejaVu Sans"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
