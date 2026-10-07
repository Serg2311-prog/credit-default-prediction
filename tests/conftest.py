"""Общие фикстуры тестов."""

import pytest

from src.data.synthetic import make_synthetic_frame


@pytest.fixture
def raw_frame():
    return make_synthetic_frame(80, seed=42)
