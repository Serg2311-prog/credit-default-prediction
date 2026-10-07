.PHONY: install prepare eda train evaluate drift test lint api

install:
	python -m pip install -r requirements.txt

prepare:
	python -m src.data.make_dataset

eda:
	python -m src.eda_report

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

drift:
	python -m src.monitoring.drift

test:
	pytest -v

lint:
	flake8 src tests
	black --check src tests

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000
