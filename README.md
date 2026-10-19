# Credit Default Prediction — MLOps Project

Учебный end-to-end проект: прогноз вероятности дефолта клиента кредитной карты.
Пайплайн готовит данные, строит признаки, обучает несколько моделей, логирует эксперименты в MLflow, сохраняет лучшую модель и отдаёт её через FastAPI.

## 1. Описание проекта

Проект повторяет короткий MLOps-контур:

- проверка и подготовка данных;
- одинаковый feature engineering на обучении и на инференсе;
- sklearn `Pipeline`, который учит препроцессинг только на train;
- несколько экспериментов с подбором гиперпараметров по ROC-AUC;
- финальная оценка на отложенном test;
- сохранение модели, DVC-стадии, API, Docker, PSI и CI.

## 2. Бизнес-задача

Нужно заранее оценить, вернёт ли клиент кредит в следующем месяце.
Модель выдаёт вероятность дефолта. Порог решения по умолчанию — `0.5`.
Из-за дисбаланса классов для логистической регрессии и случайного леса используется `class_weight="balanced"`.
У градиентного бустинга sklearn такого параметра нет, поэтому он обучается на исходных частотах классов.

## 3. Датасет

Источник: UCI Default of Credit Card Clients, файл `UCI_Credit_Card.csv`.

- 30 000 клиентов;
- 25 колонок в исходном файле, включая технический `ID`;
- пропусков нет;
- дубликатов `ID` нет;
- 35 строк имеют одинаковый профиль после удаления `ID`. Это разные клиенты, строки сохранены.

Файл для локальной работы лежит в `data/raw/UCI_Credit_Card.csv`.
В Git он не входит: строка `data/raw/*.csv` есть в `.gitignore`.
`dvc.lock` хранит контрольную сумму этого файла как зависимость стадии `prepare`.
Удалённого DVC remote нет, поэтому после `git clone` CSV нужно снова положить в `data/raw/` вручную.
Если файл лежит только в родительской папке, скрипт подготовки его тоже найдёт.

Признаки:

| Колонка | Смысл |
| --- | --- |
| `LIMIT_BAL` | кредитный лимит |
| `SEX` | 1 — мужчина, 2 — женщина |
| `EDUCATION` | 1 — аспирантура, 2 — университет, 3 — школа, 4 — другое. В выгрузке также есть коды 0, 5, 6 |
| `MARRIAGE` | 1 — в браке, 2 — не в браке, 3 — другое. Код 0 тоже встречается |
| `AGE` | возраст |
| `PAY_0`, `PAY_2` … `PAY_6` | статус погашения. Колонки `PAY_1` в датасете нет |
| `BILL_AMT1` … `BILL_AMT6` | сумма счёта |
| `PAY_AMT1` … `PAY_AMT6` | сумма платежа |

## 4. Целевая переменная

`default.payment.next.month`

- `0` — дефолта нет, 23 364 клиента (77.88%);
- `1` — дефолт есть, 6 636 клиентов (22.12%).

Имя ищется автоматически: скрипт принимает и каноническое имя, и близкое написание вроде `default_payment_next_month`.

## 5. Архитектура

```text
UCI CSV
  -> проверка Pandera и train/test split
  -> sklearn Pipeline
       FeatureEngineer
       -> импутация и масштабирование / one-hot
       -> классификатор
  -> MLflow + models/credit_default_model.pkl
  -> FastAPI /predict на сырых полях клиента
```

`ID` и target не входят в признаки. Агрегаты `BILL_AMT*`, `PAY_AMT*` и `PAY_*` считаются внутри одной строки, без статистик всей выборки. `StandardScaler` и импутеры обучаются только на train внутри `Pipeline.fit`.

## 6. Структура репозитория

```text
credit-scoring-model/
├── data/raw/                # UCI_Credit_Card.csv
├── data/processed/          # train.csv, test.csv, data_report.json
├── models/                  # credit_default_model.pkl
├── artifacts/               # графики, метрики, PSI
├── notebooks/01_eda.ipynb
├── src/
│   ├── data/                # загрузка и Pandera
│   ├── features/            # агрегаты
│   ├── models/              # pipeline, train, evaluate, predict
│   ├── monitoring/drift.py  # PSI
│   ├── api/app.py
│   └── eda_report.py
├── tests/
├── .github/workflows/ci.yml
├── dvc.yaml
├── params.yaml
├── Dockerfile
└── docker-compose.yml
```

## 7. Установка

Windows:

```powershell
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Если команда `python` уже есть в PATH, вместо `py -3` используйте `python`.

Unix:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Дальше все команды выполняются из корня `credit-scoring-model` с активированным виртуальным окружением.

## 8. Подготовка данных

```bash
python -m src.data.make_dataset
```

Скрипт:

- загружает CSV;
- удаляет `ID` из признаков;
- находит target;
- считает пропуски, дубликаты и баланс классов;
- проверяет таблицу через Pandera;
- делает стратифицированный split `test_size=0.2`, `random_state=42`;
- сохраняет `data/processed/train.csv` и `data/processed/test.csv`.

На этой выгрузке получается 24 000 строк train и 6 000 строк test. Доля класса 1 в обеих частях — 0.2212.

Отдельная проверка схемы:

```bash
python -m src.data.validation
```

## 9. Обучение

```bash
python -m src.models.train
```

Подбор идёт по train через `GridSearchCV` и `RandomizedSearchCV`, метрика — ROC-AUC, 3 стратифицированных фолда.
Test в выборе модели не участвует: он только логируется.

Лучшая по CV ROC-AUC pipeline сохраняется в `models/credit_default_model.pkl`.
Сводка экспериментов: `artifacts/training_summary.json`.

Параметры лежат в `params.yaml`: `random_state`, `test_size`, число фолдов, сетки моделей.

## 10. Эксперименты MLflow

Эксперимент: `Credit_Default_Prediction`.

Каждый run пишет алгоритм, параметры, CV ROC-AUC, test ROC-AUC, Precision, Recall, F1, Accuracy, PR-AUC, ROC-кривую, PR-кривую, матрицу ошибок и саму sklearn Pipeline (`mlflow.sklearn.log_model`).

Локальное хранилище — папка `mlruns`. Интерфейс:

```bash
mlflow ui --backend-store-uri mlruns
```

Откройте http://localhost:5000 и выберите эксперимент `Credit_Default_Prediction`.

Один запуск `python -m src.models.train` создаёт 7 runs — это семь конфигураций ниже, а не четырнадцать разных экспериментов.
В папке `mlruns` сейчас 14 runs, потому что обучение выполнили дважды: прямой запуск и повтор внутри `dvc repro`.
Вторая семёрка повторяет те же алгоритмы и те же гиперпараметры.

1. `LogisticRegression`, `C=0.1`, GridSearchCV
2. `LogisticRegression`, `C=1.0`, GridSearchCV
3. `LogisticRegression`, `C=10.0`, GridSearchCV
4. `RandomForestClassifier`, `max_depth=6`, `n_estimators=80`, RandomizedSearchCV
5. `RandomForestClassifier`, `max_depth=10`, `n_estimators=80`, RandomizedSearchCV
6. `GradientBoostingClassifier`, `max_depth=2`, `n_estimators=40`, GridSearchCV
7. `GradientBoostingClassifier`, `max_depth=3`, `n_estimators=40`, GridSearchCV

## 11. Метрики

Финальная оценка сохранённой модели:

```bash
python -m src.models.evaluate
```

Считаются ROC-AUC, PR-AUC, Precision, Recall, F1 и Accuracy. Графики ROC, Precision-Recall и матрица ошибок сохраняются в `artifacts/`.

EDA:

```bash
python -m src.eda_report
```

Ноутбук `notebooks/01_eda.ipynb` вызывает тот же код. Графики лежат в `artifacts/eda/`.

## 12. DVC

DVC связывает исходный CSV, подготовленные таблицы, обучение и метрики. Стадии в `dvc.yaml`: `prepare`, `train`, `evaluate`.

Первичная инициализация в Git-репозитории:

```bash
git init
dvc init
```

Воспроизведение:

```bash
dvc repro
```

`dvc repro` заново выполняет подготовку, обучение и оценку, если изменились код, данные или `params.yaml`.
Большие результаты (`data/processed`, `models/*.pkl`, `artifacts`, `mlruns`) не коммитятся обычным Git. В `.gitignore` они исключены, а DVC хранит зависимости в `dvc.yaml` и `dvc.lock`.

Удалённый remote не настроен. `dvc push` — необязательный шаг: он нужен только после `dvc remote add`, если данные надо отправить на другое хранилище. Локально пайплайн воспроизводится командой `dvc repro` без push.

## 13. Тестирование

```bash
pytest -v
```

Покрыты загрузка данных, валидация, feature engineering, pipeline, предсказание и API.
Негативный тест подаёт возраст 200 и проверяет, что Pandera отклоняет таблицу.

Качество кода:

```bash
flake8 src tests
black --check src tests
```

## 14. FastAPI

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Эндпоинты:

- `GET /` — краткое описание;
- `GET /health` — статус и флаг загрузки модели;
- `POST /predict` — сырые поля одного клиента.

Пример тела запроса:

```json
{
  "LIMIT_BAL": 20000,
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
  "BILL_AMT1": 3913,
  "BILL_AMT2": 3102,
  "BILL_AMT3": 689,
  "BILL_AMT4": 0,
  "BILL_AMT5": 0,
  "BILL_AMT6": 0,
  "PAY_AMT1": 0,
  "PAY_AMT2": 689,
  "PAY_AMT3": 0,
  "PAY_AMT4": 0,
  "PAY_AMT5": 0,
  "PAY_AMT6": 0
}
```

Ответ для этого клиента, полученный локально от сохранённой модели:

```json
{
  "prediction": 1,
  "default_probability": 0.8534
}
```

Клиенту не нужно считать `bill_amt_mean`, `utilisation` и остальные агрегаты: это делает сохранённая pipeline.

Пакетное предсказание из CSV:

```bash
python -m src.models.predict --input data/processed/test.csv --output artifacts/predictions.csv
```

## 15. Docker

Dockerfile задаёт `WORKDIR /app`, копирует `requirements.txt`, `src`, `models` и `params.yaml`, выставляет `PYTHONPATH=/app` и запускает `uvicorn src.api.app:app` на порте 8000.
Файл `models/credit_default_model.pkl` попадает в образ, если он лежит в контексте сборки. В `.dockerignore` его нет.
После чистого `git clone` этого файла не будет: его нет в Git. Тогда контейнер стартует, а `/predict` ответит 503, пока модель не положат в `models/` и образ не пересоберут.

```bash
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api
```

Либо:

```bash
docker compose up --build
```

Swagger: http://localhost:8000/docs

Docker-образ credit-scoring-api успешно собран и протестирован локально. Контейнер запущен, Swagger UI доступен по адресу http://localhost:8000/docs. Запрос POST /predict вернул HTTP 200 с результатом prediction=1 и default_probability=0.8534. Для работы предсказаний файл models/credit_default_model.pkl должен присутствовать при сборке образа.

## 16. Monitoring / PSI

```bash
python -m src.monitoring.drift
```

Скрипт берёт train как baseline и случайные 50% test как новые данные.
PSI считается для `LIMIT_BAL`, `AGE`, `PAY_0`, `BILL_AMT1`, `PAY_AMT1`, `utilisation`, категорий `EDUCATION`, `SEX`, `MARRIAGE` и вероятности модели.

Интерпретация:

- PSI < 0.1 — значимого drift нет;
- 0.1 <= PSI < 0.25 — умеренный drift;
- PSI >= 0.25 — существенный drift.

Нулевые доли в бинах заменяются малым числом, чтобы не делить на ноль.
На случайном split из того же датасета PSI должен быть низким: это проверка реализации, а не боевой сдвиг популяции.
Отчёт: `artifacts/psi_report.json`.

## 17. CI/CD

Файл `.github/workflows/ci.yml` запускается на `push` и `pull_request`.
Шаги: checkout, Python 3.11, установка `requirements.txt`, flake8, black --check, pytest, `python -m src.data.validation`.
Секреты не нужны. Валидация в CI использует синтетическую таблицу той же схемы и негативный пример с аномальным возрастом. Реальный CSV в Git не хранится.

GitHub Actions CI успешно выполнен для ветки main (commit c6421b2). Workflow проверил тесты, соответствие Black и Flake8, а также валидацию данных. Все шаги завершились успешно.

## 18. Результаты

Production-модель — `RandomForestClassifier` с параметрами `n_estimators=80`, `max_depth=10`, `min_samples_leaf=1` и `class_weight="balanced"`.
Её выбрал `RandomizedSearchCV`. Файл: `models/credit_default_model.pkl`.

Метрики:

- CV ROC-AUC = 0.7837
- Test ROC-AUC = 0.7771
- PR-AUC = 0.5536
- Precision = 0.4786
- Recall = 0.6081
- F1 = 0.5357
- Accuracy = 0.7668

Матрица ошибок на test, 6 000 строк:

- TN = 3794
- FP = 879
- FN = 520
- TP = 807

Модель выбрана по CV ROC-AUC на train. Test использовался только для финальной оценки.
Так сделано потому, что test — это одна отложенная проверка уже выбранной модели. Если сравнивать модели по test, эта проверка перестаёт быть независимой: выбор подстраивается под ту же выборку, по которой потом пишется отчёт. Кросс-валидация на train даёт оценку до того, как test открывают.

У градиентного бустинга с `max_depth=3` test ROC-AUC равен 0.7794, а CV ROC-AUC — 0.7834.
У выбранного леса CV ROC-AUC выше: 0.7837. Поэтому в файл production записан лес, хотя на test бустинг чуть выше.
Логистическая регрессия с балансировкой классов даёт CV ROC-AUC около 0.761.

PSI между train и половиной test по всем проверенным признакам и по вероятности ниже 0.01.

## 19. Как полностью воспроизвести проект

```bash
git clone https://github.com/Serg2311-prog/credit-default-prediction.git
cd credit-default-prediction
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
python -m pip install -r requirements.txt
dvc repro
python -m src.eda_report
python -m src.monitoring.drift
pytest -v
flake8 src tests
black --check src tests
mlflow ui --backend-store-uri mlruns
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Unix:

```bash
source .venv/bin/activate
pip install -r requirements.txt
dvc repro
python -m src.eda_report
python -m src.monitoring.drift
pytest -v
mlflow ui --backend-store-uri mlruns
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

`dvc repro` требует, чтобы `python` указывал на окружение с установленными зависимостями, и чтобы `data/raw/UCI_Credit_Card.csv` был на месте.

Перед первым запуском скачайте датасет UCI Default of Credit Card Clients и поместите файл UCI_Credit_Card.csv в каталог data/raw/. Исходный CSV не включён в Git-репозиторий, поэтому после клонирования его необходимо добавить вручную.
