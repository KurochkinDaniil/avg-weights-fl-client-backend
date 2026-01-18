# Architecture Review: Production-Ready Improvements

## Текущая архитектура ✅

```
apps/client/
├── config.py              # ✅ Pydantic Settings с .env
├── main.py                # ✅ FastAPI app
├── api/
│   ├── routes.py         # ✅ API endpoints
│   └── models.py         # ✅ Pydantic models
├── ml/
│   ├── model.py          # ✅ SwipeLSTM модель
│   ├── trainer.py        # ✅ Обучение
│   ├── dataset.py        # ✅ PyTorch Dataset
│   ├── preprocessing.py  # ✅ Препроцессинг
│   └── inference.py      # ✅ Предсказания
├── storage/
│   └── local_storage.py  # ✅ JSONL хранилище
├── grpc_client/
│   └── fl_client.py      # ✅ gRPC клиент
└── scripts/
    └── federated_cycle.py # ✅ FL цикл
```

## Что уже хорошо

### ✅ Модульность
- **Separation of Concerns**: API, ML, Storage, gRPC разделены
- **Single Responsibility**: каждый модуль отвечает за свою задачу
- **Layered Architecture**: presentation (API) → business (ML) → data (storage)

### ✅ Конфигурация
- Pydantic Settings с `.env` поддержкой
- Type hints везде
- Централизованный config.py

### ✅ API
- FastAPI с OpenAPI docs
- CORS middleware
- Health checks
- Versioned endpoints (`/api/v1`)

### ✅ ML Pipeline
- Четкое разделение: model, trainer, dataset, inference
- PyTorch best practices
- Поддержка CPU/GPU

## Что нужно улучшить для Production

### 1. ❌ Dependency Injection

**Проблема**: Прямые import и создание объектов
```python
# Текущий код
storage = LocalStorage(settings.data_dir)
model = SwipeLSTM(...)
```

**Решение**: IoC Container + DI
```python
# Production-подход
class Container:
    storage = Provide[StorageService]
    model_service = Provide[ModelService]
```

### 2. ❌ Service Layer

**Проблема**: Бизнес-логика размазана по routes и scripts

**Решение**: Выделить service layer
```
services/
├── training_service.py    # FL обучение
├── prediction_service.py  # Предсказания
└── storage_service.py     # Управление данными
```

### 3. ❌ Structured Logging

**Проблема**: Простой logging.info
```python
logger.info("Training completed")  # Плохо для мониторинга
```

**Решение**: Structured logging + rotation
```python
logger.info("training.completed", extra={
    "duration_sec": 1.23,
    "loss": 26.1,
    "samples": 17
})
```

### 4. ❌ Error Handling & Retry

**Проблема**: Try-catch без retry и exponential backoff
```python
try:
    fl_client.upload_weights(...)
except Exception as e:
    logger.error(f"Failed: {e}")  # Просто логируем
```

**Решение**: Retry decorator + circuit breaker
```python
@retry(max_attempts=3, backoff=exponential)
async def upload_weights(...):
    ...
```

### 5. ❌ Metrics & Monitoring

**Проблема**: Нет метрик (Prometheus, Grafana)

**Решение**: Добавить метрики
```python
from prometheus_client import Counter, Histogram

training_duration = Histogram('training_duration_seconds')
predictions_total = Counter('predictions_total')
```

### 6. ❌ Testing

**Проблема**: Нет тестов (unit, integration)

**Решение**: pytest + fixtures
```
tests/
├── unit/
│   ├── test_model.py
│   └── test_preprocessing.py
└── integration/
    └── test_fl_cycle.py
```

### 7. ❌ Database for Metadata

**Проблема**: JSONL для всего (сложно запросы)

**Решение**: SQLite/PostgreSQL для метаданных
```
database/
├── models.py          # SQLAlchemy models
├── repository.py      # Repository pattern
└── migrations/        # Alembic migrations
```

### 8. ❌ Background Tasks

**Проблема**: FL цикл блокирует или запускается вручную

**Решение**: Celery/APScheduler для фоновых задач
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()
scheduler.add_job(run_fl_cycle, 'interval', hours=1)
```

### 9. ❌ Graceful Shutdown

**Проблема**: Нет graceful shutdown (может потерять данные)

**Решение**: Signal handlers
```python
@app.on_event("shutdown")
async def shutdown_event():
    await save_pending_data()
    await close_connections()
```

### 10. ❌ API Rate Limiting

**Проблема**: Нет защиты от DDoS

**Решение**: Rate limiting middleware
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)
@app.post("/predict")
@limiter.limit("10/minute")
async def predict(...):
```

## Production-Ready архитектура

```
apps/client/
├── config/
│   ├── settings.py        # Настройки
│   ├── logging.py         # Конфиг логирования
│   └── di_container.py    # DI контейнер
├── api/
│   ├── routes/            # Разделенные роуты
│   │   ├── health.py
│   │   ├── predictions.py
│   │   └── swipes.py
│   ├── dependencies.py    # FastAPI dependencies
│   └── middleware.py      # Custom middleware
├── core/
│   ├── exceptions.py      # Кастомные исключения
│   ├── logging.py         # Structured logging
│   └── metrics.py         # Prometheus metrics
├── services/
│   ├── training_service.py
│   ├── prediction_service.py
│   ├── storage_service.py
│   └── grpc_service.py
├── ml/
│   ├── models/            # ML модели
│   ├── training/          # Training logic
│   └── inference/         # Inference logic
├── storage/
│   ├── database/          # SQLAlchemy models
│   ├── repositories/      # Repository pattern
│   └── jsonl/             # JSONL storage
├── tasks/
│   ├── scheduler.py       # Background tasks
│   └── fl_cycle_task.py   # FL цикл как task
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.prod.yml
└── scripts/
    ├── migrate.py         # DB migrations
    └── seed.py            # Seed data
```

## Приоритеты внедрения

### P0 (Critical) - Сейчас
1. ✅ Structured logging
2. ✅ Service layer
3. ✅ Error handling + retry
4. ✅ Database для метаданных

### P1 (High) - Следующий спринт
5. Dependency Injection
6. Metrics (Prometheus)
7. Background tasks
8. Testing (unit + integration)

### P2 (Medium) - Через 2-3 спринта
9. Circuit breaker
10. Rate limiting
11. Graceful shutdown
12. API versioning strategy

## Рекомендации

### Начните с малого
Не переписывайте всё сразу. Внедряйте постепенно:
1. **Week 1**: Service layer + Structured logging
2. **Week 2**: Database + Repository pattern
3. **Week 3**: DI Container + Tests
4. **Week 4**: Metrics + Background tasks

### Используйте паттерны
- **Repository Pattern** для data access
- **Service Layer** для бизнес-логики
- **Dependency Injection** для loose coupling
- **Factory Pattern** для создания объектов

### Инструменты
```toml
[tool.poetry.dependencies]
# Текущие
fastapi = "^0.109.0"
uvicorn = "^0.27.0"
pydantic = "^2.5.0"
torch = "^2.1.2"

# Добавить для Production
# DI
dependency-injector = "^4.41.0"

# Logging
structlog = "^24.1.0"
python-json-logger = "^2.0.7"

# Metrics
prometheus-client = "^0.19.0"
prometheus-fastapi-instrumentator = "^6.1.0"

# Background tasks
apscheduler = "^3.10.4"
celery = "^5.3.4"  # для тяжелых задач

# Database
sqlalchemy = "^2.0.25"
alembic = "^1.13.1"
asyncpg = "^0.29.0"  # async PostgreSQL

# Retry & Circuit Breaker
tenacity = "^8.2.3"
circuitbreaker = "^2.0.0"

# Rate Limiting
slowapi = "^0.1.9"

# Testing
pytest = "^7.4.4"
pytest-asyncio = "^0.23.3"
pytest-cov = "^4.1.0"
faker = "^22.0.0"
```

## Следующие шаги

Хотите, чтобы я:
1. ✅ Реализовал Service Layer?
2. ✅ Добавил Structured Logging?
3. ✅ Создал Database + Repository pattern?
4. ✅ Настроил Dependency Injection?
5. ✅ Добавил Background tasks для FL?
6. ✅ Написал тесты?

Или начнём с чего-то конкретного?

