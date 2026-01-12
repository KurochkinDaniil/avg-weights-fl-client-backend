# Python Client Backend

FastAPI backend для FL клавиатуры.

## Запуск

```bash
# API сервер
./run.sh

# Обучение
./train.sh
```

## API

- `POST /api/v1/predict` - предсказать слово
- `POST /api/v1/swipes` - сохранить свайп для обучения
- `GET /api/v1/stats` - статистика данных
- `GET /health` - health check

## Структура

```
apps/client/
├── run.sh              # Запуск API
├── train.sh            # Запуск обучения
├── model2.pt           # Обученная модель
├── api/                # FastAPI endpoints
├── ml/                 # ML код (модель, датасет, обучение)
├── storage/            # Локальное хранилище (JSONL)
├── grpc_client/        # Клиент для Go сервера
└── scripts/            # Исполняемые скрипты
```

## Конфиг

```bash
# .env
NUM_EPOCHS=5
BATCH_SIZE=32
API_PORT=8000
CLIENT_ID=client-001
```

## FL цикл

`train.sh` выполняет:

1. Скачивает глобальные веса (если есть сервер)
2. Обучает на локальных данных
3. Вычисляет дельту `Δ = w_local - w_global`
4. Отправляет дельту на сервер
