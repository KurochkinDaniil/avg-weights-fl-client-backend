# 🚀 Quick Start: New Architecture

## Запуск API

```bash
cd apps/client

# Активировать venv (если есть)
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Запустить API
python main.py
# или
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Использование API

### 1. Предсказание (синхронное)

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gesture_id": "test-123",
    "coords": [
      {"x": 49.5, "y": 84.5, "t": 0.0},
      {"x": 147.5, "y": 84.5, "t": 0.1}
    ],
    "word": ""
  }'

# Response:
{
  "gesture_id": "test-123",
  "predicted_word": "йц"
}
```

### 2. Сохранение свайпа (фоновое, мгновенный ответ)

```bash
curl -X POST http://localhost:8000/api/v1/swipes \
  -H "Content-Type: application/json" \
  -d '{
    "gesture_id": "swipe-456",
    "coords": [...],
    "word": "привет"
  }'

# Response (мгновенно ~5ms):
{
  "status": "accepted",
  "gesture_id": "swipe-456",
  "message": "Swipe gesture accepted, saving in background"
}
```

### 3. Запуск обучения (фоновое)

```bash
curl -X POST http://localhost:8000/api/v1/train

# Response (мгновенно):
{
  "status": "training_started",
  "message": "Federated learning training cycle started in background"
}

# Обучение идёт в фоне:
# - Скачивает веса с сервера (MinIO)
# - Обучает локально
# - Загружает дельту на сервер
# - Hot reload модели (без перезапуска API!)
```

### 4. Статистика

```bash
curl http://localhost:8000/api/v1/stats

# Response:
{
  "total_swipes": 17,
  "total_files": 3,
  "data_directory": "./data"
}
```

## Программное использование

### В Python коде:

```python
from services import PredictionService, StorageService, TrainingService
from core.model_manager import model_manager

# Prediction
prediction_service = PredictionService()
word = prediction_service.predict(coords)

# Storage
storage_service = StorageService()
storage_service.save_swipe(gesture_id, coords, word)

# Training (async)
training_service = TrainingService()
results = await training_service.run_training_cycle()

# Model hot reload
model_manager.reload_from_weights(new_weights)
```

## Интеграция с фронтендом

### Сохранение свайпов (фоновое):

```javascript
// frontend/demo/main.js

async function saveSwipe(coords, word) {
  // Отправка с мгновенным ответом
  const response = await fetch('http://localhost:8000/api/v1/swipes', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      gesture_id: crypto.randomUUID(),
      coords: coords,
      word: word
    })
  });
  
  // 202 Accepted - сохранение идёт в фоне
  console.log('Swipe accepted');
}
```

### Запуск обучения по кнопке:

```javascript
async function startTraining() {
  const response = await fetch('http://localhost:8000/api/v1/train', {
    method: 'POST'
  });
  
  const data = await response.json();
  console.log(data.message); // "Training started in background"
  
  // Обучение идёт в фоне, можно продолжать работать
}
```

## Hot Reload модели

После обучения модель автоматически обновляется:

```python
# До обучения
predict("йц") → "неизвестно" (старая модель)

# Запускается обучение (фон)
POST /api/v1/train

# После обучения (автоматически)
predict("йц") → "йц" (новая модель, без перезапуска API!)
```

## Мониторинг

### Логи:

```bash
# Запустить с логами
python main.py

# Вы увидите:
# 2026-01-18 20:00:00 - INFO - Starting up FL Client API...
# 2026-01-18 20:00:00 - INFO - Model loaded successfully
# 2026-01-18 20:00:05 - INFO - Accepted swipe: xxx, word: 'привет' (saving in background)
# 2026-01-18 20:01:00 - INFO - FL training cycle started in background
# 2026-01-18 20:01:05 - INFO - Training completed on 17 examples
# 2026-01-18 20:01:05 - INFO - Model hot reloaded successfully
```

### Health Check:

```bash
curl http://localhost:8000/health

# Response:
{"status": "healthy"}
```

## Отличия от старой версии

| Аспект | Старое | Новое |
|--------|--------|-------|
| **Сохранение** | Блокирует | Background (5ms) |
| **Обучение** | Ручной скрипт | API endpoint |
| **Модель** | Перезапуск API | Hot reload |
| **Ответ /swipes** | 201 Created | 202 Accepted |
| **Эндпоинт /train** | ❌ Нет | ✅ Есть |

## Что дальше?

1. **Интегрируйте с сервером**:
   - Настройте `server_grpc_url` в `config.py`
   - Запустите Go сервер
   - Веса будут скачиваться из MinIO

2. **Автоматическое обучение**:
   - Добавьте cron или APScheduler
   - Запускайте `/train` каждый час

3. **Мониторинг**:
   - Добавьте Prometheus metrics
   - Настройте Grafana dashboards

Всё работает из коробки! 🎉

