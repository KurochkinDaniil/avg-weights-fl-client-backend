# Federated Learning Training Guide

## Процесс дообучения клиента

### 1. Сбор данных

Используйте фронтенд для сбора свайпов:
```bash
cd apps/frontend/demo
# Откройте index.html в браузере
# Делайте свайпы, исправляйте предсказания, сохраняйте
```

Данные сохраняются в `apps/client/data/raw/YYYY-MM-DD/swipes.jsonl`

### 2. Проверка готовности

```bash
cd apps/client
python fl_status.py
```

Должно показать:
- ✓ Training Data: количество свайпов
- ✓ Pre-trained Model: model2.pt
- ✓ Configuration: настройки клиента

### 3. Запуск дообучения

**Вариант 1: Python (рекомендуется для Windows)**
```bash
cd apps/client
python fl_train.py
```

**Вариант 2: Bash (для Linux/Mac)**
```bash
cd apps/client
bash fl_train.sh
```

**Вариант 3: Прямой запуск скрипта**
```bash
cd apps/client
python scripts/federated_cycle.py
```

### 4. Процесс дообучения

FL цикл состоит из 5 шагов:

```
┌──────────────────────────────────────────┐
│ Step 1: Download global weights         │
│ ↓                                        │
│ Step 2: Load local data (JSONL)         │
│ ↓                                        │
│ Step 3: Train model locally (N epochs)  │
│ ↓                                        │
│ Step 4: Compute delta (trained - global)│
│ ↓                                        │
│ Step 5: Upload delta to server          │
└──────────────────────────────────────────┘
```

### 5. Что происходит

1. **Download global weights** (Step 1):
   - Подключается к gRPC серверу `localhost:50051`
   - Запрашивает последние глобальные веса
   - Если сервер недоступен → обучение с нуля

2. **Load local data** (Step 2):
   - Читает все `*.jsonl` файлы из `data/raw/`
   - Применяет preprocessing: вычисляет vx, vy, ax, ay, нормализует
   - Создает PyTorch Dataset

3. **Train model** (Step 3):
   - Обучает модель на локальных данных
   - Использует CTC Loss для sequence-to-sequence
   - Параметры из `config.py`:
     - `num_epochs`: 5 (по умолчанию)
     - `batch_size`: 32
     - `learning_rate`: 0.001

4. **Compute delta** (Step 4):
   - Вычисляет разницу: `delta = trained_weights - global_weights`
   - Это то, что мы отправляем на сервер (экономим трафик)

5. **Upload delta** (Step 5):
   - Отправляет дельту на сервер через gRPC
   - Сервер аггрегирует дельты от всех клиентов
   - Создает новые глобальные веса: FedAvg

### 6. Конфигурация

Файл: `apps/client/config.py`

```python
# Client Configuration
client_id: str = "client-001"              # Уникальный ID клиента
server_grpc_url: str = "localhost:50051"   # Адрес gRPC сервера

# Training Configuration
batch_size: int = 32                        # Размер батча
learning_rate: float = 0.001               # Learning rate
num_epochs: int = 5                        # Количество эпох

# Model Configuration
max_seq_len: int = 300                     # Макс. длина последовательности
input_size: int = 7                        # x, y, dt, vx, vy, ax, ay
hidden_size: int = 512                     # Размер LSTM hidden state
alphabet_size: int = 40                    # Размер алфавита
```

### 7. Логи

Во время обучения вы увидите:

```
============================================================
Starting Federated Learning Cycle
============================================================

[Step 1/5] Downloading global weights from server...
✓ Loaded global weights

[Step 2/5] Loading local data...
✓ Loaded 15 samples from 3 files

[Step 3/5] Training model on local data...
Epoch 1/5, Loss: 2.3456
Epoch 2/5, Loss: 1.8765
Epoch 3/5, Loss: 1.4321
Epoch 4/5, Loss: 1.1234
Epoch 5/5, Loss: 0.9876
✓ Training completed on 15 examples

[Step 4/5] Computing weight delta...
✓ Computed delta with 8 layers

[Step 5/5] Uploading delta to server...
✓ Successfully uploaded delta to server

============================================================
Federated Learning Cycle Completed
============================================================
```

### 8. Troubleshooting

#### "No training data found"
```bash
# Проверьте наличие данных
ls -R apps/client/data/raw/
```

Если пусто → соберите свайпы через фронтенд.

#### "Could not connect to server"
```bash
# Проверьте, запущен ли gRPC сервер
# В другом терминале:
cd apps/server
go run cmd/server/main.go
```

Если сервер не нужен → FL цикл продолжит обучение локально.

#### "Error loading model2.pt"
```bash
# Проверьте модель
python -c "import torch; print(torch.load('model2.pt', map_location='cpu').keys())"
```

Если модель битая → удалите её, будет обучение с нуля.

#### "Out of memory"
Уменьшите `batch_size` в `config.py`:
```python
batch_size: int = 16  # вместо 32
```

### 9. Режим работы без сервера

Если сервер недоступен, FL цикл всё равно:
1. ✓ Загрузит предобученную модель из `model2.pt`
2. ✓ Обучит на локальных данных
3. ✓ Вычислит дельту
4. ✗ Не сможет отправить на сервер (но дельта будет сохранена)

Это полезно для отладки и тестирования.

### 10. Следующие шаги

После успешного дообучения:
1. Проверьте качество модели:
   ```bash
   python -m api.routes  # Запустите API
   # Протестируйте через фронтенд
   ```

2. Повторите FL цикл:
   - Соберите новые свайпы
   - Запустите `python fl_train.py` снова
   - Сервер аггрегирует дельты

3. Посмотрите статистику:
   ```bash
   curl http://localhost:8000/api/v1/stats
   ```

## Полезные команды

```bash
# Статус клиента
python fl_status.py

# Запуск FL цикла
python fl_train.py

# Запуск API сервера
python main.py

# Тест предсказания
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @test_swipe.json

# Просмотр логов
tail -f logs/client.log
```

