#!/bin/bash
# Federated Learning Training Script
# Запускает один цикл дообучения: загрузка весов → обучение → отправка дельты

set -e

echo "======================================"
echo "Federated Learning Training Script"
echo "======================================"
echo

# Проверка наличия данных
DATA_DIR="./data/raw"
if [ ! -d "$DATA_DIR" ] || [ -z "$(find $DATA_DIR -name '*.jsonl' 2>/dev/null)" ]; then
    echo "❌ ERROR: No training data found in $DATA_DIR"
    echo "   Please collect some swipes first using the frontend."
    exit 1
fi

# Подсчет количества свайпов
SWIPES_COUNT=$(find $DATA_DIR -name '*.jsonl' -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
echo "📊 Found $SWIPES_COUNT swipes in local storage"
echo

# Проверка наличия модели
if [ ! -f "model2.pt" ]; then
    echo "⚠️  WARNING: No pre-trained model (model2.pt) found"
    echo "   Will train from random initialization"
    echo
fi

# Активация виртуального окружения
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "🔧 Activating virtual environment..."
    source ../venv/bin/activate
fi

# Запуск FL цикла
echo "🚀 Starting Federated Learning Cycle..."
echo
python scripts/federated_cycle.py

# Проверка результата
if [ $? -eq 0 ]; then
    echo
    echo "======================================"
    echo "✅ FL Cycle completed successfully!"
    echo "======================================"
else
    echo
    echo "======================================"
    echo "❌ FL Cycle failed"
    echo "======================================"
    exit 1
fi

