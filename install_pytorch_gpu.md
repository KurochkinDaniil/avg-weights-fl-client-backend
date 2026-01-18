# Установка PyTorch с GPU поддержкой (CUDA)

## Для RTX 2060 Super

Ваша карта поддерживает CUDA Compute Capability 7.5, что совместимо с CUDA 11.8 и 12.1.

## Вариант 1: Автоматическая установка (рекомендуется)

```bash
cd apps/client

# Создайте скрипт (если на Linux/Mac)
bash install_pytorch_gpu.sh

# Или вручную (Windows/Linux/Mac)
pip install install_pytorch_gpu.txt
```

## Вариант 2: Ручная установка

### Шаг 1: Удалить CPU-версию

```bash
pip uninstall torch torchvision torchaudio -y
```

### Шаг 2: Установить GPU-версию

**Для CUDA 11.8 (рекомендуется):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Для CUDA 12.1 (если CUDA 12.x уже установлена):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Шаг 3: Проверить установку

```bash
python check_gpu.py
```

Должно показать:
```
[OK] CUDA available: YES
[OK] CUDA version: 11.8 (или 12.1)
[OK] GPU 0: NVIDIA GeForce RTX 2060 SUPER
     Memory: 8.00 GB
```

## Вариант 3: С использованием requirements

Создайте файл `requirements_gpu.txt`:

```
# PyTorch с CUDA 11.8
torch==2.1.2+cu118
torchvision==0.16.2+cu118
torchaudio==2.1.2+cu118
--extra-index-url https://download.pytorch.org/whl/cu118

# Остальные зависимости
fastapi
uvicorn
pydantic
pydantic-settings
```

Установка:
```bash
pip install -r requirements_gpu.txt
```

## Проверка CUDA Toolkit (опционально)

Если у вас не установлен CUDA Toolkit:

### Windows:
1. Скачайте [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Установите (требует ~3GB места)
3. Перезагрузите компьютер

### Linux:
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

**Примечание**: PyTorch включает собственные CUDA библиотеки, поэтому установка CUDA Toolkit не обязательна для базового использования.

## Проверка драйверов NVIDIA

```bash
# Windows
nvidia-smi

# Linux
nvidia-smi
```

Должно показать:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 5xx.xx       Driver Version: 5xx.xx       CUDA Version: 11.x  |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 206... WDDM  | 00000000:01:00.0  On |                  N/A |
| 30%   40C    P8    15W / 175W |    500MB /  8192MB   |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

## Troubleshooting

### "CUDA not available" после установки

1. Проверьте версию PyTorch:
```bash
python -c "import torch; print(torch.__version__)"
```
Должно быть: `2.1.2+cu118` (не `2.1.2+cpu`)

2. Проверьте CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Должно быть: `True`

3. Переустановите:
```bash
pip cache purge
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "Out of memory" во время обучения

Уменьшите `batch_size` в `config.py`:
```python
batch_size: int = 16  # вместо 32
```

Или:
```python
batch_size: int = 8   # для очень больших моделей
```

### GPU не используется во время обучения

Проверьте логи:
```bash
python fl_train_simple.py
```

Должно быть:
```
Using device: cuda
```

Если показывает `cpu`, значит CUDA не доступна.

## После установки

1. **Проверьте GPU**:
```bash
python check_gpu.py
```

2. **Запустите обучение**:
```bash
python fl_train_simple.py
```

3. **Мониторьте GPU**:
В отдельном терминале:
```bash
# Windows/Linux
nvidia-smi -l 1  # обновление каждую секунду
```

Во время обучения вы увидите:
- GPU-Util: 90-100% (загрузка GPU)
- Memory-Usage: увеличится на 500-2000MB
- Temp: повысится до 60-75°C

## Ожидаемое ускорение

Для RTX 2060 Super (8GB VRAM):
- **CPU**: ~1-2 секунды на эпоху (17 свайпов)
- **GPU**: ~0.1-0.3 секунды на эпоху (10-20x быстрее)

Для больших датасетов (1000+ свайпов):
- **CPU**: 30-60 секунд на эпоху
- **GPU**: 2-5 секунд на эпоху

## Полезные команды

```bash
# Проверка GPU
python check_gpu.py

# Мониторинг GPU
nvidia-smi -l 1

# Проверка PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Обучение на GPU
python fl_train_simple.py

# Проверка использования памяти
python -c "import torch; print(f'Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB')"
```

## Готово!

После установки просто запускайте:
```bash
python fl_train_simple.py
```

Скрипт автоматически определит GPU и будет обучать на нем! 🚀

