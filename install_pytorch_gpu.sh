#!/bin/bash
# Quick PyTorch GPU Installation Script

set -e

echo "======================================"
echo "PyTorch GPU Installation"
echo "======================================"
echo

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found!"
    echo "        Please install NVIDIA drivers first."
    exit 1
fi

echo "[1] Checking NVIDIA GPU..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo

echo "[2] Uninstalling CPU version of PyTorch..."
pip uninstall torch torchvision torchaudio -y
echo

echo "[3] Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo

echo "[4] Verifying installation..."
python check_gpu.py

if [ $? -eq 0 ]; then
    echo
    echo "======================================"
    echo "[SUCCESS] GPU setup complete!"
    echo "======================================"
    echo
    echo "You can now train on GPU:"
    echo "  python fl_train_simple.py"
else
    echo
    echo "======================================"
    echo "[ERROR] GPU setup failed"
    echo "======================================"
    exit 1
fi

