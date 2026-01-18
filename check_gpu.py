#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Check - проверка доступности CUDA и GPU
"""
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_gpu():
    """Check GPU availability and info"""
    print("=" * 60)
    print("GPU Availability Check")
    print("=" * 60)
    print()
    
    # 1. Check PyTorch
    print("[1] PyTorch Installation:")
    try:
        import torch
        print(f"   [OK] PyTorch version: {torch.__version__}")
    except ImportError:
        print("   [ERROR] PyTorch not installed!")
        return False
    
    print()
    
    # 2. Check CUDA
    print("[2] CUDA Support:")
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print(f"   [OK] CUDA available: YES")
        print(f"   [OK] CUDA version: {torch.version.cuda}")
        print(f"   [OK] cuDNN version: {torch.backends.cudnn.version()}")
    else:
        print(f"   [WARN] CUDA available: NO")
        print()
        print("   This means PyTorch was installed without GPU support.")
        print("   To enable GPU, reinstall PyTorch with CUDA:")
        print()
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print()
        return False
    
    print()
    
    # 3. GPU Info
    print("[3] GPU Information:")
    num_gpus = torch.cuda.device_count()
    print(f"   [OK] Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"   [OK] GPU {i}: {gpu_name}")
        print(f"        Memory: {gpu_memory:.2f} GB")
    
    print()
    
    # 4. Test GPU
    print("[4] GPU Test:")
    try:
        # Create tensor on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        
        print("   [OK] GPU computation test: PASSED")
        print(f"   [OK] Result shape: {z.shape}")
        print(f"   [OK] Device: {z.device}")
        
        # Cleanup
        del x, y, z
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   [ERROR] GPU test failed: {e}")
        return False
    
    print()
    
    # 5. Memory
    print("[5] GPU Memory:")
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / (1024**2)
        reserved = torch.cuda.memory_reserved(i) / (1024**2)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**2)
        
        print(f"   GPU {i}:")
        print(f"      Allocated: {allocated:.2f} MB")
        print(f"      Reserved:  {reserved:.2f} MB")
        print(f"      Total:     {total:.2f} MB")
    
    print()
    print("=" * 60)
    print("[SUCCESS] GPU is ready for training!")
    print("=" * 60)
    print()
    print("To use GPU for training, just run:")
    print("  python fl_train_simple.py")
    print()
    print("The script will automatically detect and use GPU.")
    print()
    
    return True

if __name__ == "__main__":
    try:
        success = check_gpu()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

