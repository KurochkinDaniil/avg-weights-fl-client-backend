#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick PyTorch GPU Installation Script (Windows/Linux/Mac)
"""
import sys
import io
import subprocess
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def run_command(cmd, check=True):
    """Run shell command"""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if check and result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(1)
    
    return result.returncode == 0

def main():
    print("=" * 60)
    print("PyTorch GPU Installation")
    print("=" * 60)
    print()
    
    # 1. Check NVIDIA GPU
    print("[1] Checking NVIDIA GPU...")
    has_nvidia = run_command("nvidia-smi", check=False)
    
    if not has_nvidia:
        print("[ERROR] nvidia-smi not found!")
        print("        Please install NVIDIA drivers first.")
        print()
        print("Download from: https://www.nvidia.com/Download/index.aspx")
        sys.exit(1)
    
    print()
    
    # 2. Uninstall CPU version
    print("[2] Uninstalling CPU version of PyTorch...")
    print("    This may take a minute...")
    run_command("pip uninstall torch torchvision torchaudio -y")
    print()
    
    # 3. Install GPU version
    print("[3] Installing PyTorch with CUDA 11.8...")
    print("    This will download ~2GB of packages...")
    print()
    
    install_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    run_command(install_cmd)
    print()
    
    # 4. Verify
    print("[4] Verifying installation...")
    print()
    
    # Run check_gpu.py if it exists
    if os.path.exists("check_gpu.py"):
        success = run_command("python check_gpu.py", check=False)
    else:
        # Quick inline check
        print("Running quick check...")
        check_code = """
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
"""
        result = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True)
        print(result.stdout)
        success = "CUDA available: True" in result.stdout
    
    print()
    
    if success:
        print("=" * 60)
        print("[SUCCESS] GPU setup complete!")
        print("=" * 60)
        print()
        print("Your RTX 2060 Super is ready for training!")
        print()
        print("To train on GPU, just run:")
        print("  python fl_train_simple.py")
        print()
        print("The script will automatically use GPU (10-20x faster than CPU)")
    else:
        print("=" * 60)
        print("[ERROR] GPU setup failed")
        print("=" * 60)
        print()
        print("Please check:")
        print("  1. NVIDIA drivers are installed (nvidia-smi works)")
        print("  2. PyTorch version contains +cu118 (not +cpu)")
        print()
        print("If issue persists, try manual installation:")
        print("  See install_pytorch_gpu.md for details")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("[WARN] Installation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print()
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

