#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated Learning Training Script (standalone version)
Запускает один цикл дообучения: загрузка весов → обучение → отправка дельты
"""
import sys
import io
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Цвета для консоли
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text, color=''):
    """Print colored text (с поддержкой Windows через colorama если доступен)"""
    try:
        import colorama
        colorama.init()
    except ImportError:
        pass
    print(f"{color}{text}{Colors.ENDC}")

def check_data():
    """Проверка наличия данных для обучения"""
    data_dir = Path("./data/raw")
    
    if not data_dir.exists():
        print_colored(f"❌ ERROR: Data directory not found: {data_dir}", Colors.FAIL)
        return False
    
    jsonl_files = list(data_dir.glob("**/*.jsonl"))
    
    if not jsonl_files:
        print_colored(f"❌ ERROR: No training data found in {data_dir}", Colors.FAIL)
        print("   Please collect some swipes first using the frontend.")
        return False
    
    # Подсчет количества свайпов
    total_swipes = 0
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            total_swipes += sum(1 for _ in f)
    
    print_colored(f"[DATA] Found {total_swipes} swipes in {len(jsonl_files)} files", Colors.OKBLUE)
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        print(f"   - {jsonl_file.relative_to(data_dir.parent)}: {count} swipes")
    
    return True

def check_model():
    """Проверка наличия предобученной модели"""
    model_path = Path("model2.pt")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print_colored(f"✓ Pre-trained model found: {model_path} ({size_mb:.2f} MB)", Colors.OKGREEN)
        return True
    else:
        print_colored("⚠️  WARNING: No pre-trained model (model2.pt) found", Colors.WARNING)
        print("   Will train from random initialization")
        return False

def main():
    """Main entry point"""
    print("=" * 60)
    print_colored("Federated Learning Training Script", Colors.BOLD)
    print("=" * 60)
    print()
    
    # Проверки
    print("🔍 Pre-flight checks...")
    print()
    
    if not check_data():
        sys.exit(1)
    
    check_model()
    print()
    
    # Запуск FL цикла
    print_colored("🚀 Starting Federated Learning Cycle...", Colors.HEADER)
    print("=" * 60)
    print()
    
    # Импортируем и запускаем
    try:
        from scripts.federated_cycle import run_federated_cycle
        run_federated_cycle()
        
        print()
        print("=" * 60)
        print_colored("✅ FL Cycle completed successfully!", Colors.OKGREEN)
        print("=" * 60)
        
    except KeyboardInterrupt:
        print()
        print_colored("⚠️  Interrupted by user", Colors.WARNING)
        sys.exit(130)
        
    except Exception as e:
        print()
        print("=" * 60)
        print_colored(f"❌ FL Cycle failed: {e}", Colors.FAIL)
        print("=" * 60)
        
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

