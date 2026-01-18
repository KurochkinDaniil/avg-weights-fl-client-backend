#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FL Status Checker - проверка готовности к дообучению
"""
import sys
import io
from pathlib import Path
import torch

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_status():
    """Проверка статуса клиента FL"""
    print("=" * 60)
    print("[*] Federated Learning Status Check")
    print("=" * 60)
    print()
    
    status = {
        'has_data': False,
        'has_model': False,
        'total_swipes': 0,
        'data_files': []
    }
    
    # 1. Проверка данных
    print("[1] Training Data:")
    data_dir = Path("./data/raw")
    
    if data_dir.exists():
        jsonl_files = list(data_dir.glob("**/*.jsonl"))
        
        if jsonl_files:
            status['has_data'] = True
            status['data_files'] = jsonl_files
            
            for jsonl_file in jsonl_files:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    count = sum(1 for _ in f)
                    status['total_swipes'] += count
                    rel_path = jsonl_file.relative_to(data_dir.parent)
                    print(f"   ✓ {rel_path}: {count} swipes")
            
            print(f"   Total: {status['total_swipes']} swipes")
        else:
            print("   ✗ No JSONL files found")
    else:
        print(f"   ✗ Data directory not found: {data_dir}")
    
    print()
    
    # 2. Проверка модели
    print()
    print("[2] Pre-trained Model:")
    model_path = Path("model2.pt")
    
    if model_path.exists():
        status['has_model'] = True
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ {model_path} ({size_mb:.2f} MB)")
        
        # Попытка загрузить модель
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            num_layers = len(state_dict)
            print(f"   ✓ Valid PyTorch model ({num_layers} layers)")
            
            # Показать несколько ключей
            keys = list(state_dict.keys())[:5]
            print(f"   Sample layers: {', '.join(keys)}")
            
        except Exception as e:
            print(f"   ✗ Error loading model: {e}")
            status['has_model'] = False
    else:
        print(f"   ✗ Model file not found: {model_path}")
        print("   Will train from random initialization")
    
    print()
    
    # 3. Проверка конфигурации
    print()
    print("[3] Configuration:")
    try:
        from config import settings
        print(f"   ✓ Client ID: {settings.client_id}")
        print(f"   ✓ Server: {settings.server_grpc_url}")
        print(f"   ✓ Epochs: {settings.num_epochs}")
        print(f"   ✓ Batch size: {settings.batch_size}")
        print(f"   ✓ Learning rate: {settings.learning_rate}")
    except Exception as e:
        print(f"   ✗ Error loading config: {e}")
    
    print()
    
    # 4. Итоговый статус
    print()
    print("=" * 60)
    if status['has_data'] and status['total_swipes'] >= 10:
        print("[OK] READY TO TRAIN!")
        print(f"   You have {status['total_swipes']} swipes ready for training.")
        print()
        print("   Run: python fl_train.py")
        print("   Or:  bash fl_train.sh")
    elif status['has_data'] and status['total_swipes'] < 10:
        print("[WARN] LOW DATA WARNING")
        print(f"   You have only {status['total_swipes']} swipes.")
        print("   Collect at least 10-20 swipes for meaningful training.")
    else:
        print("[ERROR] NOT READY")
        print("   No training data found.")
        print("   Please collect swipes using the frontend first.")
    
    print("=" * 60)
    
    return status

if __name__ == "__main__":
    try:
        status = check_status()
        sys.exit(0 if status['has_data'] else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

