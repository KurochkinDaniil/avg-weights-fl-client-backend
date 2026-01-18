#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save trained model after FL cycle
Сохраняет обученную модель для использования в API
"""
import sys
import io
from pathlib import Path
import torch
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from config import settings
from ml.model import SwipeLSTM

def save_model():
    """Save trained model"""
    print("=" * 60)
    print("Save Trained Model")
    print("=" * 60)
    print()
    
    # Initialize model
    model = SwipeLSTM(
        input_size=settings.input_size,
        hidden_size=settings.hidden_size,
        alphabet_size=settings.alphabet_size
    )
    
    # Load trained weights (from FL cycle)
    try:
        trained_weights = torch.load("model2.pt", map_location='cpu')
        model.load_state_dict(trained_weights)
        print("[OK] Loaded model from model2.pt")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"model_trained_{timestamp}.pt"
    
    torch.save(model.state_dict(), output_path)
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    
    print(f"[OK] Saved trained model to: {output_path} ({size_mb:.2f} MB)")
    print()
    print("To use this model in API:")
    print(f"  1. Copy: cp {output_path} model2.pt")
    print("  2. Restart API: python main.py")
    print()
    
    return True

if __name__ == "__main__":
    try:
        if save_model():
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

