#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated Learning Training Script - Simple Version (no emojis)
"""
import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_data():
    """Check training data availability"""
    data_dir = Path("./data/raw")
    
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return False
    
    jsonl_files = list(data_dir.glob("**/*.jsonl"))
    
    if not jsonl_files:
        print(f"[ERROR] No training data found in {data_dir}")
        print("Please collect some swipes first using the frontend.")
        return False
    
    # Count swipes
    total_swipes = 0
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
            total_swipes += count
            rel_path = jsonl_file.relative_to(data_dir.parent)
            print(f"[DATA] {rel_path}: {count} swipes")
    
    print(f"[DATA] Total: {total_swipes} swipes")
    return True

def check_model():
    """Check pre-trained model"""
    model_path = Path("model2.pt")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"[MODEL] Found: {model_path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"[MODEL] Not found: {model_path}")
        print("[MODEL] Will train from random initialization")
        return False

def main():
    """Main entry point"""
    print("=" * 60)
    print("Federated Learning Training")
    print("=" * 60)
    print()
    
    # Pre-flight checks
    print("[1] Checking data...")
    if not check_data():
        sys.exit(1)
    
    print()
    print("[2] Checking model...")
    check_model()
    
    print()
    print("=" * 60)
    print("[3] Starting FL Cycle...")
    print("=" * 60)
    print()
    
    # Import and run
    try:
        from scripts.federated_cycle import run_federated_cycle
        run_federated_cycle()
        
        print()
        print("=" * 60)
        print("[OK] FL Cycle completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print()
        print("[WARN] Interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"[ERROR] FL Cycle failed: {e}")
        print("=" * 60)
        
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

