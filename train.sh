#!/bin/bash

# Simple wrapper to run federated learning cycle

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv (Windows or Unix)
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate  # Windows
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate      # Unix
fi

# Run FL cycle
python scripts/federated_cycle.py
