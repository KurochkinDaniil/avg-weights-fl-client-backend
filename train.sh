#!/bin/bash

# Simple wrapper to run federated learning cycle

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run FL cycle
python scripts/federated_cycle.py
