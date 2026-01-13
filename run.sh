#!/bin/bash
# Simple run script for client API

cd "$(dirname "$0")"

# Activate venv (Windows or Unix)
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate  # Windows
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate      # Unix
fi

export PYTHONPATH="$(pwd):$PYTHONPATH"

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
