#!/bin/bash
# Simple run script for client API

cd "$(dirname "$0")"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="$(pwd):$PYTHONPATH"

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
