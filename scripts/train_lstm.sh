#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export BASEBALL_DATA_ROOT="${BASEBALL_DATA_ROOT:-./data}"

echo "=========================================="
echo "LSTM Action Classification Training"
echo "=========================================="
echo "Data path: $BASEBALL_DATA_ROOT"
echo "=========================================="

python scripts/train_lstm.py
