#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=========================================="
echo "YOLOv13 Training"
echo "=========================================="

python src/detection/train.py
