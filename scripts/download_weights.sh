#!/bin/bash

WEIGHTS_DIR="weights"
mkdir -p $WEIGHTS_DIR

echo "=========================================="
echo "Download Model Weights"
echo "=========================================="

echo "[1/3] YOLOv13 weights..."
echo "Download from GitHub Releases"

echo "[2/3] LSTM weights..."
echo "Download from GitHub Releases"

echo "[3/3] ViTPose weights (363MB)..."
echo "Download from Google Drive"

echo ""
echo "=========================================="
echo "Place weights in weights/ directory:"
echo "  - weights/yolov13s.pt"
echo "  - weights/lstm_best_model.pth"
echo "  - weights/vitpose_baseball.pth"
echo "=========================================="
