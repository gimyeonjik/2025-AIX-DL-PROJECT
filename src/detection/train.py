#!/usr/bin/env python3

import sys
import os
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent.parent

WEIGHTS_DIR = PROJECT_ROOT / 'weights'
WEIGHTS_DIR.mkdir(exist_ok=True)
WEIGHTS_PATH = WEIGHTS_DIR / 'yolov13s.pt'
WEIGHTS_URL = 'https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13s.pt'

if not WEIGHTS_PATH.exists():
    print(f"Downloading YOLOv13s weights from {WEIGHTS_URL}...")
    import urllib.request
    urllib.request.urlretrieve(WEIGHTS_URL, str(WEIGHTS_PATH))
    print(f"Weights saved to {WEIGHTS_PATH}")

model = YOLO(str(WEIGHTS_PATH))

DATA_CONFIG = PROJECT_ROOT / 'configs' / 'yolov13' / 'baseball_player.yaml'
OUTPUT_DIR = PROJECT_ROOT / 'runs' / 'train'

model.train(
    data=str(DATA_CONFIG),
    epochs=100,
    imgsz=640,
    batch=128,
    device='0,1,2,3,4,5,6,7',
    workers=8,
    project=str(OUTPUT_DIR),
    name='baseball_yolov13s',
    amp=True,
    patience=50,
    save=True,
    save_period=10,
    val=True,
    plots=True,
    verbose=True,
)
