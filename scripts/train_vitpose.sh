#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

DATA_ROOT="${BASEBALL_DATA_ROOT:-./data/baseball_pose}"
CONFIG_FILE="configs/vitpose/baseball_vitpose_base.py"
WORK_DIR="work_dirs/vitpose_baseball"

echo "=========================================="
echo "ViTPose Training"
echo "=========================================="
echo "Data path: $DATA_ROOT"
echo "Config: $CONFIG_FILE"
echo "Output: $WORK_DIR"
echo "=========================================="

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    $(python -c "import mmpose; print(mmpose.__path__[0])")/tools/train.py \
    $CONFIG_FILE \
    --work-dir $WORK_DIR \
    --launcher pytorch \
    --amp
