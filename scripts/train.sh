#!/bin/bash

# Training script for K-Conditioned Decomposition
# Usage: bash scripts/train.sh <data_root>

set -e

if [ -z "$1" ]; then
    echo "Error: Please provide data root directory"
    echo "Usage: bash scripts/train.sh <data_root>"
    exit 1
fi

DATA_ROOT=$1
MODEL_CONFIG="configs/model.yaml"
TRAIN_CONFIG="configs/train.yaml"

echo "========================================="
echo "K-Conditioned Decomposition Training"
echo "========================================="
echo "Data root: $DATA_ROOT"
echo "Model config: $MODEL_CONFIG"
echo "Train config: $TRAIN_CONFIG"
echo "========================================="

python -m kcd.train \
    --model-config "$MODEL_CONFIG" \
    --train-config "$TRAIN_CONFIG" \
    --data-root "$DATA_ROOT"

echo "Training completed!"
