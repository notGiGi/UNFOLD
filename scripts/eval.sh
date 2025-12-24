#!/bin/bash

# Evaluation script for K-Conditioned Decomposition
# Usage: bash scripts/eval.sh <checkpoint_path> <data_root> [num_slots]

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Please provide checkpoint path and data root directory"
    echo "Usage: bash scripts/eval.sh <checkpoint_path> <data_root> [num_slots]"
    exit 1
fi

CHECKPOINT_PATH=$1
DATA_ROOT=$2
NUM_SLOTS=${3:-5}
MODEL_CONFIG="configs/model.yaml"
TRAIN_CONFIG="configs/train.yaml"
OUTPUT_DIR="outputs/figures"

echo "========================================="
echo "K-Conditioned Decomposition Evaluation"
echo "========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data root: $DATA_ROOT"
echo "Number of slots: $NUM_SLOTS"
echo "Model config: $MODEL_CONFIG"
echo "Train config: $TRAIN_CONFIG"
echo "Output dir: $OUTPUT_DIR"
echo "========================================="

python -m kcd.eval \
    --model-config "$MODEL_CONFIG" \
    --train-config "$TRAIN_CONFIG" \
    --checkpoint "$CHECKPOINT_PATH" \
    --data-root "$DATA_ROOT" \
    --num-slots "$NUM_SLOTS" \
    --output-dir "$OUTPUT_DIR"

echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
