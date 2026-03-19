#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"
TRAIN_DIR="$PROJECT_ROOT/vllm-ltr/train"
DATA_DIR="$PROJECT_ROOT/data/datasets/Llama3-Trace"
LOG_DIR="$PROJECT_ROOT/rep/final_results/training/logs"

cd "$TRAIN_DIR"

if [ -d "$DATA_DIR" ]; then
    mkdir -p jsonfiles
    for f in "$DATA_DIR"/*.jsonl; do
        base=$(basename "$f")
        [ -e "jsonfiles/$base" ] || ln -sf "$f" "jsonfiles/$base"
    done
else
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

mkdir -p "$LOG_DIR"

# pctl10-mse (percentile-balanced, 10 classes, MSE loss)
echo "=== Training pctl10-mse (LMSYS) ==="
python trainer_percentile.py --config configs/config_prefill_opt_classify.txt \
    --file jsonfiles/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-lmsys-pctl10-mse-b4 \
    --batch-size 4 \
    --num-classes 10 \
    --loss mse \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_lmsys_pctl10_mse.log"

echo "=== Training pctl10-mse (ShareGPT) ==="
python trainer_percentile.py --config configs/config_prefill_opt_classify.txt \
    --file jsonfiles/llama3-8b-sharegpt-train-t1-s0-8192.jsonl \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-sharegpt-pctl10-mse-b4 \
    --batch-size 4 \
    --num-classes 10 \
    --loss mse \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_sharegpt_pctl10_mse.log"

echo "pctl10-mse training complete. Logs in: $LOG_DIR"
