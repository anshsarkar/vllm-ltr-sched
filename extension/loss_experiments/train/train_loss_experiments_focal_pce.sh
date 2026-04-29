#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
TRAIN_DIR="$PROJECT_ROOT/vllm-ltr/train"
DATA_DIR="$PROJECT_ROOT/data/datasets/Llama3-Trace"
LOG_DIR="$SCRIPT_DIR/../logs"

cd "$TRAIN_DIR"

# Symlink datasets into the trainer's expected location
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

LMSYS_FILE="jsonfiles/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl"
SHAREGPT_FILE="jsonfiles/llama3-8b-sharegpt-train-t1-s0-8192.jsonl"

# ──────────────────────────────────────────────
# Experiment 5: Focal CE
# ──────────────────────────────────────────────
echo "=== [1/4] Training Focal CE — LMSYS ==="
python trainer_focal_ce.py --config configs/config_prefill_opt_classify.txt \
    --file "$LMSYS_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-lmsys-pctl10-focal-b4r2-ext \
    --batch-size 4 \
    --num-classes 10 \
    --epoch 5 \
    --gamma 2.0 \
    2>&1 | tee "$LOG_DIR/train_lmsys_focal_ce.log"

echo "=== [2/4] Training Focal CE — ShareGPT ==="
python trainer_focal_ce.py --config configs/config_prefill_opt_classify.txt \
    --file "$SHAREGPT_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-sharegpt-pctl10-focal-b4r2-ext \
    --batch-size 4 \
    --num-classes 10 \
    --epoch 5 \
    --gamma 2.0 \
    2>&1 | tee "$LOG_DIR/train_sharegpt_focal_ce.log"

# ──────────────────────────────────────────────
# Experiment 6: Pairwise CE
# ──────────────────────────────────────────────
echo "=== [3/4] Training Pairwise CE — LMSYS ==="
python trainer_pairwise_ce.py --config configs/config_prefill_opt_classify.txt \
    --file "$LMSYS_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-lmsys-pctl10-pairwise-b4r2-ext \
    --batch-size 4 \
    --num-classes 10 \
    --epoch 5 \
    --pairwise-lambda 1.0 \
    --pairwise-margin 0.5 \
    2>&1 | tee "$LOG_DIR/train_lmsys_pairwise_ce.log"

echo "=== [4/4] Training Pairwise CE — ShareGPT ==="
python trainer_pairwise_ce.py --config configs/config_prefill_opt_classify.txt \
    --file "$SHAREGPT_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-sharegpt-pctl10-pairwise-b4r2-ext \
    --batch-size 4 \
    --num-classes 10 \
    --epoch 5 \
    --pairwise-lambda 1.0 \
    --pairwise-margin 0.5 \
    2>&1 | tee "$LOG_DIR/train_sharegpt_pairwise_ce.log"

echo ""
echo "All round 2 loss experiments trained. Logs in: $LOG_DIR"
echo "Checkpoints in: $TRAIN_DIR/MODEL/results/*-b4r2-ext/"