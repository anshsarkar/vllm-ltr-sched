#!/usr/bin/env bash
#
# Train all 4 B4 loss experiment variants on both datasets (8 runs total).
#
# Experiment 1a: Cost-sensitive CE + expected cost (trainer_costsensitive_ce.py)
# Experiment 1b: Cost-sensitive MSE in token space (trainer_costsensitive_mse.py)
# Experiment 2:  CORAL ordinal regression (trainer_coral.py)
# Experiment 3:  Direct token regression (trainer_regression.py)
#
# Same backbone (OPT-125m), same percentile boundaries (10 classes),
# same hyperparameters (batch 4, epoch 5, lr 2e-5) as the pctl10-mse baseline.
#
# Checkpoints land in vllm-ltr/train/MODEL/results/<run-id>/
# Logs land in extension/loss_experiments/logs/

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
# Experiment 1: Cost-Sensitive CE
# ──────────────────────────────────────────────
echo "=== [1/8] Training cost-sensitive CE — LMSYS ==="
python trainer_costsensitive_ce.py --config configs/config_prefill_opt_classify.txt \
    --file "$LMSYS_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-lmsys-pctl10-costsensitive-b4-ext \
    --batch-size 4 \
    --num-classes 10 \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_lmsys_costsensitive_ce.log"

echo "=== [2/8] Training cost-sensitive CE — ShareGPT ==="
python trainer_costsensitive_ce.py --config configs/config_prefill_opt_classify.txt \
    --file "$SHAREGPT_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-sharegpt-pctl10-costsensitive-b4-ext \
    --batch-size 4 \
    --num-classes 10 \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_sharegpt_costsensitive_ce.log"

# ──────────────────────────────────────────────
# Experiment 1b: Cost-Sensitive MSE (token-space)
# ──────────────────────────────────────────────
echo "=== [3/8] Training cost-sensitive MSE — LMSYS ==="
python trainer_costsensitive_mse.py --config configs/config_prefill_opt_classify.txt \
    --file "$LMSYS_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-lmsys-pctl10-costsensitive-mse-b4-ext \
    --batch-size 4 \
    --num-classes 10 \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_lmsys_costsensitive_mse.log"

echo "=== [4/8] Training cost-sensitive MSE — ShareGPT ==="
python trainer_costsensitive_mse.py --config configs/config_prefill_opt_classify.txt \
    --file "$SHAREGPT_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-sharegpt-pctl10-costsensitive-mse-b4-ext \
    --batch-size 4 \
    --num-classes 10 \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_sharegpt_costsensitive_mse.log"

# ──────────────────────────────────────────────
# Experiment 2: CORAL Ordinal Regression
# ──────────────────────────────────────────────
echo "=== [5/8] Training CORAL ordinal — LMSYS ==="
python trainer_coral.py --config configs/config_prefill_opt_classify.txt \
    --file "$LMSYS_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-lmsys-pctl10-coral-b4-ext \
    --batch-size 4 \
    --num-classes 10 \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_lmsys_coral.log"

echo "=== [6/8] Training CORAL ordinal — ShareGPT ==="
python trainer_coral.py --config configs/config_prefill_opt_classify.txt \
    --file "$SHAREGPT_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-sharegpt-pctl10-coral-b4-ext \
    --batch-size 4 \
    --num-classes 10 \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_sharegpt_coral.log"

# ──────────────────────────────────────────────
# Experiment 3: Direct Token Regression
# ──────────────────────────────────────────────
echo "=== [7/8] Training regression — LMSYS ==="
python trainer_regression.py --config configs/config_prefill_opt_classify.txt \
    --file "$LMSYS_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-lmsys-pctl10-regression-b4-ext \
    --batch-size 4 \
    --num-classes 10 \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_lmsys_regression.log"

echo "=== [8/8] Training regression — ShareGPT ==="
python trainer_regression.py --config configs/config_prefill_opt_classify.txt \
    --file "$SHAREGPT_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-sharegpt-pctl10-regression-b4-ext \
    --batch-size 4 \
    --num-classes 10 \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_sharegpt_regression.log"

echo ""
echo "All 3 loss experiments trained (8 runs total). Logs in: $LOG_DIR"
echo "Checkpoints in: $TRAIN_DIR/MODEL/results/*-b4-ext/"