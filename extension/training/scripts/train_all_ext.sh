#!/usr/bin/env bash
#
# Fresh retrain of our 4 novel predictor variants for the extension work.
# Authors' models (opt-xxx ranking, tpt-class10 classifier) are used as-is
# from data/models/predictors/.
#
# Run-ids use an -ext suffix to distinguish from original checkpoints.
# Checkpoints land in vllm-ltr/train/MODEL/results/<run-id>/.
# Logs land in extension/training/logs/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
TRAIN_DIR="$PROJECT_ROOT/vllm-ltr/train"
DATA_DIR="$PROJECT_ROOT/data/datasets/Llama3-Trace"
LOG_DIR="$PROJECT_ROOT/extension/training/logs"

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
# 1. tpt-class82 — ~82-class classifier (bucket width 100)
# ──────────────────────────────────────────────
echo "=== [1/4] Training class82 (bucket 100) — LMSYS ==="
python trainer.py --config configs/config_prefill_opt_classify.txt \
    --file "$LMSYS_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-lmsys-class-trainbucket100-b4-ext \
    --batch-size 4 \
    --label-group-size 100 \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_lmsys_class82.log"

echo "=== [1/4] Training class82 (bucket 100) — ShareGPT ==="
python trainer.py --config configs/config_prefill_opt_classify.txt \
    --file "$SHAREGPT_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-sharegpt-class-trainbucket100-b4-ext \
    --batch-size 4 \
    --label-group-size 100 \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_sharegpt_class82.log"

# ──────────────────────────────────────────────
# 2. tpt-width10 — ~820-class classifier (bucket width 10)
# ──────────────────────────────────────────────
echo "=== [2/4] Training width10 (bucket 10) — LMSYS ==="
python trainer.py --config configs/config_prefill_opt_classify.txt \
    --file "$LMSYS_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-lmsys-class-trainbucket10-b4-ext \
    --batch-size 4 \
    --label-group-size 10 \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_lmsys_width10.log"

echo "=== [2/4] Training width10 (bucket 10) — ShareGPT ==="
python trainer.py --config configs/config_prefill_opt_classify.txt \
    --file "$SHAREGPT_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-sharegpt-class-trainbucket10-b4-ext \
    --batch-size 4 \
    --label-group-size 10 \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_sharegpt_width10.log"

# ──────────────────────────────────────────────
# 3. tpt-pctl10 — Percentile-balanced, 10 classes, cross-entropy
# ──────────────────────────────────────────────
echo "=== [3/4] Training pctl10 (percentile, CE) — LMSYS ==="
python trainer_percentile.py --config configs/config_prefill_opt_classify.txt \
    --file "$LMSYS_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-lmsys-pctl10-b4-ext \
    --batch-size 4 \
    --num-classes 10 \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_lmsys_pctl10.log"

echo "=== [3/4] Training pctl10 (percentile, CE) — ShareGPT ==="
python trainer_percentile.py --config configs/config_prefill_opt_classify.txt \
    --file "$SHAREGPT_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-sharegpt-pctl10-b4-ext \
    --batch-size 4 \
    --num-classes 10 \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_sharegpt_pctl10.log"

# ──────────────────────────────────────────────
# 4. tpt-pctl10-mse — Percentile-balanced, 10 classes, MSE loss
# ──────────────────────────────────────────────
echo "=== [4/4] Training pctl10-mse (percentile, MSE) — LMSYS ==="
python trainer_percentile.py --config configs/config_prefill_opt_classify.txt \
    --file "$LMSYS_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-lmsys-pctl10-mse-b4-ext \
    --batch-size 4 \
    --num-classes 10 \
    --loss mse \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_lmsys_pctl10_mse.log"

echo "=== [4/4] Training pctl10-mse (percentile, MSE) — ShareGPT ==="
python trainer_percentile.py --config configs/config_prefill_opt_classify.txt \
    --file "$SHAREGPT_FILE" \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-sharegpt-pctl10-mse-b4-ext \
    --batch-size 4 \
    --num-classes 10 \
    --loss mse \
    --epoch 5 \
    2>&1 | tee "$LOG_DIR/train_sharegpt_pctl10_mse.log"

echo ""
echo "All 4 variants trained (8 runs total). Logs in: $LOG_DIR"
echo "Checkpoints in: $TRAIN_DIR/MODEL/results/*-ext/"
