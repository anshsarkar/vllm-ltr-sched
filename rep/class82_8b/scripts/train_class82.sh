#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
TRAIN_DIR="$PROJECT_ROOT/vllm-ltr/train"
DATA_DIR="$PROJECT_ROOT/data/datasets/Llama3-Trace"

cd "$TRAIN_DIR"

# Symlink training data from downloaded location into jsonfiles/
if [ -d "$DATA_DIR" ]; then
    mkdir -p jsonfiles
    for f in "$DATA_DIR"/*.jsonl; do
        base=$(basename "$f")
        [ -e "jsonfiles/$base" ] || ln -sf "$f" "jsonfiles/$base"
    done
    echo "Symlinked datasets from $DATA_DIR into jsonfiles/"
else
    echo "WARNING: Data directory not found: $DATA_DIR"
    echo "Make sure to run scripts/download_data.sh first"
    exit 1
fi

echo "=== Training class82 (bucket=100) classifiers for 8B ==="

# LMSYS
python trainer.py --config configs/config_prefill_opt_classify.txt \
    --file jsonfiles/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-lmsys-class-trainbucket100-b32 \
    --batch-size 32 \
    --label-group-size 100

# ShareGPT
python trainer.py --config configs/config_prefill_opt_classify.txt \
    --file jsonfiles/llama3-8b-sharegpt-train-t1-s0-8192.jsonl \
    --job-dir MODEL \
    --run-id opt-125m-llama3-8b-sharegpt-class-trainbucket100-b32 \
    --batch-size 32 \
    --label-group-size 100

echo "=== Training complete ==="
