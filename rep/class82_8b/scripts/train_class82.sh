#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
TRAIN_DIR="$PROJECT_ROOT/vllm-ltr/train"

cd "$TRAIN_DIR"

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
