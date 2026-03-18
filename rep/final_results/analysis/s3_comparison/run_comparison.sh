#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"
DATA_DIR="$PROJECT_ROOT/data/datasets"
OUTPUT_DIR="$SCRIPT_DIR/results"

echo "=== S³ vs vllm-ltr Distribution Comparison ==="

# Download Alpaca
bash "$SCRIPT_DIR/download_alpaca.sh"

# Check datasets
SHAREGPT="$DATA_DIR/Llama3-Trace/llama3-8b-sharegpt-train-t1-s0-8192.jsonl"
LMSYS="$DATA_DIR/Llama3-Trace/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl"
ALPACA="$DATA_DIR/alpaca_data.json"

for f in "$SHAREGPT" "$LMSYS" "$ALPACA"; do
    [ -f "$f" ] || { echo "ERROR: Missing $f"; exit 1; }
done

mkdir -p "$OUTPUT_DIR"

python "$SCRIPT_DIR/compare_distributions.py" \
    --alpaca "$ALPACA" \
    --sharegpt "$SHAREGPT" \
    --lmsys "$LMSYS" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/comparison.log"

echo ""
echo "Results: $OUTPUT_DIR/"