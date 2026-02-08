#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

echo "=== Downloading datasets and models ==="
echo "Data directory: $DATA_DIR"
echo ""

# Check HuggingFace login
if ! huggingface-cli whoami &>/dev/null; then
    echo "ERROR: Not logged in to HuggingFace."
    echo "Run: huggingface-cli login"
    exit 1
fi

mkdir -p "$DATA_DIR/datasets" "$DATA_DIR/models/predictors"

# ---- 1. Download Llama3-Trace datasets ----
echo "[1/3] Downloading Llama3-Trace datasets..."
huggingface-cli download LLM-ltr/Llama3-Trace \
    --local-dir "$DATA_DIR/datasets/Llama3-Trace" \
    --repo-type dataset
echo "  Done."

# ---- 2. Download pre-trained OPT predictors ----
echo "[2/3] Downloading pre-trained OPT predictors..."
huggingface-cli download LLM-ltr/OPT-Predictors \
    --local-dir "$DATA_DIR/models/predictors"
echo "  Done."

# ---- 3. Download ShareGPT raw data ----
echo "[3/3] Downloading ShareGPT data..."
if [ ! -f "$DATA_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json" ]; then
    wget -q --show-progress \
        -O "$DATA_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json" \
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    echo "  Done."
else
    echo "  Already exists, skipping."
fi

echo ""
echo "=== All downloads complete ==="
echo ""
echo "Downloaded files:"
echo "  Datasets:   $DATA_DIR/datasets/"
echo "  Predictors: $DATA_DIR/models/predictors/"
echo ""
echo "Llama models will be auto-downloaded by vLLM when we use them on their first run."
echo "      It will be cached in the Docker 'hf-cache' volume."
