#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"
DATA_DIR="$PROJECT_ROOT/data/datasets"

mkdir -p "$DATA_DIR"

ALPACA_FILE="$DATA_DIR/alpaca_data.json"

if [ -f "$ALPACA_FILE" ]; then
    echo "Alpaca already exists: $ALPACA_FILE"
else
    echo "Downloading Stanford Alpaca dataset..."
    wget -q --show-progress \
        -O "$ALPACA_FILE" \
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    echo "Done."
fi