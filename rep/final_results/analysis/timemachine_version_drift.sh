#!/usr/bin/env bash
# Show what package versions would have been installed at different points in time
# using the authors' exact requirement specs through pypi-timemachine.
#
# Prerequisites:
#   pip install pypi-timemachine
#   conda
#   CUDA-capable machine (for torch, flash-attn, xformers wheels)
#
# Keeps all conda envs so you can test whether benchmarks run at each date.
#
# Usage:
#   bash timemachine_version_drift.sh
#
# After running, activate any env to test:
#   conda activate tm_20240424

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AUTHORS_REPO="/tmp/vllm-ltr-full"
OUTPUT_CSV="${SCRIPT_DIR}/version_drift.csv"

# Clone authors' repo if not already present
if [ ! -d "$AUTHORS_REPO" ]; then
    echo "Cloning authors' repo..."
    git clone --depth 1 https://github.com/hao-ai-lab/vllm-ltr.git "$AUTHORS_REPO"
fi

# 5 dates to test
DATES=("2024-04-24" "2024-07-31" "2024-10-31" "2025-01-31" "2025-03-24")
LABELS=("vLLM 0.4.1 release" "+3 months" "Last commit" "+3 months after" "Today")

# Libraries to track in the plot
TRACK_LIBS=("vllm" "torch" "flash-attn" "xformers" "transformers")

BASE_PORT=4242

# flash-attn has no PyPI package, must come from GitHub
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

echo "date,label,library,version,install_success" > "$OUTPUT_CSV"

for i in "${!DATES[@]}"; do
    DATE="${DATES[$i]}"
    LABEL="${LABELS[$i]}"
    ENV_NAME="tm_${DATE//-/}"
    PORT=$((BASE_PORT + i))

    echo ""
    echo "========================================"
    echo "Date: $DATE ($LABEL)"
    echo "========================================"

    # Skip if env already exists
    if conda env list | grep -q "$ENV_NAME"; then
        echo "Env $ENV_NAME already exists, recording versions from it."
        eval "$(conda shell.bash hook 2>/dev/null)"
        conda activate "$ENV_NAME"
        for lib in "${TRACK_LIBS[@]}"; do
            ver=$(pip show "$lib" 2>/dev/null | grep "^Version:" | awk '{print $2}')
            if [ -z "$ver" ]; then ver="NOT_INSTALLED"; fi
            echo "$DATE,$LABEL,$lib,$ver,true" >> "$OUTPUT_CSV"
            echo "  $lib: $ver"
        done
        conda deactivate 2>/dev/null || true
        continue
    fi

    # Start pypi-timemachine
    echo "Starting pypi-timemachine on port $PORT for $DATE..."
    pypi_timemachine "$DATE" -p "$PORT" &
    TM_PID=$!
    sleep 3

    if ! kill -0 "$TM_PID" 2>/dev/null; then
        echo "ERROR: pypi-timemachine failed to start for $DATE"
        for lib in "${TRACK_LIBS[@]}"; do
            echo "$DATE,$LABEL,$lib,FAILED,false" >> "$OUTPUT_CSV"
        done
        continue
    fi

    TM_INDEX="http://localhost:${PORT}/simple"
    # PyTorch CUDA wheels live on their own index, not PyPI
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    INSTALL_OK="true"

    # Create conda env
    echo "Creating conda env: $ENV_NAME"
    conda create -y -n "$ENV_NAME" python=3.10 -q
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "$ENV_NAME"

    # ---- Install everything using authors' exact specs ----
    # All PyPI packages go through timemachine. Torch CUDA wheels need extra index.
    # We combine both indexes so pip can resolve from either.

    # Step 1: torch + CUDA packages (authors pin torch==2.2.1, xformers==0.0.25)
    echo "[1/4] Installing torch, xformers, triton..."
    pip install --no-cache-dir \
        --index-url "$TM_INDEX" \
        --extra-index-url "$TORCH_INDEX" \
        --trusted-host localhost \
        "torch==2.2.1" \
        "xformers==0.0.25" \
        "triton>=2.2.0" \
        2>&1 | tail -5 || INSTALL_OK="partial"

    # Step 2: flash-attn from GitHub (not on PyPI)
    echo "[2/4] Installing flash-attn from GitHub wheel..."
    pip install --no-cache-dir "$FLASH_ATTN_WHEEL" \
        2>&1 | tail -3 || INSTALL_OK="partial"

    # Step 3: All other deps using authors' specs through timemachine
    echo "[3/4] Installing remaining packages (authors' specs, frozen to $DATE)..."
    pip install --no-cache-dir \
        --index-url "$TM_INDEX" \
        --extra-index-url https://pypi.org/simple/ \
        --trusted-host localhost \
        "transformers>=4.40.0" \
        "tokenizers>=0.19.1" \
        "ray>=2.9" \
        "pydantic>=2.0" \
        "prometheus_client>=0.18.0" \
        "filelock>=3.10.4" \
        "tiktoken==0.6.0" \
        "lm-format-enforcer==0.9.3" \
        "outlines==0.0.34" \
        "pynvml==11.5.0" \
        "vllm-nccl-cu12>=2.18,<2.19" \
        "psutil" \
        "sentencepiece" \
        "numpy" \
        "requests" \
        "py-cpuinfo" \
        "fastapi" \
        "uvicorn[standard]" \
        "typing_extensions" \
        "ninja" \
        "packaging" \
        "wheel" \
        "cmake>=3.21" \
        "setuptools>=49.4.0" \
        "accelerate" \
        "aiohttp" \
        "datasets" \
        "einops" \
        "evaluate" \
        "fschat" \
        "huggingface-hub" \
        "matplotlib" \
        "pandas" \
        "scipy" \
        "scikit-learn" \
        "pillow" \
        "httpx" \
        "safetensors" \
        "tqdm" \
        "PyYAML" \
        "rich" \
        "cloudpickle" \
        "msgpack" \
        "protobuf" \
        "grpcio" \
        2>&1 | tail -5 || INSTALL_OK="partial"

    # Step 4: Build vllm from authors' source
    echo "[4/4] Building vllm from authors' source..."
    pip install -e "$AUTHORS_REPO" --no-build-isolation \
        2>&1 | tail -5 || INSTALL_OK="partial"

    # ---- Record versions ----
    echo ""
    echo "Resolved versions for $DATE:"
    for lib in "${TRACK_LIBS[@]}"; do
        ver=$(pip show "$lib" 2>/dev/null | grep "^Version:" | awk '{print $2}')
        if [ -z "$ver" ]; then
            ver="NOT_INSTALLED"
            echo "$DATE,$LABEL,$lib,$ver,false" >> "$OUTPUT_CSV"
        else
            echo "$DATE,$LABEL,$lib,$ver,$INSTALL_OK" >> "$OUTPUT_CSV"
        fi
        echo "  $lib: $ver"
    done

    # Save full pip freeze for this date
    FULL_FREEZE="${SCRIPT_DIR}/freeze_${DATE//-/}.txt"
    pip freeze > "$FULL_FREEZE"
    echo "Full freeze: $FULL_FREEZE"

    # Stop timemachine, keep env
    conda deactivate 2>/dev/null || true
    kill "$TM_PID" 2>/dev/null || true
    wait "$TM_PID" 2>/dev/null || true
    echo "Env $ENV_NAME ready. Activate with: conda activate $ENV_NAME"
done

echo ""
echo "========================================"
echo "Done. Results: $OUTPUT_CSV"
echo "========================================"
cat "$OUTPUT_CSV"
echo ""
echo "Environments for benchmark testing:"
for i in "${!DATES[@]}"; do
    echo "  conda activate tm_${DATES[$i]//-/}   # ${DATES[$i]} (${LABELS[$i]})"
done
echo ""
echo "Cleanup: for d in ${DATES[*]}; do conda env remove -y -n tm_\${d//-/}; done"
