#!/usr/bin/env bash
# Set up a conda environment with all dependencies for running
# vllm-ltr benchmarks locally (without Docker).
#
# Installs: conda, Python 3.10, PyTorch 2.2.1, flash-attn, xformers,
#           vllm-ltr (editable), HuggingFace CLI, and all extras.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="vllm-ltr"

echo "=== vllm-ltr-sched: Conda Environment Setup ==="
echo ""

# ---- 1. Install Miniconda if not present ----
if ! command -v conda &>/dev/null; then
    echo "[1/6] Installing Miniconda..."
    wget -q --show-progress \
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh

    # Initialize conda for the current shell
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    "$HOME/miniconda3/bin/conda" init bash
    echo "Miniconda installed at $HOME/miniconda3"
else
    echo "[1/6] Conda already installed: $(conda --version)"
    eval "$(conda shell.bash hook)"
fi

# Auto-accept Anaconda channel ToS for non-interactive use
conda config --set plugins.auto_accept_tos yes

# ---- 2. Create conda environment ----
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[2/6] Conda env '$ENV_NAME' already exists. Activating..."
else
    echo "[2/6] Creating conda environment '$ENV_NAME' with Python 3.10..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"
echo "  Python: $(python --version)"

# ---- 3. Install PyTorch (CUDA 12.1) ----
echo "[3/6] Installing PyTorch 2.2.1 + CUDA 12.1..."
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ---- 4. Install flash-attn and xformers ----
echo "[4/6] Installing flash-attn 2.5.6 and xformers 0.0.25..."

# Try pre-built wheel first (much faster than building from source)
FLASH_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
pip install "$FLASH_WHEEL" 2>/dev/null || {
    echo "  Pre-built wheel failed, building from source (this takes ~20 min)..."
    MAX_JOBS=4 pip install flash-attn==2.5.6 --no-build-isolation
}

pip install xformers==0.0.25

# ---- 5. Install vllm-ltr from source ----
echo "[5/6] Installing vllm-ltr from source (editable mode)..."
echo "  This compiles CUDA kernels and may take 10-15 minutes on first install."

cd "$PROJECT_ROOT/vllm-ltr"

# Target A100 only for faster compilation
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4
export NVCC_THREADS=4
export VLLM_INSTALL_PUNICA_KERNELS=1

pip install -e .

# Extra libraries from the paper's README
pip install numpy==1.25.2 fschat accelerate gcsfs scikit-learn scipy matplotlib evaluate

cd "$PROJECT_ROOT"

# ---- 6. HuggingFace CLI ----
echo "[6/6] Setting up HuggingFace CLI..."
pip install huggingface_hub[cli]

echo ""
if huggingface-cli whoami &>/dev/null; then
    echo "  Already logged in as: $(huggingface-cli whoami 2>/dev/null | head -1)"
else
    echo "  Log in to HuggingFace (required for Meta-Llama-3-8B-Instruct):"
    echo "    huggingface-cli login"
fi

echo ""
echo "=== Conda environment setup complete ==="
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Next steps:"
echo "  1. huggingface-cli login  (if not already logged in)"
echo "  2. bash scripts/download_data.sh"
echo "  3. bash scripts/run_bench_8b.sh"
