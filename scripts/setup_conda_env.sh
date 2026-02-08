#!/usr/bin/env bash
# Sets up a conda environment with all vllm-ltr dependencies.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="vllm-ltr"

# ---- Install Miniconda if missing ----
if ! command -v conda &>/dev/null; then
    echo "Installing Miniconda..."
    wget -q --show-progress -O /tmp/miniconda.sh \
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    "$HOME/miniconda3/bin/conda" init bash
else
    eval "$(conda shell.bash hook)"
fi

# Accept Anaconda ToS non-interactively
conda config --set plugins.auto_accept_tos yes

# ---- Create and activate env ----
conda create -n "$ENV_NAME" python=3.10 -y 2>/dev/null || true
conda activate "$ENV_NAME"

# ---- Install all dependencies ----
echo "Installing PyTorch 2.2.1 + CUDA 12.1..."
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "Installing flash-attn 2.5.6 (pre-built wheel)..."
FLASH_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
pip install "$FLASH_WHEEL" || MAX_JOBS=4 pip install flash-attn==2.5.6 --no-build-isolation

echo "Installing xformers..."
pip install xformers==0.0.25

echo "Installing build tools (ccache speeds up recompilation)..."
sudo apt-get install -y ccache >/dev/null 2>&1 || true
pip install ninja cmake>=3.21

echo "Installing vllm-ltr from source (compiles CUDA kernels)..."
cd "$PROJECT_ROOT/vllm-ltr"
NJOBS=$(nproc)
TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS="$NJOBS" NVCC_THREADS="$NJOBS" \
    VLLM_INSTALL_PUNICA_KERNELS=1 \
    pip install -e .
cd "$PROJECT_ROOT"

echo "Installing extra libraries..."
pip install numpy==1.25.2 fschat accelerate gcsfs scikit-learn scipy matplotlib evaluate
pip install huggingface_hub[cli]

echo ""
echo "Done! Next steps:"
echo "  conda activate $ENV_NAME"
echo "  huggingface-cli login"
echo "  bash scripts/download_data.sh"
