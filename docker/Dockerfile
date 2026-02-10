FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt-get update -y && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git wget curl vim ccache \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Workaround for triton/pytorch compat issues
RUN ldconfig /usr/local/cuda-12.1/compat/

WORKDIR /workspace

# ---------- Pre-built wheels (cached layer — rarely changes) ----------

# PyTorch 2.2.1 + CUDA 12.1
RUN pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121

# xformers 0.0.25 (pre-built wheel from PyPI, matches torch 2.2.1)
RUN pip install xformers==0.0.25

# flash-attn 2.5.6 pre-built wheel (cu122 is compatible with cu121 runtime)
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN pip install ${FLASH_ATTN_WHEEL} || \
    pip install flash-attn==2.5.6 --no-build-isolation

# ---------- vllm-ltr dependencies ----------

# Copy requirements files first (cache-friendly: code changes won't bust this layer)
COPY vllm-ltr/requirements-common.txt /tmp/requirements-common.txt
COPY vllm-ltr/requirements-cuda.txt /tmp/requirements-cuda.txt

# Install all deps from requirements-cuda.txt (torch/xformers already installed above, pip will skip)
RUN pip install -r /tmp/requirements-cuda.txt

# Build tools
RUN pip install cmake>=3.21 ninja packaging "setuptools>=49.4.0"

# Pin versions compatible with PyTorch 2.2.1 / vLLM 0.4.1
RUN pip install transformers==4.40.2 huggingface_hub==0.23.5 tokenizers==0.19.1

# Extra libs from the paper's README
RUN pip install numpy==1.25.2 fschat accelerate gcsfs scikit-learn scipy matplotlib evaluate

# ---------- Build vllm-ltr CUDA kernels ----------

# Target only A100 architecture for faster compilation
ARG TORCH_CUDA_ARCH_LIST="8.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Note: setup.py computes actual_jobs = MAX_JOBS // NVCC_THREADS
ARG MAX_JOBS=192
ENV MAX_JOBS=${MAX_JOBS}
ARG NVCC_THREADS=2
ENV NVCC_THREADS=${NVCC_THREADS}
ENV VLLM_INSTALL_PUNICA_KERNELS=1

# Copy full vllm-ltr source for building CUDA kernels
COPY vllm-ltr/ /workspace/vllm-ltr/

# Build vllm-ltr in editable mode
ENV CCACHE_DIR=/root/.cache/ccache
WORKDIR /workspace/vllm-ltr
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    pip install -e .

# Backup compiled .so files — the vllm-ltr bind mount at runtime hides them,
# start.sh restores them before launching bash
RUN mkdir -p /opt/vllm-kernels && \
    find /workspace/vllm-ltr -name '*.so' -exec cp {} /opt/vllm-kernels/ \;

# ---------- Runtime setup ----------

WORKDIR /workspace

# HuggingFace cache directory (mount as volume at runtime)
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/hub

WORKDIR /workspace

CMD ["/bin/bash"]
