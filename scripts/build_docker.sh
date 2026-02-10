#!/usr/bin/env bash
# Build the vllm-ltr-dev Docker image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Building vllm-ltr-dev Docker image ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check that vllm-ltr source exists
if [ ! -f "$PROJECT_ROOT/vllm-ltr/setup.py" ]; then
    echo "ERROR: vllm-ltr source not found at $PROJECT_ROOT/vllm-ltr/"
    echo "Make sure the vllm-ltr code is in the vllm-ltr/ directory."
    exit 1
fi

# Enable BuildKit for cache mounts
export DOCKER_BUILDKIT=1

# Build targeting A100 only (faster compile, ~10 min vs ~30 min)
# Change TORCH_CUDA_ARCH_LIST if using different GPUs:
#   A100: "8.0"
#   H100: "9.0"
#   V100: "7.0"
#   Multiple: "7.0 8.0 9.0"
docker build \
    -t vllm-ltr-dev \
    -f "$PROJECT_ROOT/docker/Dockerfile" \
    --build-arg TORCH_CUDA_ARCH_LIST="8.0" \
    "$PROJECT_ROOT"

echo ""
echo "=== Build complete ==="
echo "Image: vllm-ltr-dev"
echo "Run: bash scripts/start.sh"
