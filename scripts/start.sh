#!/usr/bin/env bash
# Start the vllm-ltr development container.
# Usage:
#   bash scripts/start.sh          # Interactive shell
#   bash scripts/start.sh -d       # Detached mode
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check for .env file
if [ ! -f "$PROJECT_ROOT/docker/.env" ]; then
    echo "ERROR: docker/.env not found."
    echo "Create it from the template:"
    echo "  cp docker/.env.template docker/.env"
    echo "  # Then edit docker/.env with your HF_TOKEN"
    exit 1
fi

# Check Docker image exists
if ! docker image inspect vllm-ltr-dev &>/dev/null; then
    echo "ERROR: Docker image 'vllm-ltr-dev' not found."
    echo "Build it first: bash scripts/build_docker.sh"
    exit 1
fi

DETACHED=""
if [ "${1:-}" = "-d" ]; then
    DETACHED="-d"
    echo "Starting container in detached mode..."
fi

# Ensure mount directories exist on host
mkdir -p "$PROJECT_ROOT/data" "$PROJECT_ROOT/experiments"

# Resolve host HF cache directory
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

docker run --rm \
    --gpus all \
    --name vllm-ltr \
    --env-file "$PROJECT_ROOT/docker/.env" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e HF_HOME=/root/.cache/huggingface \
    -v "$PROJECT_ROOT/vllm-ltr":/workspace/vllm-ltr \
    -v "$PROJECT_ROOT/scripts":/workspace/scripts \
    -v "$PROJECT_ROOT/data":/workspace/data \
    -v "$HF_CACHE_DIR":/root/.cache/huggingface \
    -v "$PROJECT_ROOT/experiments":/workspace/experiments \
    -p 8000:8000 \
    -w /workspace \
    -it $DETACHED \
    vllm-ltr-dev \
    bash -c "cp /opt/vllm-kernels/*.so /workspace/vllm-ltr/vllm/ 2>/dev/null; pip install -e /workspace/vllm-ltr --no-build-isolation --no-deps -q; exec bash"

if [ -n "$DETACHED" ]; then
    echo ""
    echo "Container started. Attach with:"
    echo "  docker exec -it vllm-ltr bash"
    echo ""
    echo "Stop with:"
    echo "  docker stop vllm-ltr"
fi
