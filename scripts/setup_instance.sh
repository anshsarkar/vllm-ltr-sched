#!/usr/bin/env bash
set -euo pipefail

echo "=== vllm-ltr-sched: Instance Setup ==="
echo ""

# ---- 1. Install Docker ----
if ! command -v docker &>/dev/null; then
    echo "[1/4] Installing Docker..."
    sudo apt-get update -y
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker "$USER"
    echo "Docker installed. You may need to log out and back in for group changes."
else
    echo "[1/4] Docker already installed: $(docker --version)"
fi

# ---- 2. Install NVIDIA Container Toolkit ----
if ! dpkg -s nvidia-container-toolkit &>/dev/null 2>&1; then
    echo "[2/4] Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -y
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "NVIDIA Container Toolkit installed."
else
    echo "[2/4] NVIDIA Container Toolkit already installed."
fi

# ---- 3. Install nvtop ----
if ! command -v nvtop &>/dev/null; then
    echo "[3/4] Installing nvtop..."
    sudo apt-get update -y
    sudo apt-get install -y nvtop
    echo "nvtop installed."
else
    echo "[3/4] nvtop already installed."
fi

# ---- 4. Verify GPU access ----
echo "[4/4] Verifying GPU access..."
echo "  Host GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  WARNING: nvidia-smi failed on host"
echo "  Docker GPU:"
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  WARNING: Docker GPU test failed. Check NVIDIA Container Toolkit."
echo "If this fails, then try logging out and sshing back to refresh group permissions"
echo ""
echo "=== Instance setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Log out and back in (for docker group)"
echo "  2. Run: bash scripts/setup_conda_env.sh"
