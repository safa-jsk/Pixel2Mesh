#!/bin/bash
# Setup script for NVIDIA Container Toolkit (required for GPU support in Docker)

set -e

echo "======================================"
echo "NVIDIA Container Toolkit Setup"
echo "======================================"
echo ""

# Check if NVIDIA drivers are installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers first."
    echo "Visit: https://www.nvidia.com/Download/index.aspx"
    exit 1
fi

echo "✓ NVIDIA drivers detected:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo ""

# Check if already installed
if dpkg -l | grep -q nvidia-container-toolkit; then
    echo "✓ NVIDIA Container Toolkit is already installed"
    echo ""
else
    echo "Installing NVIDIA Container Toolkit..."
    echo ""
    
    # Add GPG key
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
    # Add repository
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    
    # Update and install
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    echo ""
    echo "✓ NVIDIA Container Toolkit installed"
fi

# Configure Docker
echo "Configuring Docker to use NVIDIA runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
echo "Restarting Docker daemon..."
sudo systemctl restart docker

echo ""
echo "======================================"
echo "✓ Setup Complete!"
echo "======================================"
echo ""
echo "Testing GPU access in Docker..."
if sudo docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
    echo "✓ GPU access working!"
else
    echo "⚠ GPU test failed. You may need to:"
    echo "  1. Restart your system"
    echo "  2. Check Docker daemon status: sudo systemctl status docker"
fi

echo ""
echo "You can now run containers with GPU support:"
echo "  sudo docker run --gpus all -it --rm -v \$(pwd):/workspace pixel2mesh:latest bash"
