#!/bin/bash
# Build script for Pixel2Mesh Docker image
# Base: nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
# Stack: Python 3.12 | PyTorch 2.5.1 | CUDA 12.6

echo "Building Pixel2Mesh Docker image..."
echo "This may take 10-15 minutes on first build (downloading ~4.3GB base image)"

# Build from project root (two levels up from DesignA_CPU/docker/)
cd "$(dirname "$0")/../.."
sudo docker build -t pixel2mesh:latest .

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Docker image built successfully!"
    echo ""
    echo "To run the container:"
    echo "  sudo docker run --gpus all -it --rm -v \$(pwd):/workspace pixel2mesh:latest"
    echo ""
    echo "To run with specific command:"
    echo "  sudo docker run --gpus all -it --rm -v \$(pwd):/workspace pixel2mesh:latest python entrypoint_train.py"
else
    echo "✗ Docker build failed"
    exit 1
fi
