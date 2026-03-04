FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages (python3-pip intentionally excluded — installed via get-pip.py below)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential cmake \
    python3 python3-dev python3-venv \
    libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Remove PEP 668 marker before installing pip (safe inside a container)
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

# Install pip fresh via get-pip.py (no Debian pip present = no RECORD conflict)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip install --upgrade setuptools wheel

# Set python3 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install PyTorch 2.5.1 with CUDA 12.4 support (compatible with CUDA 12.6 runtime)
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies (including ninja for faster CUDA extension builds)
RUN pip install opencv-python scipy scikit-image easydict pyyaml tensorboardx trimesh shapely ninja

# Build CUDA extensions (no GPU at build time — set arch list explicitly)
# Covers: Volta(7.0), Turing(7.5), Ampere(8.0,8.6), Ada(8.9), Hopper(9.0)
COPY external/chamfer /tmp/chamfer
RUN cd /tmp/chamfer && \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    pip install . --no-build-isolation && \
    rm -rf /tmp/chamfer

COPY external/neural_renderer /tmp/neural_renderer
RUN cd /tmp/neural_renderer && \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    pip install . --no-build-isolation && \
    rm -rf /tmp/neural_renderer

WORKDIR /workspace
CMD ["bash"]
