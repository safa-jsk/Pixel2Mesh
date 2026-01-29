FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates build-essential cmake \
    python3.8 python3-pip python3.8-dev \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set python3.8 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.3 support
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other dependencies
RUN pip install opencv-python scipy scikit-image easydict pyyaml tensorboardx trimesh shapely

WORKDIR /workspace
CMD ["bash"]
