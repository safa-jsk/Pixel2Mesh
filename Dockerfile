FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Fix GPG key issues - remove problematic CUDA repo lists and clean cache
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    apt-get update -o Acquire::AllowInsecureRepositories=true && \
    apt-get install -y --no-install-recommends --allow-unauthenticated ca-certificates gnupg && \
    rm -rf /var/cache/apt/archives/* && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 871920D1991BC93C 3B4FE6ACC0B21F32 && \
    apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget build-essential cmake \
    python3.8 python3-pip python3.8-dev \
    libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

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
