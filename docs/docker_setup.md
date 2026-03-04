# Docker Setup Guide for Pixel2Mesh

## Prerequisites

- Docker installed
- NVIDIA GPU with CUDA 12.x support (driver >= 535)
- NVIDIA drivers installed on Linux host (Ubuntu 24.04 recommended)
- NVIDIA Container Toolkit (instructions below)

## Docker Image Stack

| Component   | Version                                          |
| ----------- | ------------------------------------------------ |
| Base image  | `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04`     |
| Python      | 3.12 (Ubuntu 24.04 default)                      |
| PyTorch     | 2.5.1 (`cu124` wheels, compatible with CUDA 12.6)|
| torchvision | 0.20.1                                           |
| CUDA        | 12.6                                             |
| cuDNN       | bundled with base image                          |

## GPU Setup (NVIDIA Container Toolkit)

**Important:** GPU support requires the NVIDIA Container Toolkit. Run this once:

```bash
./DesignA_GPU/docker/setup-nvidia-docker.sh
```

This script will:

1. Verify NVIDIA drivers are installed
2. Install NVIDIA Container Toolkit
3. Configure Docker to use NVIDIA runtime
4. Restart Docker daemon
5. Test GPU access

**Manual installation (if script fails):**

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
sudo docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

## Building the Docker Image

### Option 1: Using the build script (Recommended)

```bash
chmod +x DesignA_CPU/docker/docker-build.sh
./DesignA_CPU/docker/docker-build.sh
```

### Option 2: Using docker-compose

```bash
cd DesignA_GPU/docker
sudo docker-compose build
```

### Option 3: Manual build

```bash
sudo docker build -t pixel2mesh:latest .
```

**Note:** First build will download ~4.3GB of base images and may take 10-15 minutes.

## Running Per-Design with Docker Compose

Each design has its own `docker-compose.yml` that references the shared root `Dockerfile`:

| Design      | Compose file                          | GPU? |
| ----------- | ------------------------------------- | ---- |
| DesignA_CPU | `DesignA_CPU/docker/docker-compose.yml` | No   |
| DesignA_GPU | `DesignA_GPU/docker/docker-compose.yml` | Yes  |
| DesignB     | `DesignB/docker/docker-compose.yml`     | Yes  |

```bash
# DesignA_CPU (no GPU)
cd DesignA_CPU/docker && sudo docker-compose run --rm pixel2mesh-cpu

# DesignA_GPU
cd DesignA_GPU/docker && sudo docker-compose run --rm pixel2mesh

# DesignB
cd DesignB/docker && sudo docker-compose run --rm pixel2mesh-designB
```

Or run any design directly with `docker run`:

```bash
# CPU mode  (omit --gpus)
sudo docker run -it --rm -v $(pwd):/workspace pixel2mesh:latest bash

# GPU mode
sudo docker run --gpus all -it --rm -v $(pwd):/workspace pixel2mesh:latest bash
```

## Running the Container

### Interactive shell

```bash
sudo docker run --gpus all -it --rm -v $(pwd):/workspace pixel2mesh:latest bash
```

### Using docker-compose

```bash
cd DesignA_GPU/docker
sudo docker-compose run --rm pixel2mesh
```

### Run training

```bash
sudo docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  pixel2mesh:latest \
  python entrypoint_train.py --config configs/experiments/designA_vgg_baseline.yml
```

### Run evaluation

```bash
sudo docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  pixel2mesh:latest \
  python entrypoint_eval.py --config configs/experiments/designA_vgg_baseline.yml
```

### Run prediction

```bash
sudo docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  pixel2mesh:latest \
  python entrypoint_predict.py --config configs/experiments/designA_vgg_baseline.yml
```

## Volume Mounts

The `-v $(pwd):/workspace` flag mounts the current directory into the container at `/workspace`, allowing the container to:

- Read/write datasets from `datasets/data/`
- Save checkpoints and results
- Access configuration files

## GPU Access

The `--gpus all` flag enables GPU access in the container. Ensure:

- NVIDIA drivers >= 535 are installed on the host
- NVIDIA Container Toolkit is installed
- Docker daemon is running with GPU support

## Troubleshooting

### GPU Error: "could not select device driver"

**Error:** `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`

**Solution:** NVIDIA Container Toolkit is not installed. Run:

```bash
./DesignA_GPU/docker/setup-nvidia-docker.sh
```

### Permission Issues

If you encounter permission errors:

```bash
sudo usermod -aG docker $USER
# Log out and log back in
```

### GPU Not Detected

Check NVIDIA Docker runtime:

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

### Build Fails with Network Errors

Try building with `--no-cache`:

```bash
sudo docker build --no-cache -t pixel2mesh:latest  .
```

## Development Workflow

1. Make code changes on your host machine
2. Changes are immediately reflected in container (via volume mount)
3. Run experiments inside the container
4. Results are saved back to host via volume mount

## Customization

### Modify Python packages

Edit `Dockerfile` and add packages:

```dockerfile
RUN pip install your-package-name --break-system-packages
```

Then rebuild:

```bash
./DesignA_CPU/docker/docker-build.sh
```

### Change CUDA / PyTorch version

Modify the FROM line in `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
```

Then update the PyTorch pip install line to match (see https://pytorch.org/get-started/previous-versions/).
