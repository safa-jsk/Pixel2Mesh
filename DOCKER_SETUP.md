# Docker Setup Guide for Pixel2Mesh

## Prerequisites

- Docker installed
- NVIDIA GPU with CUDA 11.3+ support
- NVIDIA drivers installed on Linux host
- NVIDIA Container Toolkit (instructions below)

## GPU Setup (NVIDIA Container Toolkit)

**Important:** GPU support requires the NVIDIA Container Toolkit. Run this once:

```bash
./setup-nvidia-docker.sh
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
sudo docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu20.04 nvidia-smi
```

## Building the Docker Image

### Option 1: Using the build script (Recommended)

```bash
chmod +x scripts/docker/docker-build.sh
./scripts/docker/docker-build.sh
```

### Option 2: Using docker-compose

```bash
cd scripts/docker
sudo docker-compose build
```

### Option 3: Manual build

```bash
sudo docker build -t pixel2mesh:latest .
```

**Note:** First build will download ~2GB of base images and may take 10-15 minutes.

## Running the Container

### Interactive shell

```bash
sudo docker run --gpus all -it --rm -v $(pwd):/workspace pixel2mesh:latest bash
```

### Using docker-compose

```bash
sudo docker-compose run --rm pixel2mesh
```

### Run training

```bash
sudo docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  pixel2mesh:latest \
  python entrypoint_train.py --config experiments/designA_vgg_baseline.yml
```

### Run evaluation

```bash
sudo docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  pixel2mesh:latest \
  python entrypoint_eval.py --config experiments/designA_vgg_baseline.yml
```

### Run prediction

```bash
sudo docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  pixel2mesh:latest \
  python entrypoint_predict.py --config experiments/designA_vgg_baseline.yml
```

## Volume Mounts

The `-v $(pwd):/workspace` flag mounts the current directory into the container at `/workspace`, allowing the container to:

- Read/write datasets from `datasets/data/`
- Save checkpoints and results
- Access configuration files

## GPU Access

The `--gpus all` flag enables GPU access in the container. Ensure:

- NVIDIA drivers are installed on the host
- nvidia-docker2 is installed
- Docker daemon is running with GPU support

## Troubleshooting

### GPU Error: "could not select device driver"

**Error:** `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`

**Solution:** NVIDIA Container Toolkit is not installed. Run:

```bash
./setup-nvidia-docker.sh
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
sudo docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu20.04 nvidia-smi
```

### Build Fails with GPG Errors

The Dockerfile includes fixes for common GPG key issues with CUDA base images. If problems persist, try:

```bash
sudo docker build --no-cache -t pixel2mesh:latest .
```

## Development Workflow

1. Make code changes on your host machine
2. Changes are immediately reflected in container (via volume mount)
3. Run experiments inside the container
4. Results are saved back to host via volume mount

## Customization

### Modify Python packages

Edit the `Dockerfile` and add packages:

```dockerfile
RUN pip install your-package-name
```

Then rebuild:

```bash
./scripts/docker/docker-build.sh
```

### Change CUDA version

Modify the FROM line in `Dockerfile`:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
```

Then update PyTorch installation to match.
