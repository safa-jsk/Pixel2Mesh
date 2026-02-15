# Pixel2Mesh Docker - Quick Reference

## First Time Setup
```bash
# 1. Build the Docker image (one time, ~10-15 minutes)
./docker-build.sh

# 2. Setup GPU support (one time)
./setup-nvidia-docker.sh
```

## Common Commands

### Interactive Shell (with GPU)
```bash
sudo docker run --gpus all -it --rm -v $(pwd):/workspace pixel2mesh:latest bash
```

### Run Training
```bash
sudo docker run --gpus all -it --rm -v $(pwd):/workspace pixel2mesh:latest \
  python entrypoint_train.py --config experiments/designA_vgg_baseline.yml
```

### Run Evaluation
```bash
sudo docker run --gpus all -it --rm -v $(pwd):/workspace pixel2mesh:latest \
  python entrypoint_eval.py --config experiments/designA_vgg_baseline.yml
```

### Run Prediction
```bash
sudo docker run --gpus all -it --rm -v $(pwd):/workspace pixel2mesh:latest \
  python entrypoint_predict.py --config experiments/designA_vgg_baseline.yml
```

### Check GPU Status
```bash
sudo docker run --rm --gpus all pixel2mesh:latest nvidia-smi
```

### Run Without GPU (CPU only)
```bash
# Just remove --gpus all flag
sudo docker run -it --rm -v $(pwd):/workspace pixel2mesh:latest bash
```

## Using Docker Compose

### Start interactive session
```bash
sudo docker-compose run --rm pixel2mesh
```

### Run specific command
```bash
sudo docker-compose run --rm pixel2mesh python entrypoint_train.py
```

## Flags Explained
- `--gpus all` - Enable all GPUs
- `-it` - Interactive terminal
- `--rm` - Remove container after exit
- `-v $(pwd):/workspace` - Mount current directory to /workspace

## Troubleshooting

### GPU not detected
```bash
# Reinstall NVIDIA Container Toolkit
./setup-nvidia-docker.sh
```

### Permission denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and log back in
```

### Container won't start
```bash
# Check Docker is running
sudo systemctl status docker
# Restart Docker
sudo systemctl restart docker
```

## File Locations in Container
- Project files: `/workspace/` (mounted from host)
- Datasets: `/workspace/datasets/data/`
- Models: `/workspace/models/`
- Checkpoints: `/workspace/checkpoints/` (if created)

## Useful Tips
1. All changes in `/workspace/` persist on your host machine
2. Installed packages in container don't persist (rebuild image if needed)
3. GPU memory is shared with host - close other GPU applications if needed
4. Use `Ctrl+D` or `exit` to leave container shell
