# Design A Setup Log

**Date:** January 29, 2026  
**Goal:** Implement Design A baseline for Pixel2Mesh

## Environment Setup

### Docker Configuration

- **Base Image:** nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
- **Python Version:** 3.8
- **PyTorch Version:** 1.12.1
- **CUDA Toolkit:** 11.3

**Note:** Updated from original guideline's CUDA 10.0 to CUDA 11.3 due to Docker image availability. This is documented as a compatibility fix for Ubuntu 22.04.

### Changes from Original Guideline

1. **CUDA Version:** 10.0 → 11.3.1 (image availability)
2. **Ubuntu Version:** 18.04 → 20.04 (base image requirement)
3. **Python Version:** 3.7 → 3.8 (compatibility with newer PyTorch)
4. **PyTorch Version:** 1.1 → 1.12.1 (CUDA 11.3 compatibility)
5. **Package Manager:** Conda → pip (faster, simpler for Docker)
6. **Added:** DEBIAN_FRONTEND=noninteractive to avoid prompts during build

### Build Command

```bash
docker build -t p2m:designA .
```

### Run Command (when build completes)

```bash
docker run --rm -it --gpus all -v "$PWD":/workspace p2m:designA
```

## Next Steps

1. ✅ Create Dockerfile
2. ✅ Build Docker image (completed successfully)
3. ✅ Verify basic container functionality (Python & PyTorch working)
4. ✅ Install NVIDIA Container Toolkit
5. ⚠️ Enable GPU access in Docker Desktop (required for CUDA compilation)
6. ⏳ Compile CUDA extensions (chamfer, neural_renderer)
7. ⏳ Download dataset and checkpoints
8. ⏳ Run inference and evaluation

## Build Results

**Docker Image:** p2m:designA  
**Image Size:** 24.2GB (8.73GB compressed)  
**Build Status:** ✅ Successful  
**PyTorch Version:** 1.12.1+cu113  
**Build Time:** ~4 minutes

### GPU Access Status

✅ **NVIDIA Container Toolkit installed** (v1.13.5)  
⚠️ **Docker Desktop GPU passthrough issue** - Using Docker Desktop on Linux requires additional GPU configuration

**Current Issue:**  
Docker Desktop on Linux doesn't support `--gpus all` flag by default. The CUDA extensions (chamfer, neural_renderer) require GPU access during compilation.

**Solutions (choose one):**

1. **Enable GPU in Docker Desktop Settings** (if available)
   - Open Docker Desktop
   - Go to Settings → Resources → Enable GPU
2. **Alternative: Native Docker Engine** (if issues persist)
   - Remove Docker Desktop
   - Install native Docker Engine
   - Re-run NVIDIA Container Toolkit setup

3. **Workaround: Compile on host system**
   - Install dependencies on host Ubuntu 22.04
   - Compile extensions natively
   - Use Docker only for final inference/evaluation

**For now, proceeding with workaround #3 is recommended.**

## Compatibility Fixes Applied

- Updated CUDA/PyTorch stack to work with available Docker images
- All changes are environment/compatibility related, no algorithmic changes
- This maintains Design A's goal of baseline reproduction
