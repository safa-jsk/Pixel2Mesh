#!/bin/bash
# Environment check: verify all dependencies are available
# Run from repo root: bash scripts/env_check.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Pixel2Mesh Environment Check ==="
echo ""

# Python
echo "Python: $(python --version 2>&1)"
echo "  Path: $(which python)"
echo ""

# PyTorch
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'  Compute capability: {cap[0]}.{cap[1]}')
    print(f'  TF32 supported: {cap[0] >= 8}')
else:
    print('  WARNING: No CUDA GPU detected')
"
echo ""

# Package imports
echo "Package imports:"
python -c "
import sys, os
sys.path.insert(0, os.path.join('$REPO_ROOT', 'src'))
try:
    import pixel2mesh; print(f'  pixel2mesh: OK (v{pixel2mesh.__version__})')
except Exception as e:
    print(f'  pixel2mesh: FAIL ({e})')

for pkg in ['numpy', 'scipy', 'skimage', 'cv2', 'trimesh', 'easydict', 'yaml', 'tensorboardX', 'PIL', 'imageio', 'tqdm']:
    try:
        __import__(pkg)
        print(f'  {pkg}: OK')
    except ImportError:
        print(f'  {pkg}: MISSING')
"
echo ""

# Chamfer extension
echo "CUDA extensions:"
python -c "
try:
    import chamfer
    print('  chamfer: OK')
except ImportError:
    print('  chamfer: NOT BUILT (run: bash scripts/build_chamfer.sh)')
"
python -c "
try:
    import neural_renderer
    print('  neural_renderer: OK')
except ImportError:
    print('  neural_renderer: NOT INSTALLED (cd external/neural_renderer && pip install .)')
"
echo ""

# Dataset check
echo "Dataset:"
if [ -d "$REPO_ROOT/datasets/data/shapenet" ]; then
    echo "  ShapeNet: found at datasets/data/shapenet/"
else
    echo "  ShapeNet: NOT FOUND (expected at datasets/data/shapenet/)"
fi
if [ -f "$REPO_ROOT/datasets/data/pretrained/tensorflow.pth.tar" ]; then
    echo "  Pretrained weights: found"
else
    echo "  Pretrained weights: NOT FOUND (expected at datasets/data/pretrained/tensorflow.pth.tar)"
fi
if [ -f "$REPO_ROOT/datasets/data/ellipsoid/info_ellipsoid.dat" ]; then
    echo "  Ellipsoid template: found"
else
    echo "  Ellipsoid template: NOT FOUND (expected at datasets/data/ellipsoid/info_ellipsoid.dat)"
fi

echo ""
echo "=== Check Complete ==="
