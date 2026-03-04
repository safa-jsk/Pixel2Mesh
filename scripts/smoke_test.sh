#!/bin/bash
# Smoke test: verify the package imports and (optionally) run a forward pass
# Run from repo root: bash scripts/smoke_test.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Pixel2Mesh Smoke Test ==="
echo ""

# Test 1: Package import
echo "[1/3] Testing pixel2mesh import..."
PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH" python -c "
import pixel2mesh
from pixel2mesh import config
from pixel2mesh.options import options
from pixel2mesh.models.p2m import P2MModel
from pixel2mesh.utils.mesh import Ellipsoid
from pixel2mesh.engine.base import CheckpointRunner
from pixel2mesh.engine.evaluator import Evaluator
from pixel2mesh.engine.trainer import Trainer
from pixel2mesh.engine.predictor import Predictor
print('  All imports OK')
"
echo ""

# Test 2: Chamfer extension
echo "[2/3] Testing chamfer CUDA extension..."
PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH" python -c "
try:
    import chamfer
    import torch
    if torch.cuda.is_available():
        a = torch.rand(1, 100, 3).cuda()
        b = torch.rand(1, 100, 3).cuda()
        d1, d2, _, _ = chamfer.forward(a, b)
        print(f'  Chamfer extension: OK (test distance: {d1.mean().item():.6f})')
    else:
        print('  Chamfer extension: imported OK (no GPU to test forward pass)')
except ImportError:
    print('  Chamfer extension: NOT BUILT (run bash scripts/build_chamfer.sh)')
    print('  (This is OK for import-only testing)')
"
echo ""

# Test 3: Model forward pass (requires GPU + data)
echo "[3/3] Testing model forward pass (optional, requires GPU)..."
PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH" python -c "
import torch
import os
import sys

if not torch.cuda.is_available():
    print('  Skipped: no GPU available')
    sys.exit(0)

try:
    from pixel2mesh.models.p2m import P2MModel
    from pixel2mesh.utils.mesh import Ellipsoid
    from pixel2mesh.options import options

    # Check ellipsoid data
    ellipsoid_path = os.path.join('$REPO_ROOT', 'datasets', 'data', 'ellipsoid', 'info_ellipsoid.dat')
    if not os.path.exists(ellipsoid_path):
        print('  Skipped: ellipsoid data not found at', ellipsoid_path)
        sys.exit(0)

    ellipsoid = Ellipsoid(options.dataset.mesh_pos)
    model = P2MModel(options.model, ellipsoid,
                     options.dataset.camera_f, options.dataset.camera_c,
                     options.dataset.mesh_pos)
    model = model.cuda()

    # Create dummy input
    dummy_img = torch.randn(1, 3, 224, 224).cuda()
    with torch.no_grad():
        out = model(dummy_img)
    
    coords = out['pred_coord']
    print(f'  Forward pass OK: output stages = {len(coords)}, final vertices = {coords[-1].shape}')
except Exception as e:
    print(f'  Forward pass FAILED: {e}')
"

echo ""
echo "=== Smoke Test Complete ==="
