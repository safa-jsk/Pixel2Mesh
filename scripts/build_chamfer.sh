#!/bin/bash
# Build the Chamfer distance CUDA extension
# Run from repo root: bash scripts/build_chamfer.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHAMFER_DIR="$REPO_ROOT/external/chamfer"

echo "Building Chamfer distance CUDA extension..."
echo "  Source: $CHAMFER_DIR"

cd "$CHAMFER_DIR"
python setup.py install 2>&1

echo ""
echo "Chamfer extension built successfully."
echo "Verify: python -c 'import chamfer; print(\"chamfer OK\")'"
