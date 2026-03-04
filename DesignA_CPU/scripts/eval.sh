#!/bin/bash
# Design A Evaluation Runner
# This script runs evaluation inside the Docker container with GPU support

# Ensure we're in the project root
if [ ! -f "entrypoint_eval.py" ]; then
    echo "ERROR: Must run from project root directory"
    echo "Usage: ./scripts/evaluation/run_designA_eval.sh"
    exit 1
fi

echo "Starting Design A evaluation in Docker container..."
echo "Dataset: ShapeNet test_tf (43,784 samples)"
echo "Checkpoint: tensorflow.pth.tar (VGG16 baseline)"
echo ""

sudo docker run --rm --gpus all \
  --shm-size=8g \
  -v "$PWD":/workspace \
  -w /workspace \
  p2m:designA \
  bash -c "
    echo '=== Building and installing CUDA extensions ==='
    cd /workspace/external/chamfer
    python setup.py build_ext --inplace
    pip install -e .
    
    cd /workspace/external/neural_renderer
    python setup.py build_ext --inplace
    pip install -e .
    
    echo ''
    echo '=== Verifying installations ==='
    python -c 'import sys; sys.path.insert(0, \"/workspace/external/chamfer\"); import chamfer; print(\"✓ chamfer loaded\")'
    python -c 'import neural_renderer; print(\"✓ neural_renderer loaded\")'
    
    echo ''
    echo '=== Starting evaluation ==='
    cd /workspace
    export PYTHONPATH=/workspace/external/chamfer:/workspace/external/neural_renderer:\$PYTHONPATH
    python entrypoint_eval.py \
      --name designA_vgg_baseline \
      --options experiments/designA_vgg_baseline.yml \
      --checkpoint datasets/data/pretrained/tensorflow.pth.tar
  "

echo ""
echo "Evaluation complete! Check logs/designA/ for results."
