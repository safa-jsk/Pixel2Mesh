#!/bin/bash
# Run Pixel2Mesh prediction to generate mesh OBJ files

echo "Running Design A prediction to generate meshes..."
echo "Output will be saved to: outputs/designA_predictions/"
echo ""

sudo docker run --rm --gpus all \
  --shm-size=8g \
  -v "$PWD":/workspace \
  -w /workspace \
  p2m:designA \
  bash -c "
    export PYTHONPATH=/workspace/external/chamfer:/workspace/external/neural_renderer:\$PYTHONPATH
    
    echo '=== Installing CUDA extensions ==='
    cd /workspace/external/chamfer && python setup.py build_ext --inplace && pip install -e . > /dev/null 2>&1
    cd /workspace/external/neural_renderer && python setup.py build_ext --inplace && pip install -e . > /dev/null 2>&1
    
    echo ''
    echo '=== Generating meshes from sample images ==='
    cd /workspace
    python entrypoint_predict.py \
      --name designA_poster_samples \
      --options experiments/designA_vgg_baseline.yml \
      --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
      --folder datasets/examples_for_poster
  "

echo ""
echo "Mesh generation complete!"
echo "Check outputs/designA_predictions/ for generated OBJ files"
