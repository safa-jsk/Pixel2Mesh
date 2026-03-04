#!/bin/bash
# Run Pixel2Mesh prediction to generate mesh OBJ files

# Ensure we're in the project root
if [ ! -f "entrypoint_predict.py" ]; then
    echo "ERROR: Must run from project root directory"
    echo "Usage: ./scripts/evaluation/run_designA_predict.sh"
    exit 1
fi

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
    pip install tqdm > /dev/null 2>&1
    cd /workspace/external/chamfer && python setup.py build_ext --inplace && pip install -e . > /dev/null 2>&1
    cd /workspace/external/neural_renderer && python setup.py build_ext --inplace && pip install -e . > /dev/null 2>&1
    
    echo ''
    echo '=== Generating meshes from sample images ==='
    cd /workspace
    
    # Time the prediction
    start_time=\$(date +%s)
    python entrypoint_predict.py \
      --name designA_poster_samples \
      --options experiments/designA_vgg_baseline.yml \
      --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
      --folder datasets/examples_for_poster
    end_time=\$(date +%s)
    
    # Calculate timing
    total_time=\$((end_time - start_time))
    num_images=\$(ls datasets/examples_for_poster/*.png | wc -l)
    avg_time=\$(python -c \"print(round(\$total_time / \$num_images, 2))\")
    
    echo ''
    echo '============================================================'
    echo 'MESH GENERATION TIMING:'
    echo \"Total time: \$total_time seconds\"
    echo \"Images processed: \$num_images\"
    echo \"Average time per mesh: \$avg_time seconds\"
    echo '============================================================'
  "

echo ""
echo "Mesh generation complete!"
echo "Check outputs/designA_predictions/ for generated OBJ files"
