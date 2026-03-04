#!/bin/bash
#===============================================================================
# Design B: Full Dataset Baseline Evaluation (Docker Version)
#===============================================================================
# Run this script FROM THE HOST to execute Design B evaluation inside Docker.
#
# Usage:
#   ./run_designB_eval_docker.sh [experiment_name] [batch_size] [gpus]
#
# Example:
#   ./run_designB_eval_docker.sh designB_full_eval 8 1
#===============================================================================

set -e

# Configuration
EXPERIMENT_NAME="${1:-designB_full_eval}"
BATCH_SIZE="${2:-8}"
GPUS="${3:-1}"
WORKSPACE="/datalust/Pixel2Mesh"

echo "==============================================================================="
echo "DESIGN B: Full Dataset Baseline Evaluation (Docker)"
echo "==============================================================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Batch Size: $BATCH_SIZE"
echo "GPUs: $GPUS"
echo "==============================================================================="

docker run --gpus all -it --rm \
    -v "$WORKSPACE:/workspace" \
    -w /workspace \
    pixel2mesh:latest \
    bash -c "chmod +x scripts/evaluation/run_designB_eval.sh && ./scripts/evaluation/run_designB_eval.sh $EXPERIMENT_NAME $BATCH_SIZE $GPUS"
