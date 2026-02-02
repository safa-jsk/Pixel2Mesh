#!/bin/bash
#===============================================================================
# Design B: Full Dataset Baseline Evaluation
#===============================================================================
# This script runs the Pixel2Mesh evaluation on all 43,784 training samples
# and generates meshes for 26 specific samples (2 per category).
#
# Metrics logged:
#   - Chamfer Distance
#   - F1-Score @ tau (1e-4)
#   - F1-Score @ 2*tau (2e-4)
#   - Timing (per sample, per batch, total)
#
# Output:
#   - logs/designB/<name>/sample_results.csv
#   - logs/designB/<name>/batch_results.csv
#   - logs/designB/<name>/evaluation_summary.json
#   - outputs/designB_meshes/*.obj (26 samples × 3 stages = 78 files)
#===============================================================================

set -e

# Configuration
EXPERIMENT_NAME="${1:-designB_full_eval}"
CHECKPOINT="datasets/data/pretrained/tensorflow.pth.tar"
OPTIONS="experiments/designB_baseline.yml"
OUTPUT_DIR="outputs/designB_meshes"
BATCH_SIZE="${2:-8}"
GPUS="${3:-1}"

# Generate timestamp for version
TIMESTAMP=$(date +"%m%d%H%M%S")
VERSION="${EXPERIMENT_NAME}_${TIMESTAMP}"

echo "==============================================================================="
echo "DESIGN B: Full Dataset Baseline Evaluation"
echo "==============================================================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Version: $VERSION"
echo "Checkpoint: $CHECKPOINT"
echo "Options: $OPTIONS"
echo "Output Dir: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "GPUs: $GPUS"
echo "==============================================================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Check if options file exists
if [ ! -f "$OPTIONS" ]; then
    echo "ERROR: Options file not found: $OPTIONS"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs/designB"
mkdir -p "summary/designB"
mkdir -p "checkpoints/designB"

# Run evaluation
echo ""
echo "Starting evaluation..."
echo "Expected duration: ~2-4 hours for 43,784 samples"
echo ""

python entrypoint_designB_eval.py \
    --options "$OPTIONS" \
    --checkpoint "$CHECKPOINT" \
    --name "$EXPERIMENT_NAME" \
    --version "$VERSION" \
    --batch-size "$BATCH_SIZE" \
    --gpus "$GPUS" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "==============================================================================="
echo "Evaluation complete!"
echo "==============================================================================="
echo "Results saved to:"
echo "  - logs/designB/$EXPERIMENT_NAME/sample_results.csv"
echo "  - logs/designB/$EXPERIMENT_NAME/batch_results.csv"
echo "  - logs/designB/$EXPERIMENT_NAME/evaluation_summary.json"
echo "  - $OUTPUT_DIR/*.obj"
echo "==============================================================================="
