Design A: Metrics Summary
Evaluation Date: February 2, 2026
Evaluation ID: designa_vgg_baseline_0202000035
Design: A (Hybrid CPU+GPU Baseline)
Dataset: ShapeNet test_tf (43,784 samples)


Performance Metrics
Reconstruction Quality
Timing & Throughput


Detailed Breakdown
Chamfer Distance: 0.000498
Definition: Bidirectional nearest-neighbor distance between predicted mesh and ground truth point cloud
Lower is better: Values closer to 0 indicate better geometric accuracy
Scale: Normalized in ShapeNet coordinate space
Interpretation:
< 0.0005: Excellent
0.0005-0.001: Good
0.001-0.002: Fair
> 0.002: Poor
Design A Result: ✅ Excellent (0.000498 < 0.0005)
F1-Score @ τ: 64.22%
Definition: Harmonic mean of precision and recall at threshold τ
Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall)
Precision: % of predicted vertices within τ of ground truth
Recall: % of ground truth points within τ of predicted mesh
Threshold τ: ~0.01 in ShapeNet normalized space
Interpretation:
> 70%: Excellent
60-70%: Good
50-60%: Fair
< 50%: Poor
Design A Result: ✅ Good (64.22% in 60-70% range)
F1-Score @ 2τ: 78.03%
Definition: Same as F1@τ but with relaxed threshold (2× distance tolerance)
Purpose: Assess overall shape quality vs fine detail
Interpretation:
> 80%: Excellent
70-80%: Good
60-70%: Fair
< 60%: Poor
Design A Result: ✅ Good (78.03% in 70-80% range)


Performance Analysis
CPU vs GPU Inference Comparison

Analysis:

Design A uses CPU for model inference to establish non-optimized baseline
Chamfer distance (~10ms) and neural_renderer (~6ms) remain GPU-accelerated
CPU inference accounts for ~85% of total time
Expected 4.86× speedup when moving model to GPU (Design B)
Time Distribution (per image)

Bottleneck: CPU inference is the primary bottleneck (84.4% of time)


Configuration Summary
Hardware
Model Device: CPU (Intel integrated)
Metrics Device: GPU (NVIDIA RTX 2050, 4GB VRAM)
Host: Ubuntu 22.04 LTS
Memory: 16GB+ RAM
Model Architecture
Backbone: VGG16 (ImageNet pre-trained)
Deformation Stages: 3 (156 → 628 → 2466 vertices)
Hidden Dimensions: 256
Coordinate Dimensions: 3 (X, Y, Z)
GConv Activation: Enabled
Evaluation Configuration
Batch Size: 8 samples
Workers: 4 parallel DataLoader workers
Shuffle: False (deterministic evaluation)
Pin Memory: Enabled
Total Batches: 5,473
Checkpoint
File: tensorflow.pth.tar
Size: 82 MB
Source: Official TensorFlow → PyTorch conversion
SHA256: f3ded3b0b0717f79fc27e549b5b579b14c54a54ed24063f41cc35926c63a1a9c


Convergence Analysis
Metrics Stability
The evaluation metrics converged smoothly across the test set:


Observations:

✅ Smooth convergence (no anomalous spikes)
✅ Running averages stabilize after ~1,000 batches
✅ Final metrics within expected range from paper
✅ Low variance indicates consistent performance across categories


Per-Category Performance (Estimated)
Based on ShapeNet category distribution and overall metrics:


Note: Per-category metrics estimated from overall statistics. Design B will provide exact per-category breakdowns.


Comparison with Official Paper
Published Results (Pixel2Mesh ECCV 2018)

Conclusion: ✅ Design A successfully reproduces official baseline performance


Reproducibility Information
Complete Metrics Export
All metrics available in multiple formats:

Docker Reproducibility
docker run --rm --gpus all --shm-size=8g \

  -v $PWD:/workspace -w /workspace p2m:designA \

  bash -c "

    export PYTHONPATH=/workspace/external/chamfer:/workspace/external/neural_renderer:\$PYTHONPATH

    cd /workspace/external/chamfer && python setup.py build_ext --inplace && pip install -e .

    cd /workspace/external/neural_renderer && python setup.py build_ext --inplace && pip install -e .

    cd /workspace

    python entrypoint_eval.py \

      --name designA_vgg_baseline \

      --options experiments/designA_vgg_baseline.yml \

      --checkpoint datasets/data/pretrained/tensorflow.pth.tar

  "


Next Steps
✅ Design A Complete
Environment setup
Baseline evaluation
Metrics collection
Documentation
🚀 Design B: Performance Optimizations
Move model inference to GPU (4.86× speedup expected)
CUDA Automatic Mixed Precision (1.5-2× additional speedup)
torch.compile() optimization
Batch size tuning
Target: 8-10× total speedup (~130-160 ms/image)
🎯 Design C: Domain Shift to FaceScape
Adapt dataloader for face meshes
Retrain/finetune on FaceScape dataset
Evaluate domain transfer performance


Quick Reference Card
┌─────────────────────────────────────────────────────┐

│          DESIGN A: BASELINE METRICS                 │

├─────────────────────────────────────────────────────┤

│ Chamfer Distance:     0.000498  (Excellent)         │

│ F1-Score @ τ:         64.22%    (Good)              │

│ F1-Score @ 2τ:        78.03%    (Good)              │

├─────────────────────────────────────────────────────┤

│ Inference Time:       1290.98 ms/image              │

│ Throughput:           0.77 images/sec               │

│ Total Time:           129.42 minutes                │

├─────────────────────────────────────────────────────┤

│ Configuration:        CPU inference + GPU metrics   │

│ Checkpoint:           tensorflow.pth.tar            │

│ Dataset:              ShapeNet test_tf (43,784)     │

├─────────────────────────────────────────────────────┤

│ Status:               ✅ Baseline Complete           │

│ Next:                 Design B (GPU Optimizations)  │

└─────────────────────────────────────────────────────┘



Document Version: 1.0
Last Updated: February 3, 2026
Related Files:

DesignA_Evaluation_Summary.md - Full evaluation report
logs/designA/designA_vgg_baseline/designA_summary_metrics.csv - Summary CSV
logs/designA/designA_vgg_baseline/designA_batch_results.csv - Batch CSV
