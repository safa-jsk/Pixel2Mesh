# Pixel2Mesh Evaluation Metrics Summary

**Last Updated**: February 5, 2026  
**Purpose**: Quick reference for all Design A/B/C evaluation metrics

---

## Design A: Legacy Baseline (VGG16)

### Evaluation: February 5, 2026

| Metric | Value | Notes |
|--------|-------|-------|
| **Chamfer Distance** | 0.000498 | Bidirectional point-to-mesh distance |
| **F1-Score @ τ** | 64.22% | Precision/recall at threshold τ |
| **F1-Score @ 2τ** | 78.03% | Precision/recall at relaxed threshold 2τ |
| **Total Evaluation Time** | 33.43 minutes | 2006.04 seconds |
| **Avg Inference Time** | 253.35 ms | Per image forward pass |
| **Batch Processing Time** | 268.8 ms | Including data loading + metrics |
| **Throughput** | 3.95 img/sec | Processing speed |
| **Total Samples** | 43,784 | ShapeNet test_tf dataset |
| **Total Batches** | 5,473 | batch_size=8 |

### Hardware Configuration

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GeForce RTX 4070 SUPER |
| **VRAM** | 12 GB GDDR6X |
| **Compute Capability** | 8.9 |
| **CUDA Version** | 11.3 (container) |
| **PyTorch Version** | 1.12.1+cu113 |

### Model Configuration

| Setting | Value |
|---------|-------|
| **Backbone** | VGG16 (ImageNet pretrained) |
| **Checkpoint** | tensorflow.pth.tar (82 MB) |
| **Batch Size** | 8 |
| **Workers** | 4 |
| **TensorFlow Alignment** | Enabled |
| **AMP** | Disabled |
| **torch.compile** | Disabled |
| **cuDNN Benchmark** | Enabled |
| **TF32** | Enabled |

---

## Comparison with Paper Results

| Metric | Paper | Design A | Status |
|--------|-------|----------|--------|
| Chamfer Distance | ~0.0004-0.0006 | 0.000498 | ✅ Within range |
| F1-Score @ τ | ~60-65% | 64.22% | ✅ Matches |

---

## Metric Definitions

### Chamfer Distance (CD)
Measures the average bidirectional nearest-neighbor distance between predicted mesh vertices and ground truth point cloud. Lower is better.

$$CD(P, Q) = \frac{1}{|P|}\sum_{p \in P} \min_{q \in Q} ||p - q||_2 + \frac{1}{|Q|}\sum_{q \in Q} \min_{p \in P} ||q - p||_2$$

### F1-Score @ τ
Harmonic mean of precision and recall at distance threshold τ:
- **Precision**: % of predicted points within τ of ground truth
- **Recall**: % of ground truth points within τ of prediction

$$F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$$

### F1-Score @ 2τ
Same as F1@τ but with relaxed threshold (2× distance tolerance), providing a measure of overall shape quality vs fine detail accuracy.

---

## Performance Optimization Notes

### GPU Warmup
- 15 warmup iterations before timing
- First iteration: ~2228 ms (cold start)
- Subsequent iterations: ~142 ms (warmed up)

### Memory Usage
- Batch size 8 fits comfortably in 12 GB VRAM
- Peak memory: ~4-5 GB during inference
- DataLoader workers: 4 (parallel data loading)

---

## File Locations

| File Type | Location |
|-----------|----------|
| Evaluation Logs | `logs/designA/designA_vgg_baseline/` |
| TensorBoard Summaries | `summary/designA/designA_vgg_baseline/` |
| Checkpoints | `checkpoints/designA/designA_vgg_baseline/` |
| Generated Meshes | `datasets/examples_for_poster/` |
| Batch Results CSV | `designA_batch_results.csv` |

---

**Document Version**: 1.0  
**Status**: Design A Complete
