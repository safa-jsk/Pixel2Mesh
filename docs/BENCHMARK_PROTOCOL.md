# Benchmark Protocol

This document specifies the standardized timing methodology, warmup procedures, synchronization rules, and validation criteria for Pixel2Mesh performance benchmarking across all designs.

---

## 1. Timing Methodology

### 1.1 CUDA-Synchronized Timing (Design B/C)

**Required for accurate GPU benchmarking.**

#### Synchronization Pattern

```python
import time
import torch

# CRITICAL: Synchronize BEFORE starting timer
torch.cuda.synchronize()
start_time = time.time()

# GPU operation (asynchronous kernel launches)
output = model(input_tensor)

# CRITICAL: Synchronize AFTER operation completes
torch.cuda.synchronize()
end_time = time.time()

elapsed = end_time - start_time  # Accurate GPU time
```

#### Why Synchronization is Required

- **GPU Asynchrony:** CUDA kernel launches return immediately to CPU (async execution)
- **Without Sync:** `time.time()` measures kernel launch overhead (~0.1ms), not actual compute
- **With Sync:** CPU waits for GPU to finish, capturing true compute time
- **Accuracy:** <1ms error with sync vs ±5-10ms without

#### Implementation in Design B

```python
# In entrypoint_designB_eval.py:200-204
torch.cuda.synchronize()  # Line 200
batch_start = time.time()
out = self.model(images)  # Line 202
torch.cuda.synchronize()  # Line 203
batch_inference_time = time.time() - batch_start
```

### 1.2 CPU Timing (Design A)

**Acceptable for CPU-only operations, but inaccurate for GPU.**

```python
import time

start_time = time.time()
output = model(input_tensor)  # CPU execution is synchronous
end_time = time.time()

elapsed = end_time - start_time  # Accurate for CPU
```

**Limitation:** If model uses GPU operations (e.g., Design A metrics on GPU), timing is inaccurate.

### 1.3 CudaTimer Utility

**Recommended wrapper for cleaner code.**

```python
from utils.perf import CudaTimer

with CudaTimer("Forward Pass") as timer:
    output = model(input_tensor)

print(f"Elapsed: {timer.elapsed:.4f} seconds")
```

**Benefits:**

- Automatic synchronization
- Context manager ensures cleanup
- Optional logging
- Reusable across codebase

---

## 2. Warmup Protocol

### 2.1 Purpose of Warmup

Warmup eliminates **cold-start timing artifacts** from:

1. **CUDA Context Initialization:** First GPU operation triggers context setup (~500ms)
2. **cuDNN Autotuner:** First forward pass benchmarks convolution algorithms (~2-5s)
3. **JIT Compilation:** torch.compile and cuDNN JIT kernels (~1-3s)
4. **Memory Allocation:** First malloc allocates GPU memory pools (~100ms)
5. **Driver Overhead:** NVIDIA driver initialization (~50ms)

**After warmup:** Subsequent iterations have stable, representative timing.

### 2.2 Warmup Procedure

#### Design B Implementation

```python
from utils.perf import warmup_model

# In entrypoint_designB_eval.py:195-205
warmup_model(
    model=self.model,
    input_shape=(8, 3, 224, 224),  # Batch size 8, RGB 224x224
    warmup_iters=15,                # 15 iterations recommended
    device="cuda",
    amp_enabled=False,              # Match eval settings
    logger=self.logger
)
```

#### Warmup Function (utils/perf.py:89-124)

```python
def warmup_model(model, input_shape, warmup_iters=15, device="cuda",
                 amp_enabled=False, logger=None):
    """
    Run dummy forward passes to eliminate cold-start artifacts.

    Args:
        model: PyTorch model (already on GPU)
        input_shape: (batch, channels, height, width)
        warmup_iters: Number of warmup iterations (default: 15)
        device: "cuda" or "cpu"
        amp_enabled: Whether to use AMP during warmup
        logger: Optional logger for progress
    """
    model.eval()
    dummy_input = torch.randn(input_shape, device=device)

    autocast_context = get_autocast_context(amp_enabled, device)

    with torch.no_grad():
        for i in range(warmup_iters):
            with autocast_context:
                _ = model(dummy_input)

            if i == 0:
                # First iteration: CUDA context init, cuDNN autotune
                torch.cuda.synchronize()
                if logger:
                    logger.info(f"  Warmup iter {i+1}/{warmup_iters}: CUDA context initialized")
            elif i == warmup_iters - 1:
                torch.cuda.synchronize()
                if logger:
                    logger.info(f"  Warmup complete: {warmup_iters} iterations")

    # Clear GPU cache after warmup
    torch.cuda.empty_cache()
```

### 2.3 Warmup Configuration

| Design           | Warmup Iterations | Rationale                                |
| ---------------- | ----------------- | ---------------------------------------- |
| **Design A**     | 0 (no warmup)     | CPU execution, no GPU warmup needed      |
| **Design A_GPU** | 0 (optional: 5)   | Simple GPU, warmup not critical          |
| **Design B**     | 15 (required)     | Accurate benchmarking, CUDA sync enabled |
| **Design C**     | 15 (required)     | Same as Design B + data pipeline warmup  |

**Recommended:** 15 iterations for Design B/C (empirically determined to stabilize timing)

---

## 3. Timing Boundaries

### 3.1 What to Include in Timed Region

**Design B Timed Region (Inference Only):**

```python
# INCLUDED:
torch.cuda.synchronize()
start = time.time()

# ✅ Model forward pass (VGG16 + GCNs)
output = model(images)

torch.cuda.synchronize()
end = time.time()
# INCLUDED: Model compute time only

inference_time = end - start  # Pure model inference
```

**Why Exclude Other Operations:**

- **Data Loading:** Variable I/O latency (disk speed, file system cache)
- **H2D Transfer:** Not a model bottleneck (5ms vs 185ms inference)
- **Metrics Computation:** Chamfer distance is a separate evaluation concern
- **Logging:** File I/O variability

### 3.2 What to Exclude from Timed Region

**Excluded Operations:**

```python
# ❌ EXCLUDE: Data loading (outside timed region)
batch = next(data_loader)  # Disk I/O, file system cache effects

# ❌ EXCLUDE: H2D transfer (outside timed region)
batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# ✅ START TIMED REGION
torch.cuda.synchronize()
start = time.time()
output = model(images)  # ONLY MODEL INFERENCE
torch.cuda.synchronize()
end = time.time()
# ✅ END TIMED REGION

# ❌ EXCLUDE: Metrics computation (outside timed region)
chamfer_distance = compute_chamfer(output, ground_truth)

# ❌ EXCLUDE: Logging (outside timed region)
logger.info(f"Batch {i}: CD={chamfer_distance}")
```

### 3.3 End-to-End vs Inference-Only Timing

**Two Timing Modes:**

| Mode               | Includes                                        | Use Case                                            |
| ------------------ | ----------------------------------------------- | --------------------------------------------------- |
| **Inference-Only** | Model forward pass only                         | GPU optimization comparison, model speedup analysis |
| **End-to-End**     | Data load + H2D + inference + metrics + logging | Real-world deployment, total pipeline cost          |

**Design B Reports Both:**

```python
# Inference-only (line 204)
batch_inference_time = time.time() - batch_start  # ~185ms

# End-to-end (line 278)
total_time = time.time() - eval_start_time  # ~255ms/batch
```

---

## 4. Environment Configuration

### 4.1 CUDA Optimizations

**cuDNN Benchmark Mode:**

```python
import torch

# Enable cuDNN autotuner (Design B/C)
torch.backends.cudnn.benchmark = True  # Autotune conv algorithms

# Disable for variable input sizes (not applicable to P2M)
torch.backends.cudnn.benchmark = False
```

**When to Enable:**

- ✅ Fixed input sizes (P2M: always 224×224)
- ✅ Repeated evaluation (autotuner cost amortized)
- ❌ Variable input sizes (autotune overhead per size)

**TF32 Tensor Cores (Ampere+ GPUs):**

```python
# Enable TF32 for faster matmul/conv (Design B/C)
torch.backends.cuda.matmul.allow_tf32 = True   # Matrix ops
torch.backends.cudnn.allow_tf32 = True         # Convolution ops
```

**GPU Compatibility:**

- ✅ Ampere+ (RTX 30xx, A100, RTX 2050) - Compute capability ≥ 8.0
- ❌ Turing (RTX 20xx) - Compute capability 7.5 (no TF32)
- ❌ Pascal/Volta - Compute capability < 8.0 (no TF32)

**Verification:**

```python
import torch

device_cap = torch.cuda.get_device_capability()
if device_cap[0] >= 8:
    print(f"✅ TF32 supported (compute capability {device_cap[0]}.{device_cap[1]})")
else:
    print(f"❌ TF32 not supported (compute capability {device_cap[0]}.{device_cap[1]})")
```

### 4.2 AMP Configuration

**Automatic Mixed Precision (FP16/BF16):**

```python
from torch.cuda.amp import autocast

# Enable AMP (if compatible)
with autocast(dtype=torch.float16):
    output = model(input)
```

**Pixel2Mesh Compatibility:**

- ❌ **Disabled in Design B:** P2M uses sparse graph operations incompatible with FP16
- ✅ **Framework Ready:** AMP context prepared for future dense model variants

### 4.3 torch.compile Configuration

**PyTorch 2.x Graph Optimization:**

```python
import torch

# Compile model (optional in Design B)
compiled_model = torch.compile(
    model,
    mode="max-autotune",  # Aggressive optimization
    backend="inductor"     # Default backend
)
```

**P2M Compatibility:**

- ⚠️ **Minimal Benefit:** P2M has dynamic graph topology (variable vertex connections)
- ✅ **Safe to Enable:** No accuracy degradation
- ~5ms speedup (2.7% improvement) - not significant for P2M

---

## 5. Reproducibility Requirements

### 5.1 Random Seed Fixing

**For Perfect Reproducibility:**

```python
import random
import numpy as np
import torch

# Fix all random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU

# Deterministic operations (slight performance cost)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Conflict with cuDNN autotune!
```

**Trade-off:**

- ✅ **Perfect Reproducibility:** Same results every run
- ❌ **Performance Cost:** ~10% slower (deterministic ops, no cuDNN autotune)

**Design B Choice:**

- **Seeds NOT fixed** (prioritize performance)
- **cuDNN benchmark enabled** (non-deterministic but faster)
- **Acceptable variance:** <1% run-to-run variation in metrics

### 5.2 Configuration Logging

**All configurations must be logged:**

```python
config = {
    "warmup_iters": 15,
    "batch_size": 8,
    "cudnn_benchmark": True,
    "tf32_enabled": True,
    "amp_enabled": False,
    "compile_enabled": False,
    "checkpoint": "tensorflow.pth.tar",
    "sha256": "f3ded3b0..."
}

# Save to JSON for audit trail
with open("evaluation_config.json", "w") as f:
    json.dump(config, f, indent=2)
```

**Required Fields:**

- Warmup iterations
- Batch size
- cuDNN/TF32/AMP/compile settings
- Checkpoint path + SHA256 hash
- PyTorch/CUDA versions
- GPU model + driver version

---

## 6. Validation Criteria

### 6.1 Timing Stability

**Acceptable Run-to-Run Variance:**

```
Coefficient of Variation (CV) = (std / mean) * 100%

✅ CV < 2%: Excellent stability
⚠️ CV 2-5%: Acceptable (warmup may be insufficient)
❌ CV > 5%: Unstable (check synchronization, warmup, system load)
```

**Example:**

```python
# Run 5 evaluation trials
times = [185.2, 186.1, 184.8, 185.7, 185.4]  # ms

mean_time = np.mean(times)  # 185.44 ms
std_time = np.std(times)    # 0.487 ms
cv = (std_time / mean_time) * 100  # 0.26%

print(f"✅ CV = {cv:.2f}% < 2% (stable)")
```

### 6.2 Metrics Accuracy

**Chamfer Distance Validation:**

```
✅ Design A_GPU == Design B (same GPU inference)
✅ Design B cuDNN/TF32 ≈ Design B baseline (< 0.1% difference)
❌ Design A ≠ Design A_GPU (CPU vs GPU rounding differences ~0.01%)
```

**F1-Score Validation:**

```
✅ F1@τ within ±1% across designs (threshold-based metric more stable)
✅ Per-category F1 variance < 2% (category-specific evaluation)
```

### 6.3 Mesh Quality Validation

**OBJ File Checks:**

```bash
# Verify vertex count per stage
for obj in outputs/designB_meshes/*/*.1.obj; do
    vert_count=$(grep -c "^v " $obj)
    if [ $vert_count -ne 156 ]; then
        echo "❌ Stage 1 should have 156 vertices, got $vert_count"
    fi
done

# Verify file sizes (stage 3 should be largest)
for stage in 1 2 3; do
    find outputs/designB_meshes/ -name "*.$stage.obj" -size -1k
    # Should return 0 files (all > 1KB)
done
```

**MeshLab Validation:**

```bash
# Load random sample in MeshLab
meshlabserver -i outputs/designB_meshes/02691156/1b171503.3.obj \
              -o /tmp/test_render.png \
              -m vc  # Vertex colors

# Check for non-manifold edges, degenerate faces
```

---

## 7. Benchmark Execution Checklist

### 7.1 Pre-Benchmark

- [ ] **System Idle:** Close unnecessary applications (browsers, IDEs)
- [ ] **GPU Idle:** No other CUDA processes running (`nvidia-smi`)
- [ ] **CPU Governor:** Set to "performance" mode (Linux)
- [ ] **Thermals:** GPU temperature < 75°C before starting
- [ ] **Disk Space:** ≥10GB free for logs/meshes
- [ ] **CUDA Extensions:** Chamfer/neural_renderer compiled
- [ ] **Checkpoint Verified:** SHA256 hash matches expected
- [ ] **Dataset Present:** test_tf.txt exists with 43,784 lines

### 7.2 During Benchmark

- [ ] **Monitor GPU Utilization:** `watch -n 1 nvidia-smi` (should be 90-100%)
- [ ] **Check Logs:** No CUDA OOM errors
- [ ] **Validate Progress:** Metrics logged every N batches
- [ ] **Temperature Monitoring:** GPU temp < 85°C (throttling threshold)

### 7.3 Post-Benchmark

- [ ] **Logs Generated:** `*.log`, `*.csv`, `*.json` files present
- [ ] **Metrics Valid:** CD > 0, F1 in [0, 1] range
- [ ] **Mesh Files Count:** 78 OBJ files for Design B (26 samples × 3 stages)
- [ ] **Timing Reported:** Inference time + end-to-end time logged
- [ ] **Config Saved:** evaluation_summary.json contains full config
- [ ] **Stability Check:** Run 3 trials, compute CV < 2%

---

## 8. Common Pitfalls

### 8.1 Missing CUDA Synchronization

**Problem:**

```python
# ❌ WRONG: No synchronization
start = time.time()
output = model(input)  # Async kernel launch
end = time.time()  # Only measures launch overhead (~0.1ms)
```

**Solution:**

```python
# ✅ CORRECT: Synchronize before/after
torch.cuda.synchronize()
start = time.time()
output = model(input)
torch.cuda.synchronize()
end = time.time()
```

### 8.2 Insufficient Warmup

**Problem:** First iteration is 10× slower (cold-start artifacts)

**Solution:** Use 15+ warmup iterations, discard first iteration timing

### 8.3 Variable Input Sizes with cuDNN Benchmark

**Problem:** cuDNN autotunes per input size, causing slowdowns for variable sizes

**Solution:** Disable `cudnn.benchmark` if input sizes vary (not applicable to P2M)

### 8.4 GPU Throttling

**Problem:** GPU temperature > 85°C causes thermal throttling (clock speed reduction)

**Solution:** Monitor temperature, improve cooling, reduce batch size

### 8.5 Background Processes

**Problem:** Other CUDA processes compete for GPU (mining, rendering, other experiments)

**Solution:** Ensure GPU is idle before benchmarking (`nvidia-smi` shows 0% utilization)

---

## 9. Design-Specific Protocols

### 9.1 Design A Protocol

```bash
# 1. Ensure CPU-only execution
# functions/evaluator.py line 102 should be commented out

# 2. Run evaluation
python entrypoint_eval.py \
  --name designA_vgg_baseline \
  --options experiments/designA_vgg_baseline.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar

# 3. Extract metrics from log
grep "Test \[" logs/designA/designA_vgg_baseline/*.log | tail -1
```

**Timing Characteristics:**

- No GPU synchronization (inaccurate)
- No warmup (not needed for CPU)
- `time.time()` measures CPU execution

### 9.2 Design B Protocol

```bash
# 1. Run with recommended settings
python entrypoint_designB_eval.py \
  --options experiments/designB_baseline.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
  --name designB_optimized \
  --warmup-iters 15 \
  --cudnn-benchmark --tf32

# 2. Validate outputs
ls -lh logs/designB/designB_optimized/
# Should see: *.log, sample_results.csv, batch_results.csv, evaluation_summary.json

ls -lh outputs/designB_meshes/*/*.3.obj | wc -l
# Should return: 26 (26 samples)

# 3. Check metrics stability (run 3 times)
for i in {1..3}; do
  python entrypoint_designB_eval.py ... --name run_$i
  grep "ms_per_sample" logs/designB/run_$i/evaluation_summary.json
done
```

**Timing Characteristics:**

- CUDA synchronization (accurate)
- 15 warmup iterations (stable)
- Inference-only timing reported

### 9.3 Design C Protocol (Planned)

```bash
# 1. Install NVIDIA DALI
pip install --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda110

# 2. Run with GPU data pipeline
python entrypoint_designC_eval.py \
  --options experiments/designC_facescape.yml \
  --checkpoint datasets/data/pretrained/pixel2mesh_facescape.pth.tar \
  --name designC_facescape \
  --data-pipeline gpu \
  --decoder nvjpeg \
  --prefetch-batches 2 \
  --warmup-iters 20  # Include data pipeline warmup
```

**Timing Characteristics:**

- CUDA synchronization (accurate)
- 20 warmup iterations (data pipeline + model)
- Separate data pipeline and inference timing

---

## 10. Reporting Template

### 10.1 Benchmark Report Structure

```markdown
# Design X Benchmark Report

## Configuration

- **Date:** 2026-02-16
- **Design:** B (Optimized GPU)
- **Checkpoint:** tensorflow.pth.tar (SHA256: f3ded3b0...)
- **GPU:** NVIDIA RTX 2050 (4GB VRAM)
- **Driver:** 580.126.09
- **CUDA:** 11.3.1
- **PyTorch:** 1.12.1+cu113

## Optimizations

- [x] GPU Warmup: 15 iterations
- [x] CUDA Synchronization: Enabled
- [x] cuDNN Benchmark: Enabled
- [x] TF32: Enabled
- [ ] AMP: Disabled (P2M incompatible)
- [ ] torch.compile: Disabled (minimal benefit)

## Timing Results

| Metric                | Value  | Unit                  |
| --------------------- | ------ | --------------------- |
| Inference Time (mean) | 185.23 | ms/image              |
| Inference Time (std)  | 0.89   | ms                    |
| CV                    | 0.48%  | (excellent stability) |
| Throughput            | 5.40   | images/sec            |
| End-to-End Time       | 255.12 | ms/batch              |

## Quality Metrics

| Metric           | Value    |
| ---------------- | -------- |
| Chamfer Distance | 0.000498 |
| F1-Score @ τ     | 64.22%   |
| F1-Score @ 2τ    | 78.03%   |

## Validation

- [x] Timing stability: CV = 0.48% < 2%
- [x] Metrics match Design A: CD difference < 0.1%
- [x] Mesh files generated: 78 OBJ files
- [x] Logs complete: CSV + JSON exported
```

---

## Related Documentation

- [`PIPELINE_OVERVIEW.md`](./PIPELINE_OVERVIEW.md) - Architecture and flow diagrams
- [`DESIGNS.md`](./DESIGNS.md) - Design configurations and execution
- [`TRACEABILITY_MATRIX.md`](./TRACEABILITY_MATRIX.md) - Code-to-methodology mapping

---

**Last Updated:** February 16, 2026  
**Maintainer:** Safa JSK  
**Status:** Design A, A_GPU, B protocols complete; Design C planned
