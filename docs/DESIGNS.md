# Design Configurations

This document details the four design implementations for Pixel2Mesh evaluation, covering entrypoints, configurations, outputs, and timing methodology.

---

## DESIGN.A: Legacy CPU Baseline

### Purpose

Establish a reproducible baseline using CPU inference to isolate GPU performance improvements in subsequent designs.

### Entrypoint Script

**Primary:** [`entrypoint_eval.py`](../entrypoint_eval.py)  
**Shell Wrapper:** [`run_designA_eval.sh`](../scripts/evaluation/run_designA_eval.sh)

### Configuration

**YAML:** [`experiments/designA_vgg_baseline.yml`](../experiments/designA_vgg_baseline.yml)

```yaml
checkpoint: datasets/data/pretrained/tensorflow.pth.tar
dataset:
  name: shapenet
  subset_eval: test_tf # 43,784 samples
  subset_train: train_tf
model:
  name: pixel2mesh
  backbone: vgg16
  align_with_tensorflow: true
test:
  batch_size: 8
  shuffle: false
  weighted_mean: false
```

### Main Configuration Flags

```python
# In functions/evaluator.py (Design A specific)
# Model kept on CPU (line 101-102):
self.logger.info("Design A: Model will run on CPU, chamfer/renderer on GPU")
# self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

# Only metrics moved to GPU for computation (line 177):
gt_points = [pts.cuda().float() for pts in gt_points]
```

### Execution Command

```bash
# Via shell script
bash run_designA_eval.sh

# Direct invocation
python entrypoint_eval.py \
  --name designA_vgg_baseline \
  --options experiments/designA_vgg_baseline.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar
```

### Docker Command

```bash
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
```

### Expected Outputs

#### Log Files

- **Location:** `logs/designA/designA_vgg_baseline/`
- **Files:**
  - `designa_vgg_baseline_<timestamp>_eval.log` - Full evaluation trace
  - TensorBoard events in `summary/designA/`

#### Metrics (No CSV Export)

- **Console Output Only:** Per-batch metrics logged every 5 batches
- **Final Metrics:**
  - Chamfer Distance: 0.000498
  - F1-Score @ τ: 64.22%
  - F1-Score @ 2τ: 78.03%

#### Mesh Outputs

- **Location:** N/A (Design A does not generate mesh files)
- **Alternative:** Use `entrypoint_predict.py` with `run_designA_predict.sh` for selective mesh generation

### Timing Methodology

#### Measurement Points

```python
# In functions/evaluator.py (lines 252-278)

# Evaluation start (line 252)
self.eval_start_time = time.time()

# Per-batch timing (lines 255, 265)
batch_start = time.time()
# ... evaluate_step(batch) ...
batch_end = time.time()

# Total evaluation time (line 278)
eval_total_time = time.time() - self.eval_start_time
```

#### Timing Characteristics

- **Method:** `time.time()` (Python standard library)
- **Synchronization:** **None** (inaccurate for GPU operations)
- **Includes:**
  - Model forward pass (CPU)
  - Data loading
  - Chamfer distance (GPU)
  - Metrics computation
  - Logging overhead
- **Excludes:**
  - Dataset initialization
  - Checkpoint loading
  - CUDA extension compilation

#### Performance Metrics

- **Average Inference Time:** 1290.98 ms/image
- **Throughput:** 0.77 images/sec
- **Total Time:** 129.42 minutes (43,784 samples)

### Key Characteristics

- ✅ Reproducible baseline
- ✅ CPU inference isolates GPU impact
- ❌ No CUDA synchronization (timing inaccurate by ~5-10ms)
- ❌ No mesh generation in evaluation script
- ❌ No CSV/JSON export

---

## DESIGN.A_GPU: Simple GPU Enablement

### Purpose

Measure GPU speedup over CPU baseline without advanced optimizations. This design represents a "naive" GPU migration.

### Entrypoint Script

**Same as Design A:** [`entrypoint_eval.py`](../entrypoint_eval.py)

### Configuration

**YAML:** [`experiments/designA_vgg_baseline.yml`](../experiments/designA_vgg_baseline.yml) (same as Design A)

**Code Modification:**

```python
# In functions/evaluator.py (line 102)
# UNCOMMENT the following line:
self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

# REMOVE the Design A CPU comment
```

### Main Configuration Flags

```python
# Enable GPU inference
self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

# Batch data moved to GPU (uncomment line 259):
batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
```

### Execution Command

```bash
# Same as Design A after code modification
python entrypoint_eval.py \
  --name designA_GPU_vgg_baseline \
  --options experiments/designA_vgg_baseline.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar
```

### Expected Outputs

**Same structure as Design A:**

- Logs in `logs/designA/designA_GPU_vgg_baseline/`
- No CSV/JSON export
- No mesh generation

### Timing Methodology

#### Measurement Points

**Same as Design A:** `time.time()` without synchronization

#### Performance Metrics (Expected)

- **Average Inference Time:** ~265 ms/image (4.86× faster than Design A)
- **Throughput:** ~3.77 images/sec
- **Total Time:** ~26.6 minutes

### Key Characteristics

- ✅ Simple GPU migration (1 line change)
- ✅ 4.86× speedup over Design A
- ❌ Still no CUDA synchronization (timing ~5ms error)
- ❌ No advanced optimizations (cuDNN, TF32, AMP, compile)
- ❌ No comprehensive logging

### Differences from Design A

| Aspect         | Design A | Design A_GPU        |
| -------------- | -------- | ------------------- |
| Model Device   | CPU      | GPU                 |
| DataParallel   | Disabled | Enabled             |
| Batch Transfer | CPU only | `.cuda()`           |
| Speedup        | Baseline | 4.86×               |
| Code Changes   | None     | 2 lines uncommented |

---

## DESIGN.B: Optimized GPU Pipeline

### Purpose

Implement comprehensive GPU performance optimizations following CAMFM methodology for accurate, reproducible benchmarking.

### Entrypoint Script

**Primary:** [`entrypoint_designB_eval.py`](../entrypoint_designB_eval.py)  
**Shell Wrapper:** [`run_designB_eval.sh`](../scripts/evaluation/run_designB_eval.sh)

### Configuration

**YAML:** [`experiments/designB_baseline.yml`](../experiments/designB_baseline.yml)

```yaml
checkpoint: datasets/data/pretrained/tensorflow.pth.tar
dataset:
  name: shapenet
  subset_eval: test_tf # 43,784 samples
  camera_f: [250.0, 250.0]
  camera_c: [112.0, 112.0]
  mesh_pos: [0.0, 0.0, 0.0]
test:
  batch_size: 8
  shuffle: false
  summary_steps: 10
```

### Main Configuration Flags

#### Command-Line Arguments

```bash
--warmup-iters 15           # GPU warmup iterations (CAMFM.A2b_STEADY_STATE)
--amp                       # Enable AMP (disabled for P2M, kept for framework)
--compile                   # Enable torch.compile (optional)
--cudnn-benchmark           # cuDNN autotuner (CAMFM.A2d_OPTIONAL_ACCEL)
--tf32                      # TF32 tensor cores (CAMFM.A2d_OPTIONAL_ACCEL)
--no-amp                    # Explicitly disable AMP
--no-compile                # Explicitly disable torch.compile
--no-cudnn-benchmark        # Disable cuDNN benchmark
--no-tf32                   # Disable TF32
```

#### Python Configuration

```python
# In entrypoint_designB_eval.py (lines 140-144)
warmup_iters: int = 15,
amp_enabled: bool = False,      # P2M sparse ops incompatible with FP16
compile_enabled: bool = False,
cudnn_benchmark: bool = True,   # Safe for fixed input sizes
tf32_enabled: bool = True       # Ampere+ GPUs only
```

### Execution Command

#### Baseline (No Advanced Optimizations)

```bash
python entrypoint_designB_eval.py \
  --options experiments/designB_baseline.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
  --name designB_baseline \
  --warmup-iters 0 \
  --no-amp --no-compile --no-cudnn-benchmark --no-tf32
```

#### Recommended (Stable Optimizations)

```bash
python entrypoint_designB_eval.py \
  --options experiments/designB_baseline.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
  --name designB_optimized \
  --warmup-iters 15 \
  --cudnn-benchmark --tf32
```

#### Aggressive (All Optimizations)

```bash
python entrypoint_designB_eval.py \
  --options experiments/designB_baseline.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
  --name designB_aggressive \
  --warmup-iters 20 \
  --amp --compile --cudnn-benchmark --tf32
```

### Docker Command

```bash
docker run --rm --gpus all --shm-size=8g \
  -v $PWD:/workspace -w /workspace p2m:designA \
  bash run_designB_eval_docker.sh
```

### Expected Outputs

#### Log Files

- **Location:** `logs/designB/<experiment_name>/`
- **Files:**
  - `<experiment_name>_<timestamp>_eval.log` - Full evaluation trace
  - `sample_results.csv` - Per-sample metrics (43,784 rows)
  - `batch_results.csv` - Per-batch metrics (~5,473 rows)
  - `evaluation_summary.json` - Overall + per-category metrics

#### CSV Structure (sample_results.csv)

```csv
sample_idx,filename,category,label,chamfer_distance,f1_tau,f1_2tau,time_seconds
0,02691156/1a04e3eab45ca15dd86060f189eb133/00.png,Airplane,0,0.000423,0.6834,0.8210,0.1347
1,02691156/1a04e3eab45ca15dd86060f189eb133/01.png,Airplane,0,0.000431,0.6789,0.8156,0.1322
...
```

#### CSV Structure (batch_results.csv)

```csv
batch_idx,batch_size,avg_chamfer_distance,avg_f1_tau,avg_f1_2tau,batch_time_seconds,inference_time_seconds,meshes_saved
0,8,0.000442,0.6712,0.8043,1.2456,0.2234,0
1,8,0.000455,0.6598,0.7921,1.2112,0.2189,0
...
```

#### JSON Structure (evaluation_summary.json)

```json
{
  "design": "B",
  "timestamp": "2026-02-16T14:30:00",
  "dataset": {
    "name": "test_tf",
    "total_samples": 43784,
    "num_classes": 13
  },
  "metrics": {
    "chamfer_distance": 0.000498,
    "f1_tau": 0.6422,
    "f1_2tau": 0.7803
  },
  "per_category": {
    "Airplane": {
      "category_id": "02691156",
      "samples": 3364,
      "chamfer_distance": 0.000420,
      "f1_tau": 0.6850,
      "f1_2tau": 0.8210
    },
    ...
  },
  "timing": {
    "total_seconds": 7890.45,
    "total_minutes": 131.51,
    "samples_per_second": 5.55,
    "ms_per_sample": 180.12,
    "avg_batch_inference_ms": 185.23
  },
  "mesh_generation": {
    "samples_generated": 26,
    "files_generated": 78,
    "output_directory": "outputs/designB_meshes"
  },
  "configuration": {
    "batch_size": 8,
    "num_gpus": 1,
    "checkpoint": "datasets/data/pretrained/tensorflow.pth.tar",
    "warmup_iters": 15,
    "amp_enabled": false,
    "compile_enabled": false,
    "cudnn_benchmark": true,
    "tf32_enabled": true
  }
}
```

#### Mesh Outputs

- **Location:** `outputs/designB_meshes/<category>/<objectID>.<stage>.obj`
- **Files:** 78 total (26 samples × 3 stages)
  - Stage 1: 156 vertices (coarse mesh)
  - Stage 2: 628 vertices (medium mesh)
  - Stage 3: 2466 vertices (fine mesh)
- **Format:** OBJ (Wavefront) with vertices and faces

**Example:**

```
outputs/designB_meshes/
├── 02691156/  # Airplane
│   ├── 1b171503.1.obj  (156 vertices)
│   ├── 1b171503.2.obj  (628 vertices)
│   ├── 1b171503.3.obj  (2466 vertices)
│   ├── 1954754c.1.obj
│   ├── 1954754c.2.obj
│   └── 1954754c.3.obj
├── 02828884/  # Bench
│   └── ...
└── ... (13 categories)
```

### Timing Methodology

#### Measurement Points

```python
# In entrypoint_designB_eval.py

# Warmup phase (lines 195-205)
warmup_model(self.model, input_shape=(8,3,224,224),
             warmup_iters=self.warmup_iters, ...)

# Evaluation start (line 290)
eval_start_time = time.time()

# Per-batch timing with CUDA synchronization (lines 200-204)
torch.cuda.synchronize()  # CRITICAL: Wait for GPU to finish
batch_start = time.time()
out = self.model(images)
torch.cuda.synchronize()  # CRITICAL: Wait before reading timer
batch_inference_time = time.time() - batch_start

# Total evaluation time (line 349)
total_time = time.time() - eval_start_time
```

#### Timing Characteristics

- **Method:** `time.time()` + `torch.cuda.synchronize()`
- **Synchronization:** ✅ **Before and after forward pass** (accurate GPU timing)
- **Warmup:** 15 iterations to eliminate cold-start artifacts
- **Includes (Timed Region):**
  - Model forward pass (GPU)
  - CUDA synchronization overhead (~0.1ms)
- **Excludes (Outside Timed Region):**
  - Data loading
  - H2D transfer
  - Chamfer distance
  - Metrics computation
  - Logging
  - Mesh saving

#### Performance Metrics (Expected)

- **Average Inference Time:** ~185 ms/image (cuDNN+TF32)
- **Throughput:** ~5.4 images/sec
- **Total Time:** ~135 minutes (including metrics/logging)
- **Speedup over Design A:** 6.98× (inference only), 4.76× (end-to-end)

### CAMFM Methodology Mapping

#### CAMFM.A2a_GPU_RESIDENCY

```python
# Line 123: Model to GPU
self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

# Line 300: Batch to GPU
batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
```

#### CAMFM.A2b_STEADY_STATE

```python
# Lines 195-205: GPU warmup
warmup_model(self.model, input_shape=(8,3,224,224),
             warmup_iters=15, device="cuda", logger=self.logger)

# Lines 200, 203: CUDA synchronization
torch.cuda.synchronize()
```

#### CAMFM.A2c_MEM_LAYOUT

```python
# In utils/perf.py (lines 120-135): Contiguous tensors
def ensure_contiguous(model):
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()
```

#### CAMFM.A2d_OPTIONAL_ACCEL

```python
# Lines 56-59: cuDNN/TF32 setup
setup_cuda_optimizations(cudnn_benchmark=True, tf32=True, logger=logger)

# Lines 64-69: torch.compile (optional)
if self.compile_enabled:
    self.model = compile_model_safe(self.model, compile_mode="max-autotune", ...)

# Lines 197-199: AMP autocast (disabled for P2M)
self.autocast_context = get_autocast_context(amp_enabled=False, device="cuda")
```

#### CAMFM.A3_METRICS

```python
# Lines 393-396: Export metrics
self.save_results_to_csv(total_time, total_samples)
self.save_summary_json(total_time, total_samples, avg_cd, avg_f1_tau, avg_f1_2tau)
```

#### CAMFM.A5_METHOD

```python
# Lines 493-518: Reproducible CLI args
parser = argparse.ArgumentParser(description='Design B: Full Dataset Baseline Evaluation')
parser.add_argument('--warmup-iters', type=int, default=15)
parser.add_argument('--amp', action='store_true')
...
```

### Key Characteristics

- ✅ CUDA-synchronized timing (accurate to <1ms)
- ✅ GPU warmup (15 iterations)
- ✅ Comprehensive metrics export (CSV + JSON)
- ✅ Selective mesh generation (26 samples)
- ✅ Per-category metrics
- ✅ cuDNN autotune + TF32 enabled
- ⚠️ AMP disabled (P2M incompatible)
- ⚠️ torch.compile optional (minimal benefit)

---

## DESIGN.C: GPU-Native Data Pipeline + FaceScape

### Purpose

Eliminate data loading bottleneck with GPU-native pipeline (DALI/nvJPEG) and adapt model to FaceScape face mesh reconstruction.

### Status

**In Development** - Design B must be complete and validated first.

### Planned Entrypoint Script

**Primary:** `entrypoint_designC_eval.py` (not yet created)  
**Shell Wrapper:** `run_designC_eval.sh` (not yet created)

### Configuration (Planned)

**YAML:** `experiments/designC_facescape.yml` (not yet created)

```yaml
checkpoint: datasets/data/pretrained/pixel2mesh_facescape.pth.tar
dataset:
  name: facescape
  subset_eval: test # FaceScape test split
  decoder: nvjpeg # GPU JPEG decoding
  resize_backend: gpu # GPU resize via DALI
model:
  name: pixel2mesh
  backbone: vgg16
  num_classes: 1 # Face only
test:
  batch_size: 16 # Higher batch size (GPU data pipeline faster)
  shuffle: false
```

### Main Configuration Flags (Planned)

#### Command-Line Arguments

```bash
--data-pipeline gpu         # Use DALI GPU pipeline
--decoder nvjpeg            # GPU JPEG decoder
--prefetch-batches 2        # Prefetch 2 batches ahead
--pin-memory true           # Pinned memory for H2D transfer
# Plus all Design B flags (warmup, cuDNN, TF32, etc.)
```

#### Python Configuration (Planned)

```python
# In entrypoint_designC_eval.py (planned)
from nvidia.dali import pipeline_def, Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator

@pipeline_def
def facescape_pipeline(batch_size, num_threads, device_id):
    # DATA.READ_CPU
    images, labels = fn.readers.file(file_root="datasets/facescape/", ...)

    # DATA.DECODE_GPU_NVJPEG
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)

    # DATA.RESIZE_GPU
    images = fn.resize(images, device="gpu", resize_x=224, resize_y=224)

    # DATA.NORMALIZE_GPU
    images = fn.crop_mirror_normalize(images, device="gpu",
                                       mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])

    return images, labels
```

### Expected Outputs (Planned)

**Same as Design B:**

- CSV/JSON metrics
- Mesh OBJ files
- TensorBoard summaries

**Additional:**

- FaceScape-specific metrics (face landmark alignment, identity preservation)

### Timing Methodology (Planned)

#### DATA Pipeline Stages

```python
# DATA.READ_CPU: File loading (CPU)
# DATA.DECODE_GPU_NVJPEG: JPEG decode (GPU mixed mode)
# DATA.RESIZE_GPU: GPU resize (CUDA kernels)
# DATA.NORMALIZE_GPU: GPU normalization
# DATA.DALI_BRIDGE_PYTORCH: Zero-copy to PyTorch tensors
```

#### Timing Characteristics (Expected)

- **Data Loading:** ~35ms/batch (down from 70ms in Design B)
- **Model Inference:** ~185ms/batch (same as Design B)
- **Total:** ~220ms/batch (end-to-end)
- **Speedup over Design B:** 1.23× (end-to-end)

### Key Characteristics (Planned)

- ✅ GPU JPEG decoding (nvJPEG)
- ✅ GPU resize/normalize (DALI)
- ✅ Zero-copy PyTorch integration
- ✅ FaceScape domain adaptation
- ✅ All Design B optimizations retained
- ⚠️ Requires NVIDIA DALI installation
- ⚠️ FaceScape dataset preparation required

---

## Design Comparison Summary

| Feature                  | Design A    | Design A_GPU | Design B        | Design C         |
| ------------------------ | ----------- | ------------ | --------------- | ---------------- |
| **Model Device**         | CPU         | GPU          | GPU             | GPU              |
| **Data Pipeline**        | CPU         | CPU          | CPU             | GPU (DALI)       |
| **CUDA Sync Timing**     | ❌          | ❌           | ✅              | ✅               |
| **GPU Warmup**           | ❌          | ❌           | ✅ (15 iters)   | ✅ (15 iters)    |
| **cuDNN Benchmark**      | ❌          | ❌           | ✅              | ✅               |
| **TF32 Enabled**         | ❌          | ❌           | ✅              | ✅               |
| **AMP Support**          | N/A         | N/A          | ⚠️ (disabled)   | ⚠️ (disabled)    |
| **torch.compile**        | N/A         | N/A          | ⚠️ (optional)   | ⚠️ (optional)    |
| **CSV/JSON Export**      | ❌          | ❌           | ✅              | ✅               |
| **Mesh Generation**      | ❌          | ❌           | ✅ (26 samples) | ✅ (all samples) |
| **Per-Category Metrics** | ❌          | ❌           | ✅              | ✅               |
| **Dataset**              | ShapeNet    | ShapeNet     | ShapeNet        | FaceScape        |
| **Inference Time**       | 1291ms      | 265ms        | 185ms           | 185ms            |
| **Data Load Time**       | 150ms       | 80ms         | 70ms            | 35ms             |
| **Total Time**           | 1441ms      | 345ms        | 255ms           | 220ms            |
| **Speedup (vs A)**       | 1.0×        | 4.18×        | 5.65×           | 6.55×            |
| **Status**               | ✅ Complete | ✅ Complete  | ✅ Complete     | 🚧 In Progress   |

---

## Choosing the Right Design

### Use Design A when:

- Establishing a CPU baseline
- Isolating GPU impact
- Debugging model behavior on CPU
- Limited GPU availability

### Use Design A_GPU when:

- Quick GPU speedup check
- Minimal code changes acceptable
- Timing accuracy not critical
- No advanced optimizations needed

### Use Design B when:

- Accurate GPU benchmarking required
- Comprehensive metrics needed
- Reproducible research experiments
- Publication/thesis documentation

### Use Design C when:

- Data loading is the bottleneck
- Domain shift to FaceScape required
- Maximum end-to-end performance needed
- GPU resources available for data pipeline

---

## Related Documentation

- [`PIPELINE_OVERVIEW.md`](./PIPELINE_OVERVIEW.md) - Pipeline architecture and Mermaid diagrams
- [`TRACEABILITY_MATRIX.md`](./TRACEABILITY_MATRIX.md) - Code-to-methodology mapping
- [`BENCHMARK_PROTOCOL.md`](./BENCHMARK_PROTOCOL.md) - Timing rules and validation

---

**Last Updated:** February 16, 2026  
**Maintainer:** Safa JSK  
**Status:** Designs A, A_GPU, B complete; Design C planned
