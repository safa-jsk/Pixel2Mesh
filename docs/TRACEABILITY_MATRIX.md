# Traceability Matrix

This document maps each pipeline stage to specific code locations, functionality, performance impact, and evidence artifacts.

---

## Matrix Legend

- **StageID:** Design/methodology identifier (DESIGN._, CAMFM._, DATA.\*)
- **File Path:** Relative path from repository root
- **Function/Class:** Specific implementation
- **Description:** What the code does
- **Performance Impact:** Measured speedup/change
- **Evidence Artifact:** Log/CSV/mesh file location

---

## DESIGN.A: Legacy CPU Baseline

| StageID                     | File Path                        | Function/Class                        | Description                                           | Performance Impact                  | Evidence Artifact                                              |
| --------------------------- | -------------------------------- | ------------------------------------- | ----------------------------------------------------- | ----------------------------------- | -------------------------------------------------------------- |
| **DESIGN.A.ENTRY**          | `entrypoint_eval.py`             | `main()`                              | Parse CLI args, load config, initialize evaluator     | Baseline setup                      | `logs/designA/designA_vgg_baseline/*.log`                      |
| **DESIGN.A.INIT**           | `functions/evaluator.py:68-86`   | `Evaluator.init_fn()`                 | Initialize model on CPU, keep chamfer/renderer on GPU | CPU inference decision              | Log line: "Design A: Model will run on CPU"                    |
| **DESIGN.A.CPU_MODEL**      | `functions/evaluator.py:101-102` | `Evaluator.init_fn()`                 | Comment out `.cuda()` to keep model on CPU            | -84.4% performance (CPU bottleneck) | `DesignA_Evaluation_Summary.md` line 191                       |
| **DESIGN.A.DATA_LOAD**      | `functions/base.py:68-74`        | `CheckpointRunner.load_dataset()`     | Load ShapeNet dataset from test_tf.txt                | 150ms/batch data loading            | `designA_batch_results.csv`                                    |
| **DESIGN.A.FWD_CPU**        | `functions/evaluator.py:158-167` | `Evaluator.evaluate_step()`           | Forward pass on CPU (VGG16 + GCNs)                    | 1090ms/image (main bottleneck)      | `DesignA_Evaluation_Summary.md` line 191                       |
| **DESIGN.A.METRICS_GPU**    | `functions/evaluator.py:168-184` | `Evaluator.evaluate_chamfer_and_f1()` | Move pred_vertices to GPU, compute chamfer distance   | 10ms/image (GPU accelerated)        | `external/chamfer/` CUDA logs                                  |
| **DESIGN.A.TIMING_NO_SYNC** | `functions/evaluator.py:252-278` | `Evaluator.evaluate()`                | `time.time()` without CUDA sync                       | ±5-10ms error in timing             | `DesignA_Evaluation_Summary.md` lines 290-294                  |
| **DESIGN.A.METRICS**        | `functions/evaluator.py:278-287` | `Evaluator.evaluate()`                | Final metrics logging to console                      | CD=0.000498, F1@τ=64.22%            | `logs/designA/designA_vgg_baseline/*.log` line 1168            |
| **DESIGN.A.EVIDENCE**       | Multiple                         | N/A                                   | Complete evaluation results                           | 129.42 min, 1290.98ms/img           | `DesignA_Evaluation_Summary.md`, `designA_summary_metrics.csv` |

---

## DESIGN.A_GPU: Simple GPU Enablement

| StageID                            | File Path                        | Function/Class               | Description                           | Performance Impact            | Evidence Artifact                            |
| ---------------------------------- | -------------------------------- | ---------------------------- | ------------------------------------- | ----------------------------- | -------------------------------------------- |
| **DESIGN.A_GPU.INIT**              | `functions/evaluator.py:101-102` | `Evaluator.init_fn()`        | **UNCOMMENT** `.cuda()` to enable GPU | +384% speedup over Design A   | `Design A GPU/DesignA_Evaluation_Summary.md` |
| **DESIGN.A_GPU.MODEL_GPU**         | `functions/evaluator.py:102`     | `self.model.cuda()`          | Move model to GPU with DataParallel   | 1090ms → 220ms inference      | Estimated from Design B timings              |
| **DESIGN.A_GPU.BATCH_GPU**         | `functions/evaluator.py:259`     | **UNCOMMENT** `batch.cuda()` | Transfer batch data to GPU            | H2D overhead ~5ms/batch       | Estimated                                    |
| **DESIGN.A_GPU.FWD_GPU**           | `models/p2m.py:50-91`            | `P2MModel.forward()`         | VGG16 + GCN forward pass on GPU       | 220ms/image (65.5% of total)  | Estimated from pipeline analysis             |
| **DESIGN.A_GPU.TIMING_INACCURATE** | `functions/evaluator.py:252-278` | `Evaluator.evaluate()`       | Still no CUDA sync, timing ±5ms error | Inaccurate but ~4.86× faster  | Assumed same structure as Design A           |
| **DESIGN.A_GPU.METRICS**           | Same as Design A                 | Same as Design A             | Same metrics computation              | Same CD/F1 scores as Design A | Expected output                              |

---

## DESIGN.B: Optimized GPU Pipeline

### CAMFM.A5_METHOD: Reproducible Methodology

| StageID                  | File Path                            | Function/Class                | Description                                                 | Performance Impact               | Evidence Artifact                                       |
| ------------------------ | ------------------------------------ | ----------------------------- | ----------------------------------------------------------- | -------------------------------- | ------------------------------------------------------- |
| **CAMFM.A5.CLI_ARGS**    | `entrypoint_designB_eval.py:493-518` | `parse_args()`                | Parse performance flags (warmup, AMP, compile, cuDNN, TF32) | Enables reproducible comparisons | `logs/designB/*/evaluation_summary.json` config section |
| **CAMFM.A5.CONFIG_YAML** | `experiments/designB_baseline.yml`   | YAML config                   | Centralized hyperparameters (batch size, camera intrinsics) | Consistent evaluation setup      | `experiments/designB_baseline.yml` lines 1-85           |
| **CAMFM.A5.CHECKPOINT**  | `functions/base.py:58-61`            | `CheckpointRunner.__init__()` | Load pre-trained checkpoint with validation                 | Ensures model reproducibility    | SHA256 in `DesignA_Evaluation_Summary.md` line 64       |
| **CAMFM.A5.SEED**        | Not implemented                      | N/A                           | **MISSING:** Random seed fixing for perfect reproducibility | Slight run-to-run variance       | Recommendation: Add `torch.manual_seed(42)`             |
| **CAMFM.A5.EVIDENCE**    | `entrypoint_designB_eval.py:440-488` | `save_summary_json()`         | Export complete config + metrics to JSON                    | Full audit trail                 | `logs/designB/*/evaluation_summary.json`                |

### CAMFM.A2a_GPU_RESIDENCY: No CPU Fallbacks

| StageID                       | File Path                                | Function/Class                | Description                                   | Performance Impact                       | Evidence Artifact                                |
| ----------------------------- | ---------------------------------------- | ----------------------------- | --------------------------------------------- | ---------------------------------------- | ------------------------------------------------ |
| **CAMFM.A2a.MODEL_CUDA**      | `entrypoint_designB_eval.py:123`         | `DesignBEvaluator.init_fn()`  | `model.cuda()` with DataParallel              | All model ops on GPU                     | `entrypoint_designB_eval.py` line 123            |
| **CAMFM.A2a.BATCH_CUDA**      | `entrypoint_designB_eval.py:300`         | `DesignBEvaluator.evaluate()` | `batch.cuda()` for all tensors                | All data on GPU during forward           | `entrypoint_designB_eval.py` line 300            |
| **CAMFM.A2a.ELLIPSOID_GPU**   | `utils/mesh.py:15-45`                    | `Ellipsoid.__init__()`        | Ellipsoid tensors on GPU (implicit via model) | Initial mesh on GPU                      | `utils/mesh.py`                                  |
| **CAMFM.A2a.NO_CPU_FALLBACK** | `entrypoint_designB_eval.py:123`         | Explicit GPU-only design      | No CPU fallback paths in forward pass         | Prevents CPU-GPU transfers mid-inference | Code inspection confirms no `.cpu()` in hot path |
| **CAMFM.A2a.CHAMFER_GPU**     | `models/layers/chamfer_wrapper.py:20-25` | `ChamferDist.forward()`       | Chamfer distance CUDA kernel on GPU           | 10ms/image (GPU accelerated)             | `external/chamfer/chamfer.cu`                    |

### CAMFM.A2b_STEADY_STATE: Warmup + Correct Timing

| StageID                   | File Path                            | Function/Class                | Description                                   | Performance Impact                                               | Evidence Artifact                                      |
| ------------------------- | ------------------------------------ | ----------------------------- | --------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------ |
| **CAMFM.A2b.WARMUP**      | `utils/perf.py:89-124`               | `warmup_model()`              | Run 15 dummy forward passes before timing     | Eliminates cold-start artifacts (CUDA init, cuDNN autotune, JIT) | `utils/perf.py` lines 89-124                           |
| **CAMFM.A2b.WARMUP_CALL** | `entrypoint_designB_eval.py:195-205` | `DesignBEvaluator.evaluate()` | Invoke warmup with input_shape=(8,3,224,224)  | Stable timing after warmup                                       | Log line: "Warmup complete: 15 iterations"             |
| **CAMFM.A2b.SYNC_START**  | `entrypoint_designB_eval.py:200`     | `torch.cuda.synchronize()`    | Wait for GPU idle before starting timer       | Accurate timing boundary                                         | `entrypoint_designB_eval.py` line 200                  |
| **CAMFM.A2b.SYNC_END**    | `entrypoint_designB_eval.py:203`     | `torch.cuda.synchronize()`    | Wait for GPU to finish before reading timer   | <1ms timing accuracy                                             | `entrypoint_designB_eval.py` line 203                  |
| **CAMFM.A2b.CUDA_TIMER**  | `utils/perf.py:127-162`              | `CudaTimer` class             | Context manager for CUDA-synchronized timing  | Easy-to-use timing wrapper                                       | `utils/perf.py` lines 127-162                          |
| **CAMFM.A2b.IMPACT**      | N/A                                  | Timing methodology            | Accurate GPU timing vs Design A ±5-10ms error | Confidence in benchmarking results                               | `DesignB_Pipeline_Implementation_Map.md` lines 200-204 |

### CAMFM.A2c_MEM_LAYOUT: Prealloc + Contiguous Tensors

| StageID                     | File Path                          | Function/Class                | Description                                   | Performance Impact                 | Evidence Artifact                         |
| --------------------------- | ---------------------------------- | ----------------------------- | --------------------------------------------- | ---------------------------------- | ----------------------------------------- |
| **CAMFM.A2c.CONTIGUOUS**    | `models/p2m.py:58-78`              | `P2MModel.forward()`          | Use `torch.cat()` with contiguous tensors     | Avoids memory fragmentation        | `models/p2m.py` lines 64, 73              |
| **CAMFM.A2c.PINNED_MEMORY** | `functions/base.py:48`             | `CheckpointRunner.__init__()` | `pin_memory=True` in DataLoader               | Faster H2D transfer (~10% speedup) | `options.py` pin_memory setting           |
| **CAMFM.A2c.BUFFER_REUSE**  | `models/layers/chamfer_wrapper.py` | `ChamferDist` tensors         | Pre-allocated distance buffers in CUDA        | Reduces malloc overhead            | `external/chamfer/chamfer.cu` lines 20-30 |
| **CAMFM.A2c.IMPACT**        | N/A                                | Memory optimizations          | ~10ms savings from contiguous + pinned memory | Estimated ~5% speedup              | Code inspection                           |

### CAMFM.A2d_OPTIONAL_ACCEL: AMP/Compile/cuDNN/TF32

| StageID                        | File Path                          | Function/Class               | Description                                   | Performance Impact                                              | Evidence Artifact                                      |
| ------------------------------ | ---------------------------------- | ---------------------------- | --------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------ |
| **CAMFM.A2d.CUDNN_SETUP**      | `utils/perf.py:26-80`              | `setup_cuda_optimizations()` | `torch.backends.cudnn.benchmark=True`         | cuDNN autotuner selects fastest conv algorithms (~15ms savings) | `utils/perf.py` lines 56-59                            |
| **CAMFM.A2d.CUDNN_CALL**       | `entrypoint_designB_eval.py:56-59` | `setup_cuda_optimizations()` | Enable cuDNN benchmark at startup             | One-time autotune cost (~2s), then consistent speedup           | Log line: "CUDA optimizations: cudnn.benchmark=True"   |
| **CAMFM.A2d.TF32_SETUP**       | `utils/perf.py:64-68`              | `setup_cuda_optimizations()` | `torch.backends.cuda.matmul.allow_tf32=True`  | TF32 tensor cores on Ampere+ GPUs (~10ms savings)               | `utils/perf.py` lines 64-68                            |
| **CAMFM.A2d.TF32_CHECK**       | `utils/perf.py:71-77`              | GPU capability check         | Verify compute capability ≥ 8.0 for TF32      | Graceful degradation on older GPUs                              | Log line: "GPU supports TF32 (compute capability 8.6)" |
| **CAMFM.A2d.AMP_DISABLED**     | `entrypoint_designB_eval.py:141`   | `amp_enabled=False`          | AMP disabled for P2M sparse graph ops         | No speedup (FP16 incompatible)                                  | `entrypoint_designB_eval.py` line 141 comment          |
| **CAMFM.A2d.AMP_CONTEXT**      | `utils/perf.py:165-188`            | `get_autocast_context()`     | Prepare AMP context (nullcontext if disabled) | Framework ready for future AMP support                          | `utils/perf.py` lines 165-188                          |
| **CAMFM.A2d.COMPILE_OPTIONAL** | `entrypoint_designB_eval.py:64-69` | `compile_model_safe()`       | torch.compile with "max-autotune" mode        | ~5ms savings (minimal for P2M dynamic graphs)                   | `utils/perf.py` lines 242-265                          |
| **CAMFM.A2d.COMPILE_SAFE**     | `utils/perf.py:242-265`            | `compile_model_safe()`       | Wrap compile in try-except for compatibility  | Graceful fallback if compile fails                              | `utils/perf.py` lines 242-265                          |
| **CAMFM.A2d.TOTAL_IMPACT**     | N/A                                | cuDNN + TF32 combined        | ~25ms savings (15ms cuDNN + 10ms TF32)        | ~12% speedup over naive GPU                                     | `PIPELINE_OVERVIEW.md` performance table               |

### CAMFM.A3_METRICS: Quality + Performance Export

| StageID                   | File Path                            | Function/Class             | Description                                         | Performance Impact                    | Evidence Artifact                                             |
| ------------------------- | ------------------------------------ | -------------------------- | --------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------- |
| **CAMFM.A3.CHAMFER**      | `entrypoint_designB_eval.py:142-154` | `compute_sample_metrics()` | Chamfer distance via CUDA kernel                    | Quality metric                        | `logs/designB/*/sample_results.csv` chamfer_distance column   |
| **CAMFM.A3.F1_SCORE**     | `entrypoint_designB_eval.py:137-140` | `evaluate_f1()`            | F1-score @ τ and 2τ thresholds                      | Quality metric                        | `logs/designB/*/sample_results.csv` f1_tau/f1_2tau columns    |
| **CAMFM.A3.PER_SAMPLE**   | `entrypoint_designB_eval.py:248-262` | `evaluate_step()`          | Store per-sample metrics in list                    | 43,784 rows exported                  | `logs/designB/*/sample_results.csv`                           |
| **CAMFM.A3.PER_BATCH**    | `entrypoint_designB_eval.py:312-325` | `evaluate()`               | Store per-batch timing + metrics                    | ~5,473 rows exported                  | `logs/designB/*/batch_results.csv`                            |
| **CAMFM.A3.PER_CATEGORY** | `entrypoint_designB_eval.py:354-368` | `evaluate()`               | Aggregate metrics by 13 categories                  | Per-category analysis                 | `logs/designB/*/evaluation_summary.json` per_category section |
| **CAMFM.A3.CSV_EXPORT**   | `entrypoint_designB_eval.py:416-438` | `save_results_to_csv()`    | Write sample_results.csv + batch_results.csv        | Excel/pandas import                   | `logs/designB/*/sample_results.csv`, `batch_results.csv`      |
| **CAMFM.A3.JSON_EXPORT**  | `entrypoint_designB_eval.py:440-488` | `save_summary_json()`      | Write evaluation_summary.json with nested structure | Programmatic parsing                  | `logs/designB/*/evaluation_summary.json`                      |
| **CAMFM.A3.TENSORBOARD**  | `functions/base.py:40`               | Summary writer             | TensorBoard scalar + image logging                  | Visual analysis                       | `summary/designB/*` event files                               |
| **CAMFM.A3.MESH_OBJ**     | `entrypoint_designB_eval.py:172-186` | `save_mesh()`              | Export OBJ files (vertices + faces)                 | 78 mesh files (26 samples × 3 stages) | `outputs/designB_meshes/`                                     |

---

## DESIGN.C: GPU-Native Data Pipeline (Planned)

### DATA.READ_CPU: Minimal CPU File Loading

| StageID            | File Path           | Function/Class                   | Description                            | Performance Impact             | Evidence Artifact                |
| ------------------ | ------------------- | -------------------------------- | -------------------------------------- | ------------------------------ | -------------------------------- |
| **DATA.READ_CPU**  | Not yet implemented | `FaceScapeDataset.__getitem__()` | Read file path only (no decode on CPU) | Eliminates CPU decode overhead | Planned: `datasets/facescape.py` |
| **DATA.FILE_LIST** | Planned             | DALI FileReader                  | Read file paths from directory/txt     | Fast file enumeration          | Planned: DALI pipeline config    |

### DATA.DECODE_GPU_NVJPEG: GPU JPEG Decoding

| StageID             | File Path | Function/Class        | Description                              | Performance Impact                     | Evidence Artifact            |
| ------------------- | --------- | --------------------- | ---------------------------------------- | -------------------------------------- | ---------------------------- |
| **DATA.DECODE_GPU** | Planned   | `fn.decoders.image()` | nvJPEG hardware-accelerated JPEG decode  | 40ms → 15ms decode time (2.67× faster) | Planned: DALI benchmark logs |
| **DATA.MIXED_MODE** | Planned   | DALI mixed device     | CPU read → GPU decode (optimal for JPEG) | Overlap I/O with decode                | Planned: DALI profiling      |

### DATA.RESIZE_GPU: GPU Resize Operations

| StageID             | File Path | Function/Class | Description                 | Performance Impact                 | Evidence Artifact            |
| ------------------- | --------- | -------------- | --------------------------- | ---------------------------------- | ---------------------------- |
| **DATA.RESIZE_GPU** | Planned   | `fn.resize()`  | GPU resize via CUDA kernels | 15ms → 5ms resize time (3× faster) | Planned: DALI benchmark logs |
| **DATA.CROP_GPU**   | Planned   | `fn.crop()`    | Center crop on GPU          | Fused with resize for efficiency   | Planned: DALI profiling      |

### DATA.NORMALIZE_GPU: GPU Normalization

| StageID                | File Path | Function/Class               | Description                            | Performance Impact            | Evidence Artifact            |
| ---------------------- | --------- | ---------------------------- | -------------------------------------- | ----------------------------- | ---------------------------- |
| **DATA.NORMALIZE_GPU** | Planned   | `fn.crop_mirror_normalize()` | ImageNet mean/std normalization on GPU | 5ms CPU → 1ms GPU (5× faster) | Planned: DALI benchmark logs |
| **DATA.HWC_TO_CHW**    | Planned   | DALI transpose               | Convert HWC → CHW on GPU               | Fused with normalize          | Planned: DALI profiling      |

### DATA.DALI_BRIDGE_PYTORCH: Zero-Copy Integration

| StageID                | File Path | Function/Class        | Description                                  | Performance Impact                | Evidence Artifact                    |
| ---------------------- | --------- | --------------------- | -------------------------------------------- | --------------------------------- | ------------------------------------ |
| **DATA.DALI_ITERATOR** | Planned   | `DALIGenericIterator` | DALI → PyTorch zero-copy bridge              | Eliminates H2D copy (5ms savings) | Planned: DALI integration code       |
| **DATA.PREFETCH**      | Planned   | DALI prefetch queue   | Prefetch N batches ahead on GPU              | Overlaps data prep with compute   | Planned: `--prefetch-batches 2` flag |
| **DATA.TOTAL_IMPACT**  | N/A       | GPU data pipeline     | 70ms CPU → 35ms GPU (2× faster data loading) | 15% end-to-end speedup            | Planned: Design C benchmarks         |

---

## Cross-Design Impact Summary

| Optimization                 | Design A | Design A_GPU | Design B | Design C | Cumulative Speedup   |
| ---------------------------- | -------- | ------------ | -------- | -------- | -------------------- |
| **GPU Model Inference**      | ❌ CPU   | ✅ GPU       | ✅ GPU   | ✅ GPU   | 4.86×                |
| **CUDA Synchronized Timing** | ❌       | ❌           | ✅       | ✅       | Accuracy improvement |
| **GPU Warmup**               | ❌       | ❌           | ✅       | ✅       | Stable metrics       |
| **cuDNN Benchmark**          | ❌       | ❌           | ✅       | ✅       | +7% (15ms)           |
| **TF32 Tensor Cores**        | ❌       | ❌           | ✅       | ✅       | +5% (10ms)           |
| **Memory Optimizations**     | ❌       | ❌           | ✅       | ✅       | +5% (10ms)           |
| **GPU Data Pipeline**        | ❌       | ❌           | ❌       | ✅       | +15% (35ms)          |
| **Total Speedup**            | 1.0×     | 4.86×        | 6.98×    | 8.0×     | 8.0×                 |
| **End-to-End Time**          | 1441ms   | 345ms        | 255ms    | 220ms    | 6.55×                |

---

## Evidence Artifact Index

### Logs

- `logs/designA/designA_vgg_baseline/*.log` - Design A console logs
- `logs/designB/*/evaluation_summary.json` - Design B complete metrics
- `logs/designB/*/sample_results.csv` - Per-sample metrics (43,784 rows)
- `logs/designB/*/batch_results.csv` - Per-batch timing (~5,473 rows)

### Documentation

- `DesignA_Evaluation_Summary.md` - Design A full report
- `DesignA_Metrics_Summary.md` - Design A metrics summary
- `DesignB_Pipeline_Implementation_Map.md` - Design B implementation details
- `docs/PIPELINE_OVERVIEW.md` - Architecture diagrams
- `docs/DESIGNS.md` - Design configurations
- `docs/BENCHMARK_PROTOCOL.md` - Timing methodology

### Mesh Outputs

- `outputs/designB_meshes/<category>/<objectID>.<stage>.obj` - 78 OBJ files
- `datasets/examples_for_poster/*.3.obj` - Design A sample meshes (26 final)

### Configuration

- `experiments/designA_vgg_baseline.yml` - Design A config
- `experiments/designB_baseline.yml` - Design B config
- `options.py` - Global configuration defaults

### Code Artifacts

- `utils/perf.py` - Performance optimization utilities (339 lines)
- `entrypoint_designB_eval.py` - Design B main script (795 lines)
- `functions/evaluator.py` - Design A evaluator (337 lines)
- `models/p2m.py` - Pixel2Mesh model (91 lines forward pass)

---

## Validation Checklist

For each design implementation, verify:

- [ ] **Timing Accuracy:** CUDA sync for GPU operations, warmup completed
- [ ] **Reproducibility:** Config + checkpoint + seed documented
- [ ] **Metrics Export:** CSV/JSON files generated with all metrics
- [ ] **Mesh Validation:** OBJ files loadable, correct vertex counts
- [ ] **Performance Evidence:** Speedup measured vs baseline
- [ ] **Code Tags:** `[DESIGN.X][CAMFM.Y]` comments in hot paths

---

## Related Documentation

- [`PIPELINE_OVERVIEW.md`](./PIPELINE_OVERVIEW.md) - Architecture and flow diagrams
- [`DESIGNS.md`](./DESIGNS.md) - Detailed design configurations
- [`BENCHMARK_PROTOCOL.md`](./BENCHMARK_PROTOCOL.md) - Timing rules and best practices

---

**Last Updated:** February 16, 2026  
**Maintainer:** Safa JSK  
**Status:** Designs A, A_GPU, B documented; Design C planned
