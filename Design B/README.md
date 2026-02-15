# Design B (Optimized GPU with CAMFM)

**Performance:** 185 ms/image (with cuDNN + TF32)  
**Speedup:** 6.97× vs Design A, 1.43× vs Design A_GPU  
**Configuration:** Full CAMFM methodology implementation

## Documentation

- [**Design_B_Pixel2Mesh_Guideline.md**](./Design_B_Pixel2Mesh_Guideline.md) - Setup and execution guide
- [**DesignB_Evaluation_Summary.md**](./DesignB_Evaluation_Summary.md) - Complete evaluation results
- [**DesignB_Mesh_Generation_Summary.md**](./DesignB_Mesh_Generation_Summary.md) - Mesh output analysis
- [**DesignB_Pipeline_Methodology.md**](./DesignB_Pipeline_Methodology.md) - CAMFM implementation details
- [**DesignB_Pipeline_Implementation_Map.md**](./DesignB_Pipeline_Implementation_Map.md) - Code mapping
- [**DesignA_vs_DesignB_Comparison.md**](./DesignA_vs_DesignB_Comparison.md) - Performance comparison

## Data Files

- [**batch_results.csv**](./batch_results.csv) - Per-batch metrics with timing

## Execution

```bash
python entrypoint_designB_eval.py \
  --options experiments/designB_baseline.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
  --name designB_optimized \
  --warmup-iters 15 \
  --cudnn-benchmark --tf32
```

## CAMFM Methodology Stages

- ✅ **A5_METHOD:** Reproducible configuration with explicit flags
- ✅ **A2a_GPU_RESIDENCY:** Model fully on GPU, no CPU fallbacks
- ✅ **A2b_STEADY_STATE:** 15-iteration warmup + CUDA-synchronized timing
- ✅ **A2c_MEM_LAYOUT:** Contiguous tensors, pinned memory, FP32 precision
- ✅ **A2d_OPTIONAL_ACCEL:** cuDNN autotune, TF32 tensor cores
- ✅ **A3_METRICS:** Quality metrics + performance export + mesh generation

## Key Characteristics

- ✅ Best performance among all designs
- ✅ Accurate GPU benchmarking (CUDA sync)
- ✅ Complete methodology traceability
- ✅ Full evidence artifacts (logs/CSV/JSON/meshes)

See [main documentation](../docs/) for complete methodology mapping.
