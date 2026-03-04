# Design B — Optimized GPU with CAMFM

**Performance:** ~185 ms/image (cuDNN + TF32)  
**Speedup:** 6.97× vs Design A CPU, 1.43× vs Design A GPU  
**Configuration:** Full CAMFM methodology implementation

## Quick Start

```bash
# From the repo root
python DesignB/scripts/eval_full.py \
  --options configs/defaults/designB.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
  --name designB_optimized

# Quick sample evaluation (poster images only)
python DesignB/scripts/eval_samples.py \
  --options configs/defaults/designB.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar

# Or use the shell wrapper
bash DesignB/scripts/run_eval.sh
```

## Directory Layout

```
DesignB/
├── README.md          ← you are here
└── scripts/
    ├── eval_full.py           ← full dataset evaluation
    ├── eval_samples.py        ← poster sample evaluation
    ├── predict.py             ← mesh generation
    ├── run_eval.sh            ← shell helper
    ├── run_eval_docker.sh     ← Docker evaluation
    └── generate_sample_meshes.sh
```

## CAMFM Methodology Stages

| Stage | Description | Status |
|-------|-------------|--------|
| A5_METHOD | Reproducible config with explicit flags | ✅ |
| A2a_GPU_RESIDENCY | Model fully on GPU, no CPU fallbacks | ✅ |
| A2b_STEADY_STATE | 15-iter warmup + CUDA-synced timing | ✅ |
| A2c_MEM_LAYOUT | Contiguous tensors, pinned memory, FP32 | ✅ |
| A2d_OPTIONAL_ACCEL | cuDNN autotune, TF32 tensor cores | ✅ |
| A3_METRICS | Quality + perf metrics + mesh generation | ✅ |

## Evaluation Reports

- [Evaluation Summary](../docs/reports/designB/evaluation_summary.md)
- [Mesh Generation Summary](../docs/reports/designB/mesh_generation_summary.md)
- [Pipeline Methodology](../docs/reports/designB/pipeline_methodology.md)
- [Pipeline Implementation Map](../docs/reports/designB/pipeline_implementation_map.md)
- [Design A vs B Comparison](../docs/reports/designB/designA_vs_designB_comparison.md)
- [Setup Guideline](../docs/reports/designB/guideline.md)

## Artifacts

Results are written to `artifacts/` (gitignored):

- `artifacts/logs/designB/` — evaluation logs
- `artifacts/outputs/` — predicted meshes + JSON summaries
- `artifacts/evaluation_results/` — CSV metrics

## Key Characteristics

- Best performance among all designs
- Accurate GPU benchmarking (CUDA event synchronization)
- Complete CAMFM methodology traceability
- Full evidence artifacts (logs/CSV/JSON/meshes)

See [main documentation](../docs/index.md) for complete methodology mapping.
