# Design A — GPU (Simple GPU Migration)

**Performance:** ~265 ms/image  
**Speedup:** 4.86× vs Design A CPU  
**Configuration:** Model fully on GPU, simple migration

## Quick Start

```bash
# From the repo root
python DesignA_GPU/scripts/eval.py \
  --options configs/experiments/baseline/lr_1e-4.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar
```

## Directory Layout

```
DesignA_GPU/
├── README.md          ← you are here
├── docker/
│   ├── docker-compose.yml
│   └── setup-nvidia-docker.sh
└── scripts/
    ├── eval.py        ← evaluation entry-point
    ├── train.py       ← training entry-point
    └── predict.py     ← mesh generation
```

## Evaluation Reports

- [Evaluation Summary](../docs/reports/designA_GPU/evaluation_summary.md)
- [Metrics Summary](../docs/reports/designA_GPU/metrics_summary.md)
- [Mesh Generation Summary](../docs/reports/designA_GPU/mesh_generation_summary.md)
- [Pipeline Documentation](../docs/reports/designA_GPU/pipeline_documentation.md)

## Artifacts

Results are written to `artifacts/` (gitignored):

- `artifacts/logs/designA/` — training/evaluation logs
- `artifacts/outputs/` — predicted meshes

## Key Characteristics

- Significant speedup (4.86×) with minimal code changes
- No advanced GPU optimizations (no warmup, no CUDA-synced timing)
- Stepping stone from CPU baseline to optimised Design B

See [main documentation](../docs/index.md) for methodology mapping.
