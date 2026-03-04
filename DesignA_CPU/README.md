# Design A — CPU Baseline

**Performance:** ~1291 ms/image  
**Configuration:** Model inference on CPU, metrics computed on GPU (hybrid)

## Quick Start

```bash
# From the repo root
python DesignA_CPU/scripts/eval.py \
  --options configs/experiments/baseline/lr_1e-4.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar

# Or use the shell wrapper
bash DesignA_CPU/scripts/eval.sh
```

## Directory Layout

```
DesignA_CPU/
├── README.md          ← you are here
├── docker/
│   ├── Dockerfile
│   ├── docker-build.sh
│   └── docker-status.sh
└── scripts/
    ├── eval.py        ← evaluation entry-point
    ├── eval.sh        ← shell helper
    ├── train.py       ← training entry-point
    └── predict.py     ← mesh generation
```

## Docker

```bash
cd DesignA_CPU/docker
bash docker-build.sh
bash docker-status.sh
```

## Evaluation Reports

- [Evaluation Summary](../docs/reports/designA_CPU/evaluation_summary.md)
- [Metrics Summary](../docs/reports/designA_CPU/metrics_summary.md)
- [Mesh Generation Summary](../docs/reports/designA_CPU/mesh_generation_summary.md)
- [Setup Guideline](../docs/reports/designA_CPU/guideline.md)

## Artifacts

Results are written to `artifacts/` (gitignored):

- `artifacts/logs/designA/` — training/evaluation logs
- `artifacts/outputs/` — predicted meshes
- `artifacts/evaluation_results/` — CSV metrics

## Key Characteristics

- Baseline for cross-design comparison
- Simple implementation (no GPU optimizations)
- Slowest of all designs due to CPU bottleneck

See [main documentation](../docs/index.md) for methodology mapping.
