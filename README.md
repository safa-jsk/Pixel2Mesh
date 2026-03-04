# Pixel2Mesh

PyTorch implementation of [Pixel2Mesh](https://arxiv.org/abs/1804.01654) — single-image 3D mesh reconstruction — extended with four progressive design variants for GPU performance benchmarking (CAMFM framework).

## Repository Layout

```
Pixel2Mesh/
├── src/pixel2mesh/          ← shared model code (VGG16 + 3-stage GCN)
│   ├── models/              backbone, layers, classifier, p2m
│   ├── losses/              chamfer + mesh regularisation
│   ├── datasets/            ShapeNet / ImageNet loaders
│   ├── engine/              trainer, evaluator, predictor, saver
│   ├── utils/               mesh helpers, timers, vis
│   └── tools/               data migrations, demo selector
├── configs/
│   ├── defaults/            designA_vgg.yml, designB.yml, default.yml
│   └── experiments/         baseline/ (35 variants), backbone/
├── DesignA_CPU/             CPU-baseline design (scripts + Docker)
├── DesignA_GPU/             Simple GPU migration
├── DesignB/                 Optimized GPU + CAMFM methodology
├── DesignC/                 FaceScape adapter (skeleton)
├── external/                chamfer CUDA ext + neural_renderer submodule
├── datasets/data/           ShapeNet data (gitignored)
├── artifacts/               logs, meshes, CSVs (gitignored)
├── docs/                    methodology, reports, Docker guides
├── tests/                   import + chamfer + perf tests
└── scripts/                 build, env-check, smoke-test, clean
```

## Design Variants

| Design | Description | Performance | Entry-point |
|--------|-------------|-------------|-------------|
| **A (CPU)** | CPU baseline | ~1291 ms/img | `DesignA_CPU/scripts/eval.py` |
| **A (GPU)** | Simple GPU migration | ~265 ms/img | `DesignA_GPU/scripts/eval.py` |
| **B** | Optimized GPU (CAMFM) | ~185 ms/img | `DesignB/scripts/eval_full.py` |
| **C** | FaceScape adapter | planned | `DesignC/scripts/eval_facescape.py` |

## Quick Start

### Prerequisites

- Python 3.7+, PyTorch ≥ 1.1, CUDA ≥ 9.0
- `pip install -r requirements.txt`
- `git submodule update --init`
- `bash scripts/build_chamfer.sh`

### Evaluate (Design A — CPU baseline)

```bash
python DesignA_CPU/scripts/eval.py \
  --options configs/experiments/baseline/lr_1e-4.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar
```

### Evaluate (Design B — optimised GPU)

```bash
python DesignB/scripts/eval_full.py \
  --options configs/defaults/designB.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
  --name designB_optimized
```

### Train

```bash
python DesignA_CPU/scripts/train.py \
  --options configs/experiments/baseline/lr_1e-4.yml \
  --name my_training_run
```

### Predict (mesh generation)

```bash
python DesignA_CPU/scripts/predict.py \
  --options configs/experiments/baseline/lr_1e-4.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
  --folder /path/to/images
```

## Dataset Setup

Download [ShapeNet subset](https://drive.google.com/drive/folders/131dH36qXCabym1JjSmEpSQZg4dmZVQid) and [meta files](https://drive.google.com/file/d/16d9druvCpsjKWsxHmsTD5HSOWiCWtDzo/view?usp=sharing), then arrange:

```
datasets/data/
├── ellipsoid/       face1-3.obj + info_ellipsoid.dat
├── pretrained/      .pth checkpoint files
└── shapenet/
    ├── data_tf/     137×137 RGBA images (default)
    ├── data/        224×224 RGB images (optional, ~300 GB)
    └── meta/
```

## Results

| Checkpoint | Protocol | CD | F1^τ | F1^2τ |
|-----------|----------|-----|------|-------|
| Official TF (migrated) | Weighted-mean | 0.439 | 66.56 | 80.17 |
| Retrained VGG | Weighted-mean | 0.451 | 65.67 | 79.51 |
| ResNet | Weighted-mean | **0.411** | 66.13 | 80.13 |

## Documentation

See [docs/index.md](docs/index.md) for the full documentation index, including:

- [Pipeline Overview](docs/methodology/pipeline_overview.md) — architecture diagrams
- [Benchmark Protocol](docs/methodology/benchmark_protocol.md) — CAMFM timing methodology
- [Traceability Matrix](docs/methodology/traceability_matrix.md) — code → methodology mapping
- [Design Reports](docs/reports/) — per-design evaluation summaries

## Validation

```bash
bash scripts/env_check.sh        # verify dependencies
bash scripts/smoke_test.sh       # import + forward-pass test
python tests/test_imports.py     # package import check
```

## Acknowledgements

Based on the official [Pixel2Mesh](https://github.com/nywang16/Pixel2Mesh) and a [PyTorch reimplementation](https://github.com/Tong-ZHAO/Pixel2Mesh-Pytorch). Core codework by [Yuge Zhang](https://github.com/ultmaster).

> Safa JSK. "GPU Optimization Strategies for 3D Mesh Reconstruction: A Comprehensive Analysis of Pixel2Mesh Implementations." MS Thesis, 2026.
