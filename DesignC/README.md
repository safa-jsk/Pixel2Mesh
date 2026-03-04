# Design C — FaceScape Adapter (Skeleton)

**Status:** Skeleton — data adapter and scripts ready, awaiting FaceScape dataset access.

## Overview

Design C extends the Pixel2Mesh pipeline to the [FaceScape](https://facescape.nju.edu.cn/) dataset for
face-specific 3D mesh reconstruction. It reuses the shared model code in `src/pixel2mesh/` and adds
a thin dataset adapter for FaceScape image/mesh pairs.

## Prerequisites

1. **FaceScape dataset** — download from https://facescape.nju.edu.cn/ (requires academic registration)
2. **Splits CSV** — a CSV with columns `image_path,mesh_path,split` indicating train/val/test splits
3. Place data under `datasets/data/facescape/`

## Quick Start

```bash
# Evaluate on FaceScape test split
python DesignC/scripts/eval_facescape.py \
  --options configs/defaults/designA_vgg.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
  --facescape_root datasets/data/facescape \
  --splits_csv datasets/data/facescape/splits.csv \
  --name designC_facescape

# Generate meshes from FaceScape images
python DesignC/scripts/predict_facescape.py \
  --options configs/defaults/designA_vgg.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
  --facescape_root datasets/data/facescape \
  --splits_csv datasets/data/facescape/splits.csv
```

## Directory Layout

```
DesignC/
├── README.md                    ← you are here
└── scripts/
    ├── eval_facescape.py        ← evaluation on FaceScape
    ├── predict_facescape.py     ← mesh generation
    └── facescape_adapter.py     ← dataset adapter (torch Dataset)
```

## FaceScape Adapter

The adapter (`facescape_adapter.py`) provides a PyTorch `Dataset` that:

- Reads image/mesh pairs from the FaceScape directory structure
- Applies the same preprocessing as ShapeNet (resize to 224×224, normalise)
- Returns tensors compatible with the Pixel2Mesh forward pass
- Raises a clear `FileNotFoundError` at import time if data is missing

## Extending This Design

To add NVIDIA DALI GPU-accelerated data loading:

1. Install `nvidia-dali-cuda110` (or matching CUDA version)
2. Replace the PyTorch `DataLoader` in `eval_facescape.py` with a DALI pipeline
3. Benchmark using the same CAMFM timing methodology as Design B

## Documentation

- [Design C Guideline](../docs/design_c/guideline.md)
- [Designs Overview](../docs/methodology/designs.md)

See [main documentation](../docs/index.md) for the full index.
