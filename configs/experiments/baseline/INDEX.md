# Baseline Experiment Configurations

This directory contains YAML experiment configs that override the defaults in `configs/defaults/`.

Each file uses `based_on` to inherit from another config in this directory (usually `default.yml` → `lr_1e-4.yml` → specialised variant).

## Key Configs

| File | Purpose |
|------|---------|
| `default.yml` | Base config (inherits `../../defaults/default.yml`) |
| `lr_1e-4.yml` | Standard learning rate — used for **Design A** evaluation |
| `lr_1e-4_resnet_dataset_all.yml` | ResNet backbone variant |
| `chamfer_only.yml` | Chamfer-only loss (no Laplacian/edge/normal) |
| `large_laplace.yml` | High Laplacian weight |
| `normal_free.yml` | No normal loss |

## Adding a New Experiment

1. Create `my_experiment.yml` in this directory.
2. Set `based_on: <parent>.yml` (relative path within this folder).
3. Override only the hyperparameters you want to change.
4. Run: `python DesignA_CPU/scripts/eval.py --options configs/experiments/baseline/my_experiment.yml`
