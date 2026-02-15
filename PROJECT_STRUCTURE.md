# Pixel2Mesh Project Structure

**Last Updated:** February 16, 2026  
**Version:** Design_A branch - Organized structure

This document provides a complete overview of the project organization.

---

## 📁 Project Tree

```
Pixel2Mesh/
│
├── 📄 Core Python Files
│   ├── config.py                    - Global configuration
│   ├── logger.py                    - Logging utilities
│   ├── options.py                   - Command-line argument parsing
│   ├── entrypoint_eval.py          - Design A evaluation entry point
│   ├── entrypoint_designB_eval.py  - Design B evaluation entry point
│   ├── entrypoint_train.py         - Training entry point
│   └── entrypoint_predict.py       - Prediction entry point
│
├── 📚 Documentation (Root)
│   ├── README.md                           - Main project documentation
│   ├── DOCUMENTATION_ORGANIZATION.md       - Documentation structure guide
│   ├── PROJECT_STRUCTURE.md               - This file
│   ├── DOCKER_SETUP.md                    - Docker setup instructions
│   └── DOCKER_QUICK_REFERENCE.md          - Docker quick reference
│
├── 🏗️ Source Code
│   ├── models/                     - Neural network models
│   │   ├── p2m.py                 - Pixel2Mesh model
│   │   ├── classifier.py          - Classification model
│   │   ├── backbones/             - VGG16, ResNet50 backbones
│   │   ├── layers/                - Custom layers (GCN, pooling)
│   │   └── losses/                - Loss functions
│   │
│   ├── functions/                  - Core functionality
│   │   ├── base.py                - Base class for train/eval
│   │   ├── trainer.py             - Training logic
│   │   ├── evaluator.py           - Evaluation logic (Design A)
│   │   ├── predictor.py           - Prediction logic
│   │   └── saver.py               - Checkpoint saving
│   │
│   ├── utils/                      - Utility modules
│   │   ├── mesh.py                - Mesh operations
│   │   ├── tensor.py              - Tensor utilities
│   │   ├── average_meter.py       - Metric tracking
│   │   ├── perf.py                - Performance utilities (warmup, timing)
│   │   └── vis/                   - Visualization tools
│   │
│   └── datasets/                   - Dataset loading
│       ├── shapenet.py            - ShapeNet dataset
│       ├── imagenet.py            - ImageNet dataset
│       ├── base_dataset.py        - Base dataset class
│       └── data/                  - Data files
│           ├── ellipsoid/         - Initial ellipsoid mesh
│           ├── pretrained/        - Pretrained checkpoints
│           └── shapenet/          - ShapeNet dataset
│
├── 🧪 Experiments
│   ├── experiments/                - Configuration files
│   │   ├── designA_vgg_baseline.yml
│   │   ├── designB_baseline.yml
│   │   ├── baseline/              - Baseline configs
│   │   ├── backbone/              - Backbone configs
│   │   └── default/               - Default configs
│   │
│   └── external/                   - External dependencies
│       ├── chamfer/               - Chamfer distance (compiled)
│       └── neural_renderer/       - Neural renderer (submodule)
│
├── 📜 Scripts (Organized)
│   └── scripts/
│       ├── README.md              - Scripts documentation
│       ├── docker/                - Docker-related scripts
│       │   ├── docker-build.sh
│       │   ├── docker-compose.yml
│       │   ├── docker-status.sh
│       │   └── setup-nvidia-docker.sh
│       │
│       ├── evaluation/            - Evaluation scripts
│       │   ├── run_designA_eval.sh
│       │   ├── run_designA_predict.sh
│       │   ├── run_designB_eval.sh
│       │   ├── run_designB_eval_docker.sh
│       │   └── generate_sample_meshes.sh
│       │
│       └── setup/                 - Setup and testing
│           ├── test.py
│           └── test_perf_utils.py
│
├── 📊 Evaluation Results (Organized)
│   └── evaluation_results/
│       ├── README.md              - Results documentation
│       ├── Design A CPU/          - CPU baseline results
│       │   ├── README.md
│       │   ├── Design_A_Pixel2Mesh_Guideline.md
│       │   ├── P2M_DesignA_Evaluation_Summary.md
│       │   ├── P2M_DesignA_Metrics_Summary.md
│       │   ├── P2M_DesignA_Mesh_Generation_Summary.md
│       │   ├── designA_batch_results.csv
│       │   └── designA_summary_metrics.csv
│       │
│       ├── Design A GPU/          - Simple GPU results
│       │   ├── README.md
│       │   ├── DesignA_Evaluation_Summary.md
│       │   ├── DesignA_Metrics_Summary.md
│       │   ├── DesignA_Mesh_Generation_Summary.md
│       │   ├── DesignA_Pipeline_Documentation.md
│       │   └── designA_batch_results.csv
│       │
│       ├── Design B/              - Optimized GPU results
│       │   ├── README.md
│       │   ├── Design_B_Pixel2Mesh_Guideline.md
│       │   ├── DesignB_Evaluation_Summary.md
│       │   ├── DesignB_Metrics_Summary.md
│       │   ├── DesignB_Mesh_Generation_Summary.md
│       │   ├── DesignB_Pipeline_Methodology.md
│       │   ├── DesignB_Pipeline_Implementation_Map.md
│       │   ├── DesignA_vs_DesignB_Comparison.md
│       │   └── batch_results.csv
│       │
│       └── .archive/              - Archived old results
│
├── 📚 Thesis Documentation
│   └── docs/
│       ├── DOCUMENTATION_INDEX.md         - Master documentation index
│       ├── PIPELINE_OVERVIEW.md          - Architecture & Mermaid diagrams
│       ├── DESIGNS.md                    - All 4 design configurations
│       ├── TRACEABILITY_MATRIX.md        - Code-to-methodology mapping
│       ├── BENCHMARK_PROTOCOL.md         - Timing & validation protocols
│       └── Design_C_Pixel2Mesh_Guideline.md
│
├── 🗂️ Runtime Outputs
│   ├── logs/                      - Evaluation logs
│   │   ├── designA/              - Design A logs
│   │   └── designB/              - Design B logs
│   │
│   ├── outputs/                   - Generated outputs
│   │   └── designB_meshes/       - Design B mesh files
│   │
│   ├── checkpoints/               - Model checkpoints
│   │   └── designA/              - Design A checkpoints
│   │
│   └── summary/                   - Summary outputs
│
├── 🐳 Docker
│   ├── Dockerfile                 - Docker image definition
│   └── .dockerignore             - Docker ignore patterns
│
└── 🔧 Configuration
    ├── .gitignore                 - Git ignore patterns
    ├── .gitmodules                - Git submodules config
    └── .vscode/                   - VS Code settings
```

---

## 📂 Directory Purposes

### Core Directories

| Directory        | Purpose                      | Key Files                            |
| ---------------- | ---------------------------- | ------------------------------------ |
| **models/**      | Neural network architectures | p2m.py, backbones/, layers/, losses/ |
| **functions/**   | Training/evaluation logic    | trainer.py, evaluator.py, base.py    |
| **utils/**       | Helper utilities             | perf.py, mesh.py, tensor.py          |
| **datasets/**    | Data loading                 | shapenet.py, data/                   |
| **experiments/** | Configuration files          | \*.yml config files                  |
| **external/**    | External dependencies        | chamfer/, neural_renderer/           |

### Organized Directories (NEW)

| Directory               | Purpose                | Contents                             |
| ----------------------- | ---------------------- | ------------------------------------ |
| **scripts/**            | All executable scripts | docker/, evaluation/, setup/         |
| **evaluation_results/** | Evaluation outputs     | Design A/B/C results with README     |
| **docs/**               | Thesis documentation   | Methodology, traceability, protocols |

### Runtime Directories

| Directory        | Purpose           | Generated By                                   |
| ---------------- | ----------------- | ---------------------------------------------- |
| **logs/**        | Evaluation logs   | entrypoint_eval.py, entrypoint_designB_eval.py |
| **outputs/**     | Generated meshes  | Evaluation scripts                             |
| **checkpoints/** | Model checkpoints | Training/evaluation                            |
| **summary/**     | Summary outputs   | Various scripts                                |

---

## 🎯 Common Tasks & File Locations

### Running Evaluations

**Design A (CPU):**

```bash
# Script location
./scripts/evaluation/run_designA_eval.sh

# Entry point
python entrypoint_eval.py --options experiments/designA_vgg_baseline.yml

# Results saved to
logs/designA/
evaluation_results/Design A CPU/  (archived results)
```

**Design B (GPU):**

```bash
# Script location
./scripts/evaluation/run_designB_eval.sh

# Entry point
python entrypoint_designB_eval.py --options experiments/designB_baseline.yml

# Results saved to
logs/designB/
evaluation_results/Design B/  (archived results)
```

### Docker Usage

**Setup:**

```bash
# Docker files
scripts/docker/docker-build.sh        # Build image
scripts/docker/docker-compose.yml     # Compose config
scripts/docker/setup-nvidia-docker.sh # NVIDIA setup

# Documentation
DOCKER_SETUP.md                       # Full setup guide
DOCKER_QUICK_REFERENCE.md            # Quick reference
```

### Documentation

**Thesis Documentation:**

```bash
docs/PIPELINE_OVERVIEW.md         # Architecture diagrams
docs/DESIGNS.md                   # All design configs
docs/TRACEABILITY_MATRIX.md       # Code mapping
docs/BENCHMARK_PROTOCOL.md        # Timing protocols
```

**Result Analysis:**

```bash
evaluation_results/Design A CPU/  # CPU baseline
evaluation_results/Design A GPU/  # Simple GPU
evaluation_results/Design B/      # Optimized GPU
```

---

## 🔍 Finding Specific Components

### Model Architecture

- **Main model:** [models/p2m.py](models/p2m.py)
- **Backbones:** [models/backbones/](models/backbones/) (VGG16, ResNet)
- **GCN layers:** [models/layers/gcn.py](models/layers/gcn.py)
- **Losses:** [models/losses/](models/losses/)

### Evaluation Logic

- **Design A evaluator:** [functions/evaluator.py](functions/evaluator.py)
- **Design B evaluator:** [entrypoint_designB_eval.py](entrypoint_designB_eval.py) (inline)
- **Performance utilities:** [utils/perf.py](utils/perf.py)
- **Mesh utilities:** [utils/mesh.py](utils/mesh.py)

### Configuration

- **Design A config:** [experiments/designA_vgg_baseline.yml](experiments/designA_vgg_baseline.yml)
- **Design B config:** [experiments/designB_baseline.yml](experiments/designB_baseline.yml)
- **Global config:** [config.py](config.py)

### Datasets

- **ShapeNet loader:** [datasets/shapenet.py](datasets/shapenet.py)
- **Data files:** [datasets/data/shapenet/](datasets/data/shapenet/)
- **Ellipsoid mesh:** [datasets/data/ellipsoid/](datasets/data/ellipsoid/)

---

## 📋 Entry Points

| Script                                                   | Purpose    | Design         |
| -------------------------------------------------------- | ---------- | -------------- |
| [entrypoint_eval.py](entrypoint_eval.py)                 | Evaluation | Design A (CPU) |
| [entrypoint_designB_eval.py](entrypoint_designB_eval.py) | Evaluation | Design B (GPU) |
| [entrypoint_train.py](entrypoint_train.py)               | Training   | All            |
| [entrypoint_predict.py](entrypoint_predict.py)           | Prediction | All            |

---

## 🗂️ Code Organization Principles

1. **Core code in root:** Entry points, config, logger
2. **Source code in subdirectories:** models/, functions/, utils/, datasets/
3. **Scripts organized:** All executable scripts in scripts/
4. **Results organized:** All evaluation results in evaluation_results/
5. **Documentation organized:** Thesis docs in docs/, design docs in evaluation_results/
6. **Runtime outputs separate:** logs/, outputs/, checkpoints/

---

## 🔄 Migration Notes

### Recent Changes (Feb 16, 2026)

**Moved:**

- `docker-*.sh, docker-compose.yml, setup-nvidia-docker.sh` → `scripts/docker/`
- `run_*.sh, generate_sample_meshes.sh` → `scripts/evaluation/`
- `test.py, test_perf_utils.py` → `scripts/setup/`
- `Design A CPU/, Design A GPU/, Design B/` → `evaluation_results/`

**Created:**

- `scripts/README.md` - Scripts documentation
- `evaluation_results/README.md` - Results documentation
- `PROJECT_STRUCTURE.md` - This file (project structure)

**Updated:**

- `README.md` - Updated with new structure
- `DOCUMENTATION_ORGANIZATION.md` - Reflects new paths

---

## 📊 Size Statistics

| Category                 | Count | Total Size   |
| ------------------------ | ----- | ------------ |
| Python files (core)      | 6     | ~1,500 lines |
| Python files (models)    | 15+   | ~3,000 lines |
| Python files (functions) | 5     | ~2,000 lines |
| Python files (utils)     | 10+   | ~1,500 lines |
| Configuration files      | 30+   | ~500 lines   |
| Documentation (markdown) | 35+   | ~8,000 lines |
| Scripts (bash)           | 10+   | ~500 lines   |

---

## 🎯 Quick Reference

### Most Important Files

**For Development:**

1. [models/p2m.py](models/p2m.py) - Main model
2. [entrypoint_designB_eval.py](entrypoint_designB_eval.py) - Design B evaluation
3. [utils/perf.py](utils/perf.py) - Performance utilities

**For Usage:**

1. [README.md](README.md) - Project overview
2. [scripts/evaluation/run_designB_eval.sh](scripts/evaluation/run_designB_eval.sh) - Run evaluation
3. [DOCKER_SETUP.md](DOCKER_SETUP.md) - Docker setup

**For Thesis:**

1. [docs/PIPELINE_OVERVIEW.md](docs/PIPELINE_OVERVIEW.md) - Architecture
2. [docs/TRACEABILITY_MATRIX.md](docs/TRACEABILITY_MATRIX.md) - Code mapping
3. [evaluation_results/Design B/DesignB_Evaluation_Summary.md](evaluation_results/Design%20B/DesignB_Evaluation_Summary.md) - Results

---

**Related Documents:**

- [DOCUMENTATION_ORGANIZATION.md](DOCUMENTATION_ORGANIZATION.md) - Documentation structure
- [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md) - Comprehensive index
- [README.md](README.md) - Main project documentation
