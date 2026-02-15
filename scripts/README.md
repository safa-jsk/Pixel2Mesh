# Scripts Directory

Organized collection of all executable scripts for the Pixel2Mesh project.

---

## 📁 Structure

```
scripts/
├── docker/          - Docker-related scripts
├── evaluation/      - Evaluation and inference scripts
└── setup/           - Setup and testing scripts
```

---

## 🐳 docker/

Docker environment management scripts.

| Script                                                    | Purpose                       |
| --------------------------------------------------------- | ----------------------------- |
| [docker-build.sh](./docker/docker-build.sh)               | Build Docker image            |
| [docker-compose.yml](./docker/docker-compose.yml)         | Docker Compose configuration  |
| [docker-status.sh](./docker/docker-status.sh)             | Check Docker container status |
| [setup-nvidia-docker.sh](./docker/setup-nvidia-docker.sh) | Setup NVIDIA Docker runtime   |

**Quick Start:**

```bash
# Build image
./scripts/docker/docker-build.sh

# Check status
./scripts/docker/docker-status.sh

# Use with docker-compose
cd scripts/docker && docker-compose up
```

---

## 📊 evaluation/

Scripts for running evaluations and generating predictions.

| Script                                                                | Design  | Purpose                            |
| --------------------------------------------------------------------- | ------- | ---------------------------------- |
| [run_designA_eval.sh](./evaluation/run_designA_eval.sh)               | A (CPU) | Evaluate Design A baseline         |
| [run_designA_predict.sh](./evaluation/run_designA_predict.sh)         | A (CPU) | Generate predictions with Design A |
| [run_designB_eval.sh](./evaluation/run_designB_eval.sh)               | B (GPU) | Evaluate Design B optimized        |
| [run_designB_eval_docker.sh](./evaluation/run_designB_eval_docker.sh) | B (GPU) | Evaluate Design B in Docker        |
| [generate_sample_meshes.sh](./evaluation/generate_sample_meshes.sh)   | -       | Generate mesh samples              |

**Quick Start:**

```bash
# Design A evaluation
./scripts/evaluation/run_designA_eval.sh

# Design B evaluation (GPU)
./scripts/evaluation/run_designB_eval.sh

# Design B in Docker
./scripts/evaluation/run_designB_eval_docker.sh
```

---

## 🔧 setup/

Setup and testing utilities.

| Script                                           | Purpose                       |
| ------------------------------------------------ | ----------------------------- |
| [test.py](./setup/test.py)                       | General testing script        |
| [test_perf_utils.py](./setup/test_perf_utils.py) | Performance utilities testing |

**Usage:**

```bash
python scripts/setup/test_perf_utils.py
```

---

## 🎯 Common Workflows

### Run Design A Evaluation

```bash
./scripts/evaluation/run_designA_eval.sh
```

### Run Design B Evaluation with Docker

```bash
# Build Docker image first
./scripts/docker/docker-build.sh

# Run evaluation
./scripts/evaluation/run_designB_eval_docker.sh
```

### Setup Development Environment

```bash
# Setup NVIDIA Docker
./scripts/docker/setup-nvidia-docker.sh

# Test performance utilities
python scripts/setup/test_perf_utils.py
```

---

## 📝 Notes

- All scripts assume they are run from the **project root directory**
- Docker scripts require Docker and NVIDIA Container Toolkit
- Evaluation scripts require datasets to be properly configured
- See main [README.md](../README.md) for complete setup instructions

---

**Related Documentation:**

- [DOCKER_SETUP.md](../DOCKER_SETUP.md) - Detailed Docker setup
- [docs/DESIGNS.md](../docs/DESIGNS.md) - Design configurations
- [docs/BENCHMARK_PROTOCOL.md](../docs/BENCHMARK_PROTOCOL.md) - Evaluation protocols
