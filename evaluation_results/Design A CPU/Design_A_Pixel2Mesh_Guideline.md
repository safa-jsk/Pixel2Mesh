# Pixel2Mesh — Design A Guideline (Baseline: Original Method + ShapeNet)

**Goal of Design A:**  
Reproduce **Pixel2Mesh** on the **standard ShapeNet subset (`data_tf`)** with **no algorithmic changes**, then produce:
- **quantitative evaluation** (CD / F1 metrics as implemented by the repo),
- **qualitative meshes** (OBJ outputs + renders) for your poster/report,
- a **reproducible runbook** (commands + configs + environment details).

This guideline assumes: **Ubuntu 22.04 + NVIDIA GPU**.

---

## 0) What “counts” as Design A (for your thesis)

✅ Allowed in Design A:
- Using a **pretrained checkpoint** (VGG migrated / official TF-converted `.pth`).
- Using the **repo’s standard dataset** (`data_tf`) and meta files.
- Minor **compatibility-only** fixes needed to run on your machine (document them).

❌ Avoid in Design A:
- Switching the network backbone to ResNet (that becomes a new design variant).
- CUDA/AMP/`torch.compile` performance tuning (save for Design B).
- Domain shift (FaceScape) (save for Design C).

> **Best baseline checkpoint for Design A:** use the **VGG migrated checkpoint** converted from the official TensorFlow model.  
> (ResNet is an altered backbone; it can be used later as an alternative/ablation.)

---

## 1) Choose your baseline execution strategy (pick one)

Pixel2Mesh PyTorch repo was originally validated on an older stack (Ubuntu 16/18, Python 3.7, PyTorch 1.1, CUDA 9/10) and requires compiling CUDA extensions (`chamfer`, `neural_renderer`).  
On Ubuntu 22.04, you have two workable strategies:

### Option A (Recommended for “pure” baseline): Run in Docker with a known-good old stack
- Pros: Closest to original environment, fewer “porting changes”, strong reproducibility.
- Cons: You must set up NVIDIA Container Toolkit.

### Option B (Native Ubuntu 22.04): Create a conda env and patch only what’s necessary
- Pros: Simpler if you already run CUDA 11.x/12.x on host.
- Cons: You may need small build fixes for CUDA extensions.

> If you can do Option A, it’s the cleanest Design A story.

---

## 2) Prerequisites (host machine)

### 2.1 Verify GPU driver
```bash
nvidia-smi
```

### 2.2 Install essential build tools (host)
```bash
sudo apt update
sudo apt install -y git build-essential cmake pkg-config
```

### 2.3 (If using Docker) Install NVIDIA Container Toolkit
Follow NVIDIA’s official guide for Ubuntu.  
After install, verify GPU passthrough:
```bash
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

---

## 3) Get the Pixel2Mesh code

```bash
git clone https://github.com/noahcao/Pixel2Mesh.git
cd Pixel2Mesh
git submodule update --init
```

---

## 4) Environment setup (Design A)

## 4A) Option A — Docker baseline environment (strongly recommended)

Create `Dockerfile` at the repo root:

```Dockerfile
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates build-essential cmake \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh && \
    bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh
ENV PATH=/opt/conda/bin:$PATH

RUN conda create -n p2m python=3.7 -y && conda clean -a -y
SHELL ["bash", "-lc"]

RUN conda activate p2m && \
    conda install -y -c pytorch pytorch=1.1 torchvision=0.3 cudatoolkit=10.0 && \
    conda install -y -c conda-forge opencv=4.1 scipy=1.3 scikit-image=0.15 && \
    pip install easydict pyyaml tensorboardx trimesh shapely

WORKDIR /workspace
CMD ["bash", "-lc", "conda activate p2m && bash"]
```

Build + run:
```bash
docker build -t p2m:designA .
docker run --rm -it --gpus all -v "$PWD":/workspace p2m:designA
```

Inside the container, you’ll be at `/workspace` with the repo mounted.

---

## 4B) Option B — Native conda environment (Ubuntu 22.04)

> Use this if Docker is not feasible.

```bash
conda create -n p2mA python=3.7 -y
conda activate p2mA

# Try matching the repo’s “known-good” versions as close as possible
conda install -y -c pytorch pytorch=1.1 torchvision=0.3
conda install -y -c conda-forge opencv=4.1 scipy=1.3 scikit-image=0.15
pip install easydict pyyaml tensorboardx trimesh shapely
```

If you get build errors later:
- ensure you have `gcc/g++` and `cmake`,
- ensure `nvcc` is available (from your CUDA toolkit),
- document any changes as “compatibility fixes”.

---

## 5) Compile required CUDA extensions (must-do)

From repo root:

```bash
cd external/chamfer
python setup.py install

cd ../neural_renderer
python setup.py install

cd ../../
```

**If this fails**, the most common issues are:
- mismatched CUDA toolkit vs PyTorch CUDA,
- too-new GCC for old CUDA (Docker option avoids this),
- missing `libgl1` / GL runtime for OpenCV.

---

## 6) Dataset setup (ShapeNet `data_tf` baseline)

### 6.1 Download the standard dataset subset
The repo uses ShapeNet, and recommends the **official `data_tf` subset** by default (137×137 RGBA).  
Download it from the repo’s linked Drive folder.

You also need the **meta files** archive, which helps create the expected folder tree.

### 6.2 Create the expected folder layout

Target structure:

```
datasets/data
├── ellipsoid
│   ├── face1.obj
│   ├── face2.obj
│   ├── face3.obj
│   └── info_ellipsoid.dat
├── pretrained
│   └── (checkpoint .pth files)
└── shapenet
    ├── data_tf
    │   ├── 02691156
    │   ├── 02828884
    │   └── ...
    └── meta
        └── ...
```

### 6.3 Link or place `data_tf`
If you downloaded and extracted `data_tf` elsewhere, you can **symlink** it:

```bash
mkdir -p datasets/data/shapenet
ln -s /ABSOLUTE/PATH/TO/data_tf datasets/data/shapenet/data_tf
```

Place meta under:
```bash
datasets/data/shapenet/meta
```

---

## 7) Checkpoints for Design A (baseline choice)

### 7.1 Download baseline checkpoint
For a “closest to original” baseline, use:
- **VGG backbone migrated checkpoint** (converted from the official TF model)

Place it here:
```bash
mkdir -p datasets/data/pretrained
# copy downloaded .pth into datasets/data/pretrained/
```

### 7.2 Record provenance (important for report)
In your report log:
- checkpoint name (filename),
- source (repo’s pretrained checkpoint link),
- date downloaded,
- SHA256 hash (optional but ideal for reproducibility).

Example:
```bash
sha256sum datasets/data/pretrained/YOUR_CHECKPOINT.pth > datasets/data/pretrained/checkpoint_sha256.txt
```

---

## 8) Configuration (YAML) — how to pick the right one

Pixel2Mesh runs through YAML configs that override defaults.  
You can use example YAML files in `experiments/`.

**Rule:** Use a YAML intended for **VGG** if you’re using the migrated VGG checkpoint.

Create a new Design-A YAML to keep things tidy, e.g.:
```
experiments/designA_vgg.yml
```

In the report, include:
- the YAML filename,
- any changes you made (if any),
- batch size / GPU id.

---

## 9) Run Design A (inference + evaluation)

### 9.1 Inference on your own images (to generate meshes for poster)
Prepare an image folder:
```
my_images/
  0001.png
  0002.png
  ...
```

Run:
```bash
python entrypoint_predict.py \
  --options experiments/designA_vgg.yml \
  --checkpoint datasets/data/pretrained/YOUR_VGG_CHECKPOINT.pth \
  --folder my_images
```

**Output:** The script produces reconstructed meshes (OBJ) and/or intermediate outputs depending on config.

> Poster tip: generate **6–10 meshes**, then render them in MeshLab/Blender with consistent lighting and camera.

### 9.2 Evaluation on ShapeNet (`data_tf`)
Run:
```bash
python entrypoint_eval.py \
  --name designA_vgg_baseline \
  --options experiments/designA_vgg.yml \
  --checkpoint datasets/data/pretrained/YOUR_VGG_CHECKPOINT.pth
```

**Record:**
- CD (Chamfer Distance),
- F1 scores,
- runtime per batch (if printed),
- GPU utilization snapshot (`nvidia-smi`).

---

## 10) (Optional) Training in Design A

If time permits, you can train from scratch or finetune **without optimization changes**:

```bash
python entrypoint_train.py --name designA_train --options experiments/designA_vgg.yml
```

**Note:** training is heavier and may require multi-GPU / long time. For the poster, inference + eval is usually enough.

---

## 11) What to include in Chapter 4 (Design A evidence checklist)

### 11.1 Methodology overview (4.1)
- Baseline model: Pixel2Mesh (VGG backbone baseline)
- Dataset: ShapeNet `data_tf` subset (aligns with official implementation)
- Losses/metrics used (as implemented): CD and F1
- Pipeline: image → feature extraction → graph deformation → mesh output

### 11.2 Preliminary design specification (4.2)
Document:
- Input: 137×137 RGBA images
- Output: reconstructed mesh (OBJ)
- Constraints: GPU-only, CUDA extension compilation
- Requirements: dataset structure, meta files, checkpoints

### 11.3 Functional verification (simulation)
Include:
- Successful inference examples (mesh renders)
- Evaluation results table (CD, F1)
- Screenshots/log excerpts showing the model ran successfully

---

## 12) Poster-ready outputs (quick recipe)

### 12.1 Collect outputs
- Save **input image**, **predicted mesh render**, and optionally **GT mesh** (if available)
- Make a 2×3 grid figure per category (airplane, car, chair, etc.)

### 12.2 Produce consistent renders
Use MeshLab:
- same material, same lighting,
- white background,
- same camera angle for all samples.

---

## 13) Troubleshooting cheatsheet

### Build errors in `chamfer` / `neural_renderer`
- Prefer Docker Option A to avoid compiler/CUDA mismatch.
- Ensure `nvcc` exists: `nvcc --version`
- Ensure PyTorch CUDA matches toolkit (or use CPU-only? repo expects GPU).

### OpenCV import error (libGL)
```bash
sudo apt install -y libgl1 libglib2.0-0
```

### “CUDA not available”
- Check `nvidia-smi`
- In Docker, verify `--gpus all`
- Confirm PyTorch sees CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

---

## 14) Design A deliverables (what you should have at the end)

1. **Reproducible environment**
   - Dockerfile or conda env export

2. **Dataset correctly placed**
   - `datasets/data/shapenet/data_tf` and `meta`

3. **Baseline checkpoint documented**
   - filename + hash + source

4. **Logs**
   - `entrypoint_eval.py` output saved
   - `nvidia-smi` snapshot

5. **Meshes and renders**
   - at least 6–10 example reconstructions

---

## 15) One-command “sanity run” checklist

After setup, these should all work:

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import neural_renderer; import chamfer"
python entrypoint_eval.py --name designA_test --options experiments/designA_vgg.yml --checkpoint datasets/data/pretrained/YOUR_VGG_CHECKPOINT.pth
```

---

## Notes for your Design Plan

- **Design A:** VGG migrated checkpoint + ShapeNet `data_tf` (baseline reproduction)
- **Design B:** Same dataset, same model behavior; introduce modern CUDA optimizations + profiling
- **Design C:** FaceScape dataloader + training/inference on faces (domain shift), after B is stable

