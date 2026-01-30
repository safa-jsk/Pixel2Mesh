# Pixel2Mesh — Design B Guideline (Modern CUDA Optimization + ShapeNet)

**Goal of Design B:**  
Keep **the same algorithm and dataset as Design A** (Pixel2Mesh + ShapeNet `data_tf`) and implement **measurable, modern CUDA-era performance improvements**—without changing output semantics more than necessary.

Design B should produce:
- **speed improvements** (throughput, latency, GPU utilization),
- **resource improvements** (VRAM, CPU usage, dataloader bottlenecks),
- **proof** via profiling (PyTorch Profiler / Nsight Systems),
- **verification** that quality is not degraded beyond acceptable tolerance.

---

## 0) What “counts” as Design B

✅ Allowed in Design B:
- Modernizing code and CUDA extension build to run on a **modern stack** (Ubuntu 22.04, CUDA 11.8/12.x, PyTorch 2.x).
- Enabling **AMP (fp16/bf16)**, **channels_last**, **cudnn benchmark**, **torch.compile**.
- Improving data input pipeline and minimizing CPU↔GPU sync.
- Replacing legacy CUDA kernels with updated equivalents **as long as the math is equivalent** (document differences).
- Writing new scripts/utilities for profiling and benchmarking.

❌ Avoid in Design B:
- Changing dataset/domain (no FaceScape yet).
- Changing model architecture/backbone (e.g., switching to ResNet) unless you explicitly treat it as a *separate ablation* (recommended to keep out of core Design B).
- Introducing new losses/metrics that change training objective (unless kept as optional experiment).

> **Design B principle:** *Same model + same data*, improved execution.

---

## 1) Design B deliverables (what you must have at the end)

1. **Modern runnable environment**
   - `environment.yml` or `requirements.txt`
   - Exact versions: Ubuntu, NVIDIA driver, CUDA toolkit, PyTorch, GCC

2. **Performance benchmark report**
   - Latency (ms/image, ms/batch)
   - Throughput (images/sec)
   - GPU utilization (%), GPU memory (MB/GB)
   - CPU utilization and dataloader time

3. **Profiling evidence**
   - PyTorch Profiler traces (Chrome trace JSON)
   - Optional: Nsight Systems timeline screenshots

4. **Correctness verification**
   - Same evaluation script & metrics as Design A
   - Quality within tolerance: e.g., CD/F1 difference small or explainable

5. **Poster-ready “before vs after”**
   - One slide/figure with speedup table + profiler snapshot

---

## 2) Strategy: build a controlled performance comparison

### 2.1 Freeze the baseline
Use Design A as the baseline:
- Same checkpoint (preferably VGG migrated checkpoint)
- Same evaluation set split
- Same batch size for the main comparison (also test scaling later)
- Same image resolution and preprocessing

### 2.2 Define success metrics (report-friendly)
Choose at least:
- **Latency**: `ms/image` and `ms/batch`
- **Throughput**: `images/sec`
- **VRAM**: peak memory during eval/inference
- **Quality**: CD and F1 from `entrypoint_eval.py`

### 2.3 Define the benchmark scenario
Example benchmark modes:
- **Inference Benchmark**: predict on N images (e.g., 1024) with fixed batch size
- **Eval Benchmark**: run `entrypoint_eval.py` and capture total time + metrics
- **Training Micro-benchmark** (optional): run 200 iterations and measure time/iter

---

## 3) Recommended Design B environment (Ubuntu 22.04 + PyTorch 2.x)

### 3.1 Create conda env (example for CUDA 11.8)
```bash
conda create -n p2mB python=3.10 -y
conda activate p2mB

# Install PyTorch with CUDA (choose the correct command from PyTorch site)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install opencv-python scikit-image scipy pyyaml easydict tensorboardx trimesh shapely
pip install matplotlib tqdm
```

> Notes:
- Python 3.10 is a good compromise for modern tooling and compatibility.
- If you need CUDA 12.x, install the matching PyTorch wheel (cu121/cu124) and ensure driver supports it.

### 3.2 Record system info (must for report)
```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cudnn', torch.backends.cudnn.version())"
nvidia-smi
gcc --version
```
Save outputs into `logs/system_info_designB.txt`.

---

## 4) Modernize and stabilize CUDA extensions (critical path)

Pixel2Mesh typically requires CUDA extensions:
- `external/chamfer`
- `external/neural_renderer`

### 4.1 Build extensions on modern PyTorch
From repo root:
```bash
pip install -U pip setuptools wheel
pip install ninja

cd external/chamfer
python setup.py install

cd ../neural_renderer
python setup.py install
cd ../../
```

### 4.2 Common modernization tasks
You may need to patch:
- deprecated ATen APIs and include paths
- `THC` usage → ATen/CUDA
- `torch.utils.cpp_extension` flags
- compute capability flags for RTX 2050 (Ampere)

Recommended build flags (if needed):
- set `TORCH_CUDA_ARCH_LIST="8.6"` (or include 8.6 for RTX 2050)
```bash
export TORCH_CUDA_ARCH_LIST="8.6"
```

> **Important:** any changes to kernel math must be documented. If you must change a kernel, add a unit test or a numeric check.

---

## 5) Performance optimization plan (implement in measured stages)

Implement improvements as **B1, B2, B3…** so you can show incremental gains.

### B1 — Baseline on modern stack (no “fast features” yet)
- Make it run on modern PyTorch/CUDA with the same outputs.
- Run eval/inference and record metrics + runtime.

Deliverable:
- `designB_B1_eval.log`
- baseline benchmark numbers on modern stack

### B2 — Enable cuDNN and memory format optimizations
#### 5.2.1 cuDNN knobs
Add (early in training/inference script):
```python
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

#### 5.2.2 channels_last (CNN feature extractor)
Where appropriate:
```python
model = model.to(memory_format=torch.channels_last)
images = images.to(memory_format=torch.channels_last)
```

Measure:
- throughput increase
- ensure no quality change

### B3 — AMP mixed precision (training + inference)
#### 5.3.1 Inference autocast
```python
with torch.cuda.amp.autocast(dtype=torch.float16):
    out = model(images)
```

#### 5.3.2 Training GradScaler
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast(dtype=torch.float16):
    loss = ...
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Measure:
- speedup
- VRAM reduction
- check metrics drift (should be minimal for inference; training may differ slightly)

### B4 — torch.compile (PyTorch 2.x)
Apply selectively:
```python
model = torch.compile(model, mode="max-autotune")
```

Rules:
- Start with inference only.
- If compile breaks due to dynamic control flow, compile submodules (e.g., encoder only).

Measure:
- compile overhead vs steady-state speed
- runtime across multiple batches (ignore first few warmup iterations)

### B5 — Data pipeline acceleration
Even if you focus on CUDA, dataloader bottlenecks matter.

Actions:
- increase `num_workers`
- enable `pin_memory=True`
- set `persistent_workers=True`
- prefetch factor
- avoid Python-side heavy transforms in the hot loop

Example:
```python
DataLoader(..., num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
```

Measure:
- dataloader time vs GPU compute time (Profiler)

### B6 — Reduce CPU↔GPU synchronization and redundant transfers
Common offenders:
- calling `.item()` in hot loop
- frequent logging every iteration
- moving tensors to CPU for debug
- unnecessary `.cpu()` or numpy conversions

Fix:
- log every N steps
- aggregate metrics on GPU where possible
- avoid `.item()` until the end of an epoch

---

## 6) Benchmarking: how to measure correctly (avoid misleading numbers)

### 6.1 Warm-up
Always run warm-up iterations before timing:
- 20–50 iterations warm-up is typical

### 6.2 Use CUDA events for accurate GPU timing
In your benchmark script:
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# forward pass loop
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end)
```

### 6.3 Report both end-to-end and pure GPU time
- **End-to-end** includes dataloader + preprocessing
- **GPU-only** isolates model compute

### 6.4 Fixed seeds (optional)
If you need closer numeric comparison:
- set seeds, disable nondeterministic ops
- but note: deterministic settings may reduce performance

---

## 7) Profiling: produce evidence screenshots for the report

### 7.1 PyTorch Profiler (recommended)
Example snippet:
```python
import torch.profiler as prof

with prof.profile(
    activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as p:
    # run ~50 iterations
    ...

print(p.key_averages().table(sort_by="cuda_time_total", row_limit=20))
p.export_chrome_trace("logs/designB_trace.json")
```

Open the trace:
- `chrome://tracing` (Chromium/Chrome) → load `designB_trace.json`

Capture:
- top kernels
- dataloader stalls
- CPU bottleneck phases

### 7.2 Nsight Systems (optional, excellent for CUDA story)
Install Nsight Systems and run:
```bash
nsys profile -o logs/designB_nsys --trace=cuda,nvtx,osrt python entrypoint_eval.py ...
```
Include one timeline screenshot in the report.

---

## 8) Verification: ensure Design B still matches Design A quality

### 8.1 Same evaluation command
Run:
```bash
python entrypoint_eval.py --name designB_eval --options experiments/designA_vgg.yml --checkpoint datasets/data/pretrained/YOUR_VGG_CHECKPOINT.pth
```

### 8.2 Acceptance criteria (suggested)
Define an acceptable range, e.g.:
- CD change within ±1–3% relative (or a small absolute delta)
- F1 change within ±1–2 points

If metrics change:
- identify if due to AMP
- rerun with AMP disabled to isolate cause
- document the tradeoff (speed vs precision)

---

## 9) A recommended directory structure for Design B outputs

```
logs/
  system_info_designB.txt
  B1_modern_stack_eval.log
  B2_channels_last_eval.log
  B3_amp_eval.log
  B4_compile_eval.log
  designB_trace.json
  designB_nsys.qdrep (optional)

bench/
  benchmark_infer.py
  benchmark_eval.sh

patches/
  chamfer_modernization.patch
  neural_renderer_modernization.patch

results/
  meshes_designB/
  renders_designB/
```

---

## 10) Suggested Design B “story” for Chapter 4

### 10.1 Design overview (4.1)
- Baseline: Design A reproduction on ShapeNet
- Problem: legacy stack, suboptimal GPU utilization
- Approach: port to modern CUDA/PyTorch and apply performance optimizations
- Verification: same dataset, same metric scripts, measured speedups

### 10.2 Design specification (4.2)
- Inputs/Outputs unchanged
- Constraints: must compile CUDA extensions; must keep model behavior consistent
- Requirements: improved performance with acceptable quality delta

### 10.3 Functional verification (simulation)
- Provide:
  - baseline metrics and runtime (A)
  - optimized metrics and runtime (B)
  - profiler evidence: “bottleneck moved from X to Y”
  - qualitative meshes: ensure visuals remain plausible

---

## 11) Practical “to-do” checklist (recommended order)

1. ✅ Make it run on modern PyTorch (B1)
2. ✅ Confirm eval works and matches Design A reasonably
3. ✅ Add benchmark scripts (repeatable timing)
4. ✅ Add AMP (B3) and measure
5. ✅ Add channels_last + cudnn benchmark (B2) and measure
6. ✅ Add torch.compile (B4) and measure
7. ✅ Fix dataloader bottlenecks (B5) using profiler
8. ✅ Produce final comparison report + poster figure

---

## 12) Common pitfalls (and how to avoid them)

- **Comparing without warm-up** → inflated speedup claims  
  ✅ Always warm up.

- **Measuring with `time.time()` only** → wrong GPU timings  
  ✅ Use CUDA events + `torch.cuda.synchronize()`.

- **AMP changing outputs**  
  ✅ Validate with AMP off; keep precision notes in report.

- **torch.compile breaks due to dynamic graphs**  
  ✅ Compile submodules first; fall back to eager for problematic parts.

- **Extensions compiled for wrong arch**  
  ✅ Set `TORCH_CUDA_ARCH_LIST` appropriately.

- **Dataloader becomes bottleneck after GPU is optimized**  
  ✅ Increase workers, pin memory, persistent workers; profile.

---

## 13) Final Design B “must include” table in your report (suggested)

Include a table comparing Design A vs Design B:
- Environment (PyTorch/CUDA)
- Inference throughput
- Eval time
- VRAM peak
- CD, F1
- Notes (AMP/compile enabled?)

> Keep the table to one page and add one profiler screenshot for credibility.

---

## 14) One-command sanity checks (after each B-stage)

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__, torch.version.cuda)"
python -c "import neural_renderer; import chamfer"
python entrypoint_eval.py --name designB_quick --options experiments/designA_vgg.yml --checkpoint datasets/data/pretrained/YOUR_VGG_CHECKPOINT.pth
```

---

## Suggested mapping to your designs

- **Design A:** VGG migrated checkpoint + ShapeNet `data_tf` (baseline reproduction)
- **Design B:** same as A but modernized + optimized CUDA execution (B1→B6 staged)
- **Design C:** domain shift to FaceScape once B is stable

