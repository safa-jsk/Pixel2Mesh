# Design C Guideline — Pixel2Mesh + Modern CUDA + FaceScape (Ubuntu 22.04)

> **Design C goal:** keep the Pixel2Mesh method and training logic *as close as possible to the original*, keep your **Design B CUDA modernization**, and **swap the dataset** from ShapeNet → **FaceScape** (single-image → 3D mesh for faces).

---

## 0) Scope, ethics, and “what you can show” (important for poster/report)

FaceScape is a facial dataset with strict license terms and a **publishable list** (only some identities/expressions are allowed to appear in publications/presentations).

**Hard rule for your poster:**
- Only use **publishable** FaceScape items for any screenshots, mesh renders, qualitative examples, or demo images.
- For everything else, keep visualizations private (for internal experiments only).

Why this matters:
- The FaceScape project page explicitly states that portraits/images/rendered models **cannot be published**, *except* the entries listed in the “publishable list”, and that the dataset is for **non-commercial** research use. (See the FaceScape project page section “Data Access / KEY POINTS” and “FAQ”.)

---

## 1) Choose the FaceScape source that best matches Pixel2Mesh

Pixel2Mesh needs (at minimum):
1. **A single input image**.
2. **A ground-truth 3D mesh** (or at least a point cloud sampled from that mesh).
3. A camera model (for differentiable rendering losses), or a consistent “synthetic camera” setup.

FaceScape offers multiple sources. For Pixel2Mesh, you have two practical options:

### Option C1 (recommended): FaceScape **Multi-view Data**
This is closest to Pixel2Mesh’s needs because it includes:
- Multi-view images
- Camera parameters
- Corresponding reconstructed 3D shapes (meshes)

**How we use it in Design C:**
- Treat **each view** as a training sample: `(image_view → 3D face mesh)`
- Use the provided camera parameters to build the projection matrix required by the renderer.
- Use one view per sample (single-view training), but optionally randomize views for augmentation.

### Option C2 (supporting asset): FaceScape **TU Models**
TU Models are topologically-uniform meshes (same connectivity across identities/expressions), plus textures and displacement maps.

**How we use it in Design C:**
- Build a **better template mesh** for Pixel2Mesh (face-shaped instead of ellipsoid).
- Optionally create extra synthetic renders (if you decide to expand training data later).

> In practice: **Multi-view Data** drives training; **TU Models** help you build a face template and/or extra synthetic samples.

---

## 2) Which of your downloaded FaceScape files to use (based on your list)

You mentioned these downloads:
- `fsmview_trainset_images_001-020.zip`
- `fsmview_trainset_shape_001-020.zip`
- `facescape_trainset_001_100.zip`

### Mapping to the options above
- `fsmview_*images*` + `fsmview_*shape*`  → **Option C1 (Multi-view Data)**  
  These appear under “Multi-View Data” and are very large; start with a few parts to validate your pipeline.

- `facescape_trainset_001_100.zip` → likely **Option C2 (TU Models)** chunk(s)  
  Use it primarily to create a face template and adjacency/unpooling hierarchy (see Section 6).

### Recommended order to start
1. Start with the **Multi-view Data** pair for a small chunk:
   - `fsmview_trainset_images_001-020.zip`
   - `fsmview_trainset_shape_001-020.zip`
2. Use `facescape_trainset_001_100.zip` only after your loader + camera projection + training loop are confirmed.

---

## 3) Prerequisites (assumes Design B environment is already working)

Design C should be built **on top of** your Design B “modern CUDA” branch.

**Minimum success criteria before Design C work:**
- Pixel2Mesh trains on ShapeNet subset (even for a few iterations)
- Chamfer module builds and runs on your GPU
- Neural renderer builds and can render at least one batch without crashing

If any of the above fails, fix them in Design B first — otherwise you will not know whether failures come from dataset changes or CUDA/toolchain issues.

---

## 4) Directory layout you should standardize early

Create a clean `data/` layout (example):

```
Pixel2Mesh/
  data/
    facescape/
      mview/
        images/                # extracted multi-view images
        shapes/                # extracted per-sample meshes/shape info
        meta/                  # publishable list, id splits, cached index
      tu/
        models_reg/
        dpmap/
        textures/
        meta/
```

**Keep a single “index file”** that maps each training sample to:
- image path
- mesh path
- camera parameters path (or camera record)
- identity id / expression id / view id
- publishable flag

Example:
```
data/facescape/mview/meta/index_train.jsonl
data/facescape/mview/meta/index_val.jsonl
data/facescape/mview/meta/index_test.jsonl
```

---

## 5) Dataset preparation pipeline (Multi-view Data)

### 5.1 Extract + sanity-check
Unzip into the `mview/` folder. Verify:
- you have images
- you have meshes for the same items
- you have camera parameter files (or a single camera file per tuple)

**Sanity tests you must do before training:**
1. Load one image.
2. Load the paired GT mesh.
3. Render the mesh using the camera parameters.
4. Overlay the render on the image and verify alignment.

If alignment is wrong, training losses will be meaningless.

### 5.2 Create a “single-view sample” abstraction
Pixel2Mesh expects a sample like:
- `img`: tensor (H, W, C)
- `gt_mesh`: vertices + faces (or sampled GT points)
- `camera`: projection matrix or intrinsics/extrinsics in the format your renderer expects

In your FaceScape adapter, each view becomes:
```
(sample_id = identity, expression, view)
img = image(identity, expression, view)
gt_mesh = mesh(identity, expression)  # if mesh is shared across views
camera = camera(identity, expression, view)
```

### 5.3 Identity-based splits (avoid leakage)
Do not randomly split by views, or you risk the same identity appearing in train and test.

**Recommended split:**
- Split by **identity** (subject).  
- Keep all expressions/views of a subject in the same split.

Typical split (example):
- train: 80% identities
- val: 10% identities
- test: 10% identities

Store these identity lists in `mview/meta/`.

---

## 6) Template mesh strategy (critical “Design C” choice)

Pixel2Mesh predicts mesh deformations from an initial template (in many repos: an ellipsoid-based graph pyramid).

For faces, you have two viable tracks:

### Track C-Minimal (fast baseline): keep ellipsoid template
Pros:
- Minimal code changes
- Lets you focus on dataset + camera + training stability

Cons:
- Slower convergence
- Worse fidelity for faces (template is far from target)

**You should do this first** to get end-to-end training.

### Track C-FaceTemplate (recommended for quality): build a face template from TU models
Because FaceScape TU models are topologically uniform, they are ideal to create:
- A **mean neutral face** template
- A **graph pyramid** via mesh simplification
- Consistent adjacency matrices

Steps:
1. Pick a **neutral expression** mesh from TU models for many identities.
2. Compute a mean face (vertex-wise average).
3. Save as `facescape_face_template.obj`.
4. Build multi-resolution versions (coarse → fine) and store:
   - vertices at each level
   - faces at each level
   - adjacency
   - unpooling maps (coarse-to-fine correspondences)

**Important:** The exact hierarchy method depends on your Pixel2Mesh implementation (some use precomputed unpool indices). Keep the same method as the repo uses for ShapeNet templates.

---

## 7) Input image format: match Pixel2Mesh expectations

In your Pixel2Mesh PyTorch README, the `data_tf` ShapeNet subset uses:
- **137×137**
- **4 channels (RGB + alpha)**

For FaceScape, decide one of these:

### Option I (closest): create 137×137 RGBA
- Resize/crop the face region to 137×137
- Create an **alpha mask** for the face (foreground=1, background=0)

How to create alpha:
- If FaceScape provides a facial mask or you can derive it from TU model projection, use that.
- Otherwise, use a segmentation model (only as preprocessing; keep it out of “core method” to avoid scope creep).

### Option II (simpler): use 3-channel RGB
- Modify the first conv layer to accept 3 channels (if it expects 4)
- Adjust normalization accordingly

> Recommendation: start with **Option II** for pipeline validation, then add alpha once training is stable.

---

## 8) Camera parameters: how to make them usable in Pixel2Mesh

Pixel2Mesh’s differentiable renderer usually expects a **(B×3×4) projection matrix** or equivalent.

Your job is to convert FaceScape camera data (intrinsics/extrinsics) into that expected matrix.

### 8.1 Standard form
A common form is:
```
P = K [R | t]
```
where:
- `K` = intrinsics (3×3)
- `R` = rotation (3×3)
- `t` = translation (3×1)

### 8.2 Your non-negotiable validation
Before training:
- Render GT mesh using `P`
- Check that the silhouette and facial contour align with the image

**If not aligned:**
- You may have coordinate system mismatch (right-handed vs left-handed)
- Or different conventions (camera-to-world vs world-to-camera)
- Or your renderer expects normalized device coordinates (NDC)

Fix this *first*, otherwise the silhouette loss / reprojection losses will push the network in wrong directions.

---

## 9) Training plan (what to run, in what order)

### Phase 1 — “Smoke test” (1–2 hours)
Goal: prove the whole pipeline works on **tiny data**.

1. Use **10–50 samples** only.
2. Disable expensive losses at first (keep Chamfer + basic regularization).
3. Train for ~200–500 iterations.
4. Confirm:
   - loss decreases
   - outputs are non-NaN
   - meshes are valid (no exploding vertices)

Deliverables:
- A few predicted meshes (OBJ) + renders
- A screenshot for your report (internal)

### Phase 2 — “Baseline training” (1–2 days on your GPU)
Goal: stable training on a small but meaningful subset.

1. Use 10–30 identities, all expressions, a few views each.
2. Enable silhouette / reprojection loss once camera conversion is correct.
3. Use mixed precision only if stable.

Deliverables:
- Quantitative metrics on val split
- Qualitative examples from **publishable list** only (for poster)

### Phase 3 — “Scale up” (1–2 weeks, depending on compute)
Goal: train on a larger portion of the dataset.

1. Increase identities gradually.
2. Cache GT sampled points (for Chamfer) to speed up epochs.
3. Optimize data loading (prefetch, workers, pinned memory).

Deliverables:
- Final metrics
- Final poster-quality renders (publishable entries)

---

## 10) What to measure (Design C evaluation checklist)

### Quantitative (recommended)
- Chamfer Distance (CD)
- F-score at one or two thresholds (e.g., 1mm/2mm in face scale, if you normalize consistently)
- Normal consistency (optional)
- Edge length / Laplacian regularization terms (as training diagnostics)

### Qualitative
- Side-by-side: input image, predicted render, GT render
- Mesh-only rotation video (optional, for demo)
- Expression robustness: show a few expressions per identity (publishable only)

---

## 11) Poster/report assets you should produce during Design C

**Minimum poster visuals:**
- Pipeline diagram: Image → CNN → Graph deformation stages → Mesh
- One “failure case” and one “success case” (publishable only)
- A table: Design A vs Design B vs Design C (dataset + CUDA + outcome)
- A plot: training loss curve

**Make this easy on yourself:**
- Create a `poster_assets/` folder and auto-export:
  - PNG renders
  - OBJ meshes
  - metrics CSV
  - training logs

---

## 12) Common pitfalls (and how to avoid wasting days)

1. **Camera mismatch** → losses don’t converge  
   Fix by GT render overlay test.

2. **Train/test leakage by identity**  
   Always split by subject id.

3. **Template too far (ellipsoid)**  
   Use it only to validate; move to face template when stable.

4. **Alpha channel confusion**  
   Start with RGB, then add alpha.

5. **Publishing non-publishable faces**  
   Enforce a “publishable-only export” rule in code.

---

## 13) A practical “Day-by-day” kickoff plan (Ubuntu 22.04)

### Day 1
- Confirm Design B environment builds and runs.
- Extract FaceScape mini chunk (1 zip pair).
- Write loader to return `(img, mesh, camera)`.

### Day 2
- Implement GT render overlay test.
- Create identity split + index files.
- Run smoke test training on 10–50 samples.

### Day 3–4
- Stabilize losses + configs.
- Export publishable-only qualitative samples for poster.

### Day 5+
- Build face template from TU models (optional but recommended).
- Scale training gradually.
- Benchmark performance (speed/memory) and compare vs Design A/B.

---

## 14) “Reference facts” you may cite in your report (put in your own words)

These are useful, report-friendly facts you can reference:
- FaceScape provides large-scale 3D face models and multi-view images; multi-view data includes images + camera parameters + 3D shapes, and is very large (hundreds of thousands of images).
- TU models are topologically uniform and include base meshes + displacement maps + textures, which helps in alignment and building models.

*(Keep any dataset access/policy notes consistent with the FaceScape license terms.)*

---

## 15) Commands / links (kept inside code blocks)

```text
Pixel2Mesh (noahcao, PyTorch):
https://github.com/noahcao/Pixel2Mesh

FaceScape project page (data description + license notes):
https://nju-3dv.github.io/projects/FaceScape/

FaceScape GitHub (docs + toolkit references):
https://github.com/zhuhao-nju/facescape
```

---

## 16) “Done” definition for Design C (what you should be able to show)

You can consider Design C successful when you can:
- Train Pixel2Mesh end-to-end on FaceScape multi-view-derived single-view samples
- Produce predicted face meshes that are visually reasonable
- Report at least one quantitative metric (e.g., Chamfer Distance) on an identity-held-out test split
- Provide poster-ready qualitative examples using **publishable** FaceScape items only

---

### Appendix A — Suggested config naming convention

Use a consistent naming scheme:

- `configs/facescape_vgg_ellipsoid.yaml`   (Track C-Minimal)
- `configs/facescape_resnet_ellipsoid.yaml`
- `configs/facescape_resnet_facemplate.yaml` (Track C-FaceTemplate)

---

### Appendix B — “Publishable-only export” safeguard (recommended)

In your visualization/export script:
- Load publishable list
- Refuse to export anything not in the list
- Print a clear warning

This avoids accidental license violations when preparing poster figures.
