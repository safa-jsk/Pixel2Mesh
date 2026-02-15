# Evaluation Results

This directory contains evaluation results, metrics, and analysis for all design variants.

---

## 📁 Structure

```
evaluation_results/
├── Design A CPU/      - CPU baseline (1291 ms/image)
├── Design A GPU/      - Simple GPU (265 ms/image, 4.86× speedup)
├── Design B/          - Optimized GPU (185 ms/image, 6.97× speedup)
└── .archive/          - Archived old results
```

---

## 🔵 [Design A CPU](./Design%20A%20CPU/)

**Performance:** 1291 ms/image  
**Approach:** CPU-based baseline

### Contents:

- Evaluation summaries and metrics
- Batch results (1,095 entries)
- Mesh generation analysis
- CSV data files

[📂 View Design A CPU Results](./Design%20A%20CPU/)

---

## 🟢 [Design A GPU](./Design%20A%20GPU/)

**Performance:** 265 ms/image  
**Speedup:** 4.86× vs Design A

### Contents:

- Comprehensive evaluation summary (600 lines)
- Detailed metrics breakdown
- Pipeline documentation
- Batch results

[📂 View Design A GPU Results](./Design%20A%20GPU/)

---

## 🟡 [Design B](./Design%20B/)

**Performance:** 185 ms/image  
**Speedup:** 6.97× vs Design A, 1.43× vs Design A_GPU

### Contents:

- CAMFM methodology implementation
- Pipeline implementation mapping
- Design comparison analysis
- Optimized batch results

[📂 View Design B Results](./Design%20B/)

---

## 📊 Performance Summary

| Design        | Time (ms/img) | Speedup   | Methodology       |
| ------------- | ------------- | --------- | ----------------- |
| **A (CPU)**   | 1291          | 1.0×      | Baseline          |
| **A_GPU**     | 265           | **4.86×** | Simple GPU        |
| **B (CAMFM)** | 185           | **6.97×** | Full optimization |

---

## 🔍 Finding Specific Results

### By Design:

- **Design A (CPU baseline):** [Design A CPU/](./Design%20A%20CPU/)
- **Design A_GPU (simple GPU):** [Design A GPU/](./Design%20A%20GPU/)
- **Design B (optimized):** [Design B/](./Design%20B/)

### By Document Type:

- **Evaluation Summaries:** `*_Evaluation_Summary.md` in each design folder
- **Metrics Analysis:** `*_Metrics_Summary.md` in each design folder
- **Batch Data:** `*_batch_results.csv` in each design folder
- **Comparisons:** [Design B/DesignA_vs_DesignB_Comparison.md](./Design%20B/DesignA_vs_DesignB_Comparison.md)

---

## 📝 Usage

### Access Results:

```bash
# View Design B evaluation summary
cat evaluation_results/"Design B"/DesignB_Evaluation_Summary.md

# Analyze batch results
python -c "import pandas as pd; df = pd.read_csv('evaluation_results/Design B/batch_results.csv'); print(df.describe())"
```

### Generate New Results:

```bash
# Run Design A evaluation
./scripts/evaluation/run_designA_eval.sh

# Run Design B evaluation
./scripts/evaluation/run_designB_eval.sh
```

Results will be saved to `logs/` directory first, then can be moved here for archival.

---

## 🗂️ Archive

Old or superseded results can be moved to [.archive/](./.archive/) folder.

---

**Related Documentation:**

- [docs/DESIGNS.md](../docs/DESIGNS.md) - Design configurations
- [docs/TRACEABILITY_MATRIX.md](../docs/TRACEABILITY_MATRIX.md) - Methodology mapping
- [DOCUMENTATION_ORGANIZATION.md](../DOCUMENTATION_ORGANIZATION.md) - Overall structure
