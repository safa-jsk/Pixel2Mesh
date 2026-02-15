# Documentation Organization

**Last Updated:** February 16, 2026

This document describes the organized structure of all documentation in the Pixel2Mesh repository.

---

## 📁 Root Directory

Essential documentation for the entire project:

| File                                                     | Purpose                                                       |
| -------------------------------------------------------- | ------------------------------------------------------------- |
| [README.md](./README.md)                                 | Main project documentation with Pipeline Traceability section |
| [DOCKER_SETUP.md](./DOCKER_SETUP.md)                     | Docker environment setup instructions                         |
| [DOCKER_QUICK_REFERENCE.md](./DOCKER_QUICK_REFERENCE.md) | Docker command quick reference                                |

---

## 📚 docs/ - Methodology Documentation

Comprehensive thesis-ready documentation for all designs:

| File                                                                            | Lines | Purpose                          |
| ------------------------------------------------------------------------------- | ----- | -------------------------------- |
| [**DOCUMENTATION_INDEX.md**](./docs/DOCUMENTATION_INDEX.md)                     | 665   | Master index and usage guide     |
| [**PIPELINE_OVERVIEW.md**](./docs/PIPELINE_OVERVIEW.md)                         | 282   | Architecture diagrams and flow   |
| [**DESIGNS.md**](./docs/DESIGNS.md)                                             | 682   | All 4 design configurations      |
| [**TRACEABILITY_MATRIX.md**](./docs/TRACEABILITY_MATRIX.md)                     | 231   | Code-to-methodology mapping      |
| [**BENCHMARK_PROTOCOL.md**](./docs/BENCHMARK_PROTOCOL.md)                       | 665   | Timing and validation procedures |
| [**Design_C_Pixel2Mesh_Guideline.md**](./docs/Design_C_Pixel2Mesh_Guideline.md) | -     | Design C (planned) guideline     |

**Total:** 2,525 lines of comprehensive documentation

---

## 🔵 Design A CPU/ - CPU Baseline (1291 ms/image)

CPU-based baseline implementation for comparison:

| File                                                                                                                       | Type     | Purpose                          |
| -------------------------------------------------------------------------------------------------------------------------- | -------- | -------------------------------- |
| [**README.md**](./evaluation_results/Design%20A%20CPU/README.md)                                                           | Index    | Quick reference for Design A     |
| [**Design_A_Pixel2Mesh_Guideline.md**](./evaluation_results/Design%20A%20CPU/Design_A_Pixel2Mesh_Guideline.md)             | Guide    | Setup and execution instructions |
| [**P2M_DesignA_Evaluation_Summary.md**](./evaluation_results/Design%20A%20CPU/P2M_DesignA_Evaluation_Summary.md)           | Results  | Complete evaluation report       |
| [**P2M_DesignA_Metrics_Summary.md**](./evaluation_results/Design%20A%20CPU/P2M_DesignA_Metrics_Summary.md)                 | Analysis | Detailed metrics breakdown       |
| [**P2M_DesignA_Mesh_Generation_Summary.md**](./evaluation_results/Design%20A%20CPU/P2M_DesignA_Mesh_Generation_Summary.md) | Analysis | Mesh output analysis             |
| **designA_batch_results.csv**                                                                                              | Data     | 1,095 batch entries              |
| **designA_summary_metrics.csv**                                                                                            | Data     | 27 summary metrics               |

---

## 🟢 Design A GPU/ - Simple GPU (265 ms/image, 4.86× speedup)

Basic GPU migration without advanced optimizations:

| File                                                                                                               | Type      | Purpose                                |
| ------------------------------------------------------------------------------------------------------------------ | --------- | -------------------------------------- |
| [**README.md**](./evaluation_results/Design%20A%20GPU/README.md)                                                   | Index     | Quick reference for Design A_GPU       |
| [**DesignA_Evaluation_Summary.md**](./evaluation_results/Design%20A%20GPU/DesignA_Evaluation_Summary.md)           | Results   | Complete evaluation report (600 lines) |
| [**DesignA_Metrics_Summary.md**](./evaluation_results/Design%20A%20GPU/DesignA_Metrics_Summary.md)                 | Analysis  | Detailed metrics breakdown             |
| [**DesignA_Mesh_Generation_Summary.md**](./evaluation_results/Design%20A%20GPU/DesignA_Mesh_Generation_Summary.md) | Analysis  | Mesh output analysis                   |
| [**DesignA_Pipeline_Documentation.md**](./evaluation_results/Design%20A%20GPU/DesignA_Pipeline_Documentation.md)   | Technical | Pipeline implementation details        |
| **designA_batch_results.csv**                                                                                      | Data      | Per-batch metrics                      |

---

## 🟡 Design B/ - Optimized GPU (185 ms/image, 6.97× speedup)

Full CAMFM methodology implementation with advanced GPU optimizations:

| File                                                                                                                 | Type      | Purpose                           |
| -------------------------------------------------------------------------------------------------------------------- | --------- | --------------------------------- |
| [**README.md**](./evaluation_results/Design%20B/README.md)                                                           | Index     | Quick reference with CAMFM stages |
| [**Design_B_Pixel2Mesh_Guideline.md**](./evaluation_results/Design%20B/Design_B_Pixel2Mesh_Guideline.md)             | Guide     | Setup and execution instructions  |
| [**DesignB_Evaluation_Summary.md**](./evaluation_results/Design%20B/DesignB_Evaluation_Summary.md)                   | Results   | Complete evaluation report        |
| [**DesignB_Mesh_Generation_Summary.md**](./evaluation_results/Design%20B/DesignB_Mesh_Generation_Summary.md)         | Analysis  | Mesh output analysis              |
| [**DesignB_Pipeline_Methodology.md**](./evaluation_results/Design%20B/DesignB_Pipeline_Methodology.md)               | Technical | CAMFM implementation details      |
| [**DesignB_Pipeline_Implementation_Map.md**](./evaluation_results/Design%20B/DesignB_Pipeline_Implementation_Map.md) | Technical | Code location mapping             |
| [**DesignA_vs_DesignB_Comparison.md**](./evaluation_results/Design%20B/DesignA_vs_DesignB_Comparison.md)             | Analysis  | Performance comparison            |
| **batch_results.csv**                                                                                                | Data      | Per-batch timing and metrics      |

---

## 📊 Performance Summary

| Design          | Performance | Speedup   | Key Features                  |
| --------------- | ----------- | --------- | ----------------------------- |
| **A (CPU)**     | 1291 ms/img | 1.0×      | Baseline, CPU inference       |
| **A_GPU**       | 265 ms/img  | **4.86×** | Simple GPU migration          |
| **B (CAMFM)**   | 185 ms/img  | **6.97×** | Full CAMFM methodology        |
| **C (Planned)** | ~220 ms/img | ~5.86×    | GPU data pipeline + FaceScape |

---

## 🎯 Finding Specific Information

### For Thesis Defense

- **Architecture Diagrams:** [docs/PIPELINE_OVERVIEW.md](./docs/PIPELINE_OVERVIEW.md)
- **Methodology Mapping:** [docs/TRACEABILITY_MATRIX.md](./docs/TRACEABILITY_MATRIX.md)
- **Timing Protocols:** [docs/BENCHMARK_PROTOCOL.md](./docs/BENCHMARK_PROTOCOL.md)

### For Implementation Details

- **Design A:** [evaluation_results/Design A CPU/Design_A_Pixel2Mesh_Guideline.md](./evaluation_results/Design%20A%20CPU/Design_A_Pixel2Mesh_Guideline.md)
- **Design B:** [evaluation_results/Design B/Design_B_Pixel2Mesh_Guideline.md](./evaluation_results/Design%20B/Design_B_Pixel2Mesh_Guideline.md)
- **All Designs:** [docs/DESIGNS.md](./docs/DESIGNS.md)

### For Performance Analysis

- **Design A Results:** [evaluation_results/Design A CPU/P2M_DesignA_Evaluation_Summary.md](./evaluation_results/Design%20A%20CPU/P2M_DesignA_Evaluation_Summary.md)
- **Design A_GPU Results:** [evaluation_results/Design A GPU/DesignA_Evaluation_Summary.md](./evaluation_results/Design%20A%20GPU/DesignA_Evaluation_Summary.md)
- **Design B Results:** [evaluation_results/Design B/DesignB_Evaluation_Summary.md](./evaluation_results/Design%20B/DesignB_Evaluation_Summary.md)
- **Comparison:** [evaluation_results/Design B/DesignA_vs_DesignB_Comparison.md](./evaluation_results/Design%20B/DesignA_vs_DesignB_Comparison.md)

### For Code Navigation

- **In-Code Tags:** Search for `[DESIGN.X][CAMFM.Y]` in Python files
- **Code Mapping:** [docs/TRACEABILITY_MATRIX.md](./docs/TRACEABILITY_MATRIX.md)
- **Pipeline Map:** [evaluation_results/Design B/DesignB_Pipeline_Implementation_Map.md](./evaluation_results/Design%20B/DesignB_Pipeline_Implementation_Map.md)

---

## 🗂️ File Organization Principles

1. **Root:** Essential project documentation only
2. **docs/:** Comprehensive methodology and cross-design documentation
3. **Design Folders:** Design-specific results, analysis, and guidelines
4. **README per Design:** Quick reference with key characteristics

---

## 📝 Maintenance

### Adding New Documentation

- **Design-Specific:** Place in appropriate `Design X/` folder
- **Cross-Design:** Place in `docs/` folder
- **General Setup:** Place in root directory

### Updating Existing Documentation

- Check [docs/DOCUMENTATION_INDEX.md](./docs/DOCUMENTATION_INDEX.md) for file locations
- Update relevant design README after changes
- Update this file if structure changes

---

**Note:** All paths use URL-safe encoding for spaces (e.g., `Design%20A%20CPU`).
