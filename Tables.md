# Metrics Table

| Metric | Design A CPU | Design A GPU | Design B GPU | A CPU → A GPU | A GPU → B GPU | A CPU → B GPU |
|---|---:|---:|---:|---:|---:|---:|
| **Total Latency** | 129.42 min | 33.43 min | 12.10 min | 3.87× | 2.76× | 10.7× |
| **Throughput** | 0.77 img/s | 3.95 img/s | 60.30 img/s | 5.1× | 15.3× | 78.3× |
| **Chamfer Distance** | 0.000498 | 0.000498 | 0.000451 | 0.0% (same) | **−9.4% (better)** | **−9.4% (better)** |
| **F1@τ** | 64.22% | 64.22% | 65.67% | 0.0% (same) | **+1.45%** | **+1.45%** |
| **F1@2τ** | 78.03% | 78.03% | 79.51% | 0.0% (same) | **+1.48%** | **+1.48%** |

---

# Timing Comparison Table

| Metric | Design A CPU | Design A GPU | Design B GPU |
|---|---:|---:|---:|
| **Total Evaluation Time** | 129.42 min | 33.43 min | 12.10 min |
| **Total Time (seconds)** | 7,765.27 s | 2,006.04 s | 726.00 s |
| **Avg Inference Time/Image** | 1,290.98 ms | 253.35 ms | 16.58 ms |
| **Avg Batch Time** | 1,305.70 ms | 268.80 ms | 125.87 ms |
| **Throughput** | 0.77 img/s | 3.95 img/s | 60.30 img/s |
