# Design A (CPU Baseline)

**Performance:** 1291 ms/image  
**Configuration:** Model runs on CPU, metrics on GPU (hybrid approach)

## Documentation

- [**Design_A_Pixel2Mesh_Guideline.md**](./Design_A_Pixel2Mesh_Guideline.md) - Setup and execution guide
- [**P2M_DesignA_Evaluation_Summary.md**](./P2M_DesignA_Evaluation_Summary.md) - Complete evaluation results
- [**P2M_DesignA_Metrics_Summary.md**](./P2M_DesignA_Metrics_Summary.md) - Detailed metrics breakdown
- [**P2M_DesignA_Mesh_Generation_Summary.md**](./P2M_DesignA_Mesh_Generation_Summary.md) - Mesh output analysis

## Data Files

- [**designA_batch_results.csv**](./designA_batch_results.csv) - Per-batch metrics (1,095 entries)
- [**designA_summary_metrics.csv**](./designA_summary_metrics.csv) - Summary statistics (27 metrics)

## Execution

```bash
python entrypoint_eval.py \
  --name designA_vgg_baseline \
  --options experiments/baseline/lr_1e-4.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar
```

## Key Characteristics

- ✅ Baseline for comparison
- ✅ Simple implementation
- ❌ Slow performance (CPU bottleneck)
- ❌ No GPU optimizations

See [main documentation](../docs/) for methodology mapping.
