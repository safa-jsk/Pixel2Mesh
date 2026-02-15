# Design A_GPU (Simple GPU Migration)

**Performance:** 265 ms/image  
**Speedup:** 4.86× vs Design A  
**Configuration:** Model fully on GPU, simple migration

## Documentation

- [**DesignA_Evaluation_Summary.md**](./DesignA_Evaluation_Summary.md) - Complete evaluation results (600 lines)
- [**DesignA_Metrics_Summary.md**](./DesignA_Metrics_Summary.md) - Detailed metrics breakdown
- [**DesignA_Mesh_Generation_Summary.md**](./DesignA_Mesh_Generation_Summary.md) - Mesh output analysis
- [**DesignA_Pipeline_Documentation.md**](./DesignA_Pipeline_Documentation.md) - Pipeline details

## Data Files

- [**designA_batch_results.csv**](./designA_batch_results.csv) - Per-batch metrics

## Execution

```bash
# Same as Design A, but with GPU-enabled model
python entrypoint_eval.py \
  --name designA_gpu \
  --options experiments/baseline/lr_1e-4.yml \
  --checkpoint datasets/data/pretrained/tensorflow.pth.tar
```

## Key Characteristics

- ✅ Significant speedup (4.86×)
- ✅ Minimal code changes from Design A
- ❌ No advanced GPU optimizations
- ❌ No warmup or CUDA-synchronized timing

See [main documentation](../docs/) for methodology mapping.
