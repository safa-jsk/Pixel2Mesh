# Design B Full ShapeNet Evaluation

- Run name: `designB_full`
- Config: `configs/defaults/designB.yml`
- Checkpoint: `datasets/data/pretrained/tensorflow.pth.tar`
- Runtime flags: `--gpus 1 --batch-size 4 --num-workers 0 --shm-size=16g --cpus=8 --warmup-iters 15 --cudnn-benchmark --tf32`
- Source log: `artifacts/logs/designB/designB_full/designb_0304100101_eval.log`

## Final Metrics

- `cd`: **0.000451**
- `f1_tau`: **0.656700**
- `f1_2tau`: **0.795100**

## Timing

- Total evaluation time: **704.08 s** (**11.73 min**)
- Average inference time per image: **0.00775 s** (**7.75 ms**) *(GPU forward pass only: 30.99 ms batch / 4)*
- Average time per sample (wall clock): **16.08 ms** *(includes metric computation)*
- Average batch inference time: **0.03099 s** (**30.99 ms**)
- Throughput: **62.18 images/s**

## Performance Configuration (Design B Optimizations)

- Warmup iterations: **15** (cold-start eliminated)
- AMP mixed precision: **disabled** (P2M sparse ops unsupported in half)
- torch.compile: **disabled**
- cuDNN benchmark: **enabled**
- TF32 tensor cores: **enabled**
- GPU: NVIDIA GeForce RTX 4090
- PyTorch: 2.5.1+cu124, CUDA 12.4

## Per-Category Results

| Category     | Samples |      CD |  F1@τ | F1@2τ |
|--------------|--------:|--------:|------:|------:|
| Airplane     |    4045 | 0.000381 | 0.7588 | 0.8464 |
| Bench        |    1816 | 0.000485 | 0.6498 | 0.7828 |
| Cabinet      |    1572 | 0.000345 | 0.6461 | 0.8062 |
| Car          |    7496 | 0.000252 | 0.6934 | 0.8559 |
| Chair        |    6778 | 0.000525 | 0.5867 | 0.7403 |
| Display      |    1095 | 0.000591 | 0.5707 | 0.7214 |
| Lamp         |    2318 | 0.001074 | 0.5606 | 0.6854 |
| Loudspeaker  |    1618 | 0.000651 | 0.5243 | 0.6957 |
| Rifle        |    2372 | 0.000394 | 0.7643 | 0.8522 |
| Sofa         |    3173 | 0.000450 | 0.5554 | 0.7377 |
| Table        |    8509 | 0.000390 | 0.7093 | 0.8308 |
| Telephone    |    1052 | 0.000363 | 0.7313 | 0.8497 |
| Watercraft   |    1939 | 0.000569 | 0.5976 | 0.7401 |
| **OVERALL**  |  **43783** | **0.000451** | **0.6567** | **0.7951** |

## Speedup vs Design A

| Metric              | Design A CPU | Design A GPU | Design B  | vs CPU   | vs GPU  |
|---------------------|-------------:|-------------:|----------:|---------:|--------:|
| cd                  | 0.000498     | 0.000498     | 0.000451  | —        | —       |
| f1_tau              | 0.642188     | 0.642188     | 0.656700  | —        | —       |
| f1_2tau             | 0.780343     | 0.780343     | 0.795100  | —        | —       |
| Total time (min)    | 137.51       | 22.00        | 11.73     | **11.72×** | **1.88×** |
| Time/sample (ms)    | 692.05       | 32.68        | 16.08     | **43.0×** | **2.03×** |
| GPU inference (ms)  | —            | 32.68        | 7.75      | —        | **4.22×** |
| Throughput (img/s)  | 1.44         | 30.60        | 62.18     | **43.2×** | **2.03×** |

## Mesh Generation

- Meshes saved: **75 files** (25 samples × 3 stages; 3 of 26 target samples not found in test_tf)
- Output directory: `artifacts/outputs/meshes/designB`
- Detailed results: `artifacts/logs/designB/designB_full/designB_full/sample_results.csv`
- Batch results: `artifacts/logs/designB/designB_full/designB_full/batch_results.csv`
- Summary JSON: `artifacts/logs/designB/designB_full/designB_full/evaluation_summary.json`
