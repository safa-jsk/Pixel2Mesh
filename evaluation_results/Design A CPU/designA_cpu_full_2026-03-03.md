# Design A CPU Full ShapeNet Evaluation

- Run name: `designA_cpu_full`
- Config: `configs/defaults/designA_vgg.yml`
- Checkpoint: `datasets/data/pretrained/tensorflow.pth.tar`
- Runtime flags: `--gpus 0 --batch-size 4 --shm-size=16g --cpus=8`
- Source log: `artifacts/logs/designA/designA_cpu_full/designa_vgg_0303162235_eval.log`

## Final Metrics

- `cd`: **0.000498**
- `f1_tau`: **0.642188**
- `f1_2tau`: **0.780343**

## Timing

- Total evaluation time: **8250.84 s** (**137.51 min**)
- Average inference time per image: **0.6920 s** (**692.05 ms**)
- Average batch processing time: **0.6976 s**
- Throughput: **1.44 images/s**
