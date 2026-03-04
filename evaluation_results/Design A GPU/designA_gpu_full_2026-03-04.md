# Design A GPU Full ShapeNet Evaluation

- Run name: `designA_gpu_full`
- Config: `configs/defaults/designA_vgg.yml`
- Checkpoint: `datasets/data/pretrained/tensorflow.pth.tar`
- Runtime flags: `--gpus 1 --batch-size 4 --num-workers 0 --shm-size=16g --cpus=8`
- Source log: `artifacts/logs/designA/designA_gpu_full/designa_vgg_0304072129_eval.log`

## Final Metrics

- `cd`: **0.000498**
- `f1_tau`: **0.642188**
- `f1_2tau`: **0.780343**

## Timing

- Total evaluation time: **1320.10 s** (**22.00 min**)
- Average inference time per image: **0.0327 s** (**32.68 ms**)
- Average batch processing time: **0.0711 s**
- Throughput: **30.60 images/s**
