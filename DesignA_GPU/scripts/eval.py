#!/usr/bin/env python
"""
Design A GPU Evaluation Entrypoint
====================================

Same as Design A CPU but with GPU-enabled model inference.
The model is moved to GPU via DataParallel in the Evaluator.

No advanced optimizations (warmup, AMP, cuDNN tune, TF32, torch.compile).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pixel2mesh.engine.evaluator import Evaluator
from pixel2mesh.options import update_options, options, reset_options


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pixel2Mesh Design A GPU Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--shuffle', help='shuffle samples', default=False, action='store_true')
    parser.add_argument('--checkpoint', help='trained checkpoint file', type=str, required=True)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    parser.add_argument('--name', help='subfolder name of this experiment', required=True, type=str)
    parser.add_argument('--gpus', help='number of GPUs to use', type=int, default=1)
    parser.add_argument('--num-workers', help='number of DataLoader workers (default 0 for stability)',
                        type=int, default=0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger, writer = reset_options(options, args, phase='eval')

    # Design A GPU: no warmup, no AMP, no compile, no cuDNN benchmark, no TF32
    # GPU enablement happens via model.cuda() + DataParallel in the Evaluator
    evaluator = Evaluator(
        options,
        logger,
        writer,
        warmup_iters=0,
        amp_enabled=False,
        compile_enabled=False,
        cudnn_benchmark=False,
        tf32_enabled=False,
        model_on_gpu=True
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
