#!/usr/bin/env python
"""
Design A Evaluation Entrypoint (Clean Baseline)
================================================

This is the legacy Design A evaluator. It does NOT include
Design B performance flags (warmup, AMP, torch.compile, etc.).

For Design B evaluation, use DesignB/scripts/eval_full.py.
"""
import argparse
import os
import sys

# Ensure src/ is on sys.path when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pixel2mesh.engine.evaluator import Evaluator
from pixel2mesh.options import update_options, options, reset_options


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pixel2Mesh Design A Evaluation',
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
    parser.add_argument('--gpus', help='number of GPUs to use', type=int)
    parser.add_argument('--num-workers', help='number of DataLoader workers', type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger, writer = reset_options(options, args, phase='eval')

    # Design A baseline: no warmup, no AMP, no compile, no cuDNN benchmark, no TF32
    evaluator = Evaluator(
        options,
        logger,
        writer,
        warmup_iters=0,
        amp_enabled=False,
        compile_enabled=False,
        cudnn_benchmark=False,
        tf32_enabled=False,
        model_on_gpu=False
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
