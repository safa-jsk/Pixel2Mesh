#!/usr/bin/env python
"""
Design A GPU Training Entrypoint
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pixel2mesh.engine.trainer import Trainer
from pixel2mesh.options import update_options, options, reset_options


def parse_args():
    parser = argparse.ArgumentParser(description='Pixel2Mesh Design A GPU Training')
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)
    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--num-epochs', help='number of epochs', type=int)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    parser.add_argument('--name', required=True, type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger, writer = reset_options(options, args)
    trainer = Trainer(options, logger, writer)
    trainer.train()


if __name__ == "__main__":
    main()
