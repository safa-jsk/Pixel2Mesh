#!/usr/bin/env python
"""
Design B: Sample Evaluation (Subset)
======================================

Same as eval_full.py but intended for quick smoke tests on a small
number of samples. Uses the same infrastructure.

Usage:
    python DesignB/scripts/eval_samples.py \\
        --options configs/defaults/designB.yml \\
        --checkpoint datasets/data/pretrained/tensorflow.pth.tar \\
        --name designB_samples_eval \\
        --warmup-iters 5 --cudnn-benchmark --tf32
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pixel2mesh.engine.designb_evaluator import main

if __name__ == "__main__":
    main()
