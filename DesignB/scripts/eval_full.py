#!/usr/bin/env python
"""
Design B: Full Dataset Evaluation with Model Pipeline Controls
================================================================

Wraps the existing DesignB evaluator (src/pixel2mesh/engine/designb_evaluator.py)
with proper sys.path setup.

This evaluates Pixel2Mesh on the full dataset with:
- GPU residency (CAMFM.A2a)
- Warmup + CUDA-synced timing (CAMFM.A2b)
- Memory/layout tuning (cuDNN benchmark, TF32) (CAMFM.A2d)
- Optional AMP / torch.compile (disabled by default for P2M)
- Comprehensive metrics + artifact logging (CAMFM.A3)

Usage:
    python DesignB/scripts/eval_full.py \\
        --options configs/defaults/designB.yml \\
        --checkpoint datasets/data/pretrained/tensorflow.pth.tar \\
        --name designB_full_eval \\
        --warmup-iters 15 --cudnn-benchmark --tf32
"""
import os
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Import and run the canonical Design B evaluator entrypoint
from pixel2mesh.engine.designb_evaluator import main

if __name__ == "__main__":
    main()
