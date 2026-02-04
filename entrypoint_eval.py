\"\"\"
Pixel2Mesh Evaluation Entrypoint with Design B Performance Optimizations
=========================================================================

Performance Flags:
- --warmup-iters: GPU warmup iterations (default: 15)
- --amp/--no-amp: AMP mixed precision (default: enabled)
- --compile/--no-compile: torch.compile (default: disabled)
- --cudnn-benchmark/--no-cudnn-benchmark: cuDNN autotuner (default: enabled)
- --tf32/--no-tf32: TF32 tensor cores (default: enabled)
\"\"\"
import argparse
import sys

from functions.evaluator import Evaluator
from options import update_options, options, reset_options


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pixel2Mesh Evaluation Entrypoint',
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

    # ===== DESIGN B PERFORMANCE FLAGS =====
    parser.add_argument(
        '--warmup-iters', 
        type=int, 
        default=15,
        help='Number of GPU warmup iterations before timing (0 to disable)'
    )
    # NOTE: AMP disabled by default - P2M sparse graph convolutions don't support half precision
    parser.add_argument('--amp', dest='amp_enabled', action='store_true', default=False,
                        help='Enable AMP mixed precision (disabled by default - sparse ops unsupported)')
    parser.add_argument('--no-amp', dest='amp_enabled', action='store_false',
                        help='Disable AMP mixed precision')
    parser.add_argument('--compile', dest='compile_enabled', action='store_true', default=False,
                        help='Enable torch.compile (PyTorch 2.x)')
    parser.add_argument('--no-compile', dest='compile_enabled', action='store_false',
                        help='Disable torch.compile')
    parser.add_argument('--cudnn-benchmark', dest='cudnn_benchmark', action='store_true', default=True,
                        help='Enable cuDNN benchmark mode')
    parser.add_argument('--no-cudnn-benchmark', dest='cudnn_benchmark', action='store_false',
                        help='Disable cuDNN benchmark mode')
    parser.add_argument('--tf32', dest='tf32_enabled', action='store_true', default=True,
                        help='Enable TF32 tensor core math')
    parser.add_argument('--no-tf32', dest='tf32_enabled', action='store_false',
                        help='Disable TF32')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger, writer = reset_options(options, args, phase='eval')

    evaluator = Evaluator(
        options, 
        logger, 
        writer,
        warmup_iters=args.warmup_iters,
        amp_enabled=args.amp_enabled,
        compile_enabled=args.compile_enabled,
        cudnn_benchmark=args.cudnn_benchmark,
        tf32_enabled=args.tf32_enabled
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
