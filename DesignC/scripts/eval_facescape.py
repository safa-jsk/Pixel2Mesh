#!/usr/bin/env python3
"""
Design C — Evaluate Pixel2Mesh on FaceScape dataset.

Usage:
    python DesignC/scripts/eval_facescape.py \
        --options configs/defaults/designA_vgg.yml \
        --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
        --facescape_root datasets/data/facescape \
        --splits_csv datasets/data/facescape/splits.csv \
        --name designC_facescape
"""

import argparse
import os
import sys

# --- Bootstrap: make pixel2mesh importable ---
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_src_dir = os.path.join(_repo_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Verify import works
try:
    import pixel2mesh  # noqa: F401
except ImportError as e:
    print(f"ERROR: Cannot import pixel2mesh from {_src_dir}")
    print(f"  {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Design C — FaceScape evaluation")
    parser.add_argument("--options", type=str, required=True, help="YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--facescape_root", type=str, required=True,
                        help="Root of FaceScape dataset (e.g. datasets/data/facescape)")
    parser.add_argument("--splits_csv", type=str, required=True,
                        help="CSV with image_path,mesh_path,split columns")
    parser.add_argument("--name", type=str, default="designC_facescape")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    # --- Validate file existence with clear messages ---
    if not os.path.isfile(args.options):
        print(f"ERROR: Config file not found: {args.options}")
        sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Import adapter (will raise FileNotFoundError with helpful message if data missing)
    sys.path.insert(0, os.path.join(_repo_root, "DesignC", "scripts"))
    from facescape_adapter import FaceScapeDataset

    print(f"[Design C] Initializing FaceScape evaluation")
    print(f"  Config:     {args.options}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data root:  {args.facescape_root}")
    print(f"  Split:      {args.split}")

    # Load dataset
    dataset = FaceScapeDataset(
        root=args.facescape_root,
        splits_csv=args.splits_csv,
        split=args.split,
    )

    # Load model config
    from pixel2mesh.options import update_options, options
    update_options(args.options)
    options.name = args.name

    import torch
    from torch.utils.data import DataLoader

    # Load model
    from pixel2mesh.models.p2m import P2MModel
    model = P2MModel(options.model, options.dataset.mesh_pos)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"[Design C] Running evaluation on {len(dataset)} samples...")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch["images"].to(device)
            # Forward pass (same as Design A/B)
            output = model(images)
            if i % 50 == 0:
                print(f"  Batch {i}/{len(dataloader)}")

    print("[Design C] Evaluation complete.")
    print(f"  Full metric computation and mesh export are not yet implemented in this skeleton.")
    print(f"  See DesignB/scripts/eval_full.py for a reference implementation.")


if __name__ == "__main__":
    main()
