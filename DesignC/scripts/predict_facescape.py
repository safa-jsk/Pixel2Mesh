#!/usr/bin/env python3
"""
Design C — Generate meshes from FaceScape images.

Usage:
    python DesignC/scripts/predict_facescape.py \
        --options configs/defaults/designA_vgg.yml \
        --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
        --facescape_root datasets/data/facescape \
        --splits_csv datasets/data/facescape/splits.csv
"""

import argparse
import os
import sys

# --- Bootstrap ---
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_src_dir = os.path.join(_repo_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    import pixel2mesh  # noqa: F401
except ImportError as e:
    print(f"ERROR: Cannot import pixel2mesh from {_src_dir}")
    print(f"  {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Design C — FaceScape mesh prediction")
    parser.add_argument("--options", type=str, required=True, help="YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--facescape_root", type=str, required=True)
    parser.add_argument("--splits_csv", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="artifacts/outputs/designC_meshes")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    if not os.path.isfile(args.options):
        print(f"ERROR: Config file not found: {args.options}")
        sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    sys.path.insert(0, os.path.join(_repo_root, "DesignC", "scripts"))
    from facescape_adapter import FaceScapeDataset

    from pixel2mesh.options import update_options, options
    update_options(args.options)

    dataset = FaceScapeDataset(
        root=args.facescape_root,
        splits_csv=args.splits_csv,
        split=args.split,
    )

    import torch
    from torch.utils.data import DataLoader

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

    os.makedirs(args.output_dir, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"[Design C] Generating meshes for {len(dataset)} images → {args.output_dir}")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch["images"].to(device)
            output = model(images)

            # Skeleton: mesh saving not yet implemented
            if i % 50 == 0:
                print(f"  Batch {i}/{len(dataloader)}")

    print("[Design C] Prediction complete (mesh saving is a skeleton — see DesignB for reference).")


if __name__ == "__main__":
    main()
