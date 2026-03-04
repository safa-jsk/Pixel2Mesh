#!/usr/bin/env python3
"""
FaceScape dataset adapter for Pixel2Mesh.

Provides a PyTorch Dataset that reads image/mesh pairs from the FaceScape
directory structure. Raises a clear error at *import time* if the data root
or splits CSV is missing.

Usage:
    from facescape_adapter import FaceScapeDataset
    ds = FaceScapeDataset(root="datasets/data/facescape",
                          splits_csv="datasets/data/facescape/splits.csv",
                          split="test")
"""

import csv
import os

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:
    raise ImportError(
        "PyTorch is required for the FaceScape adapter. "
        "Install it with: pip install torch"
    ) from e


class FaceScapeDataset(Dataset):
    """Thin adapter that loads FaceScape image/mesh pairs.

    Parameters
    ----------
    root : str
        Path to the FaceScape data directory (e.g. ``datasets/data/facescape``).
    splits_csv : str
        CSV file with columns ``image_path,mesh_path,split``.
    split : str
        One of ``train``, ``val``, ``test``.
    img_size : int
        Target image size (square). Default 224.
    """

    def __init__(self, root: str, splits_csv: str, split: str = "test", img_size: int = 224):
        # --- Validate paths up-front so failures are obvious ---
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"FaceScape data root not found: {root}\n"
                f"Download FaceScape from https://facescape.nju.edu.cn/ and place it under datasets/data/facescape/"
            )
        if not os.path.isfile(splits_csv):
            raise FileNotFoundError(
                f"Splits CSV not found: {splits_csv}\n"
                f"Create a CSV with columns: image_path,mesh_path,split"
            )

        self.root = root
        self.img_size = img_size
        self.samples = []

        with open(splits_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"].strip() == split:
                    self.samples.append(
                        {
                            "image": os.path.join(root, row["image_path"].strip()),
                            "mesh": os.path.join(root, row["mesh_path"].strip()),
                        }
                    )

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found for split '{split}' in {splits_csv}. "
                f"Check the CSV contents and ensure the 'split' column has matching values."
            )

        print(f"[FaceScapeDataset] Loaded {len(self.samples)} samples for split='{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img = Image.open(sample["image"]).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        # HWC -> CHW
        img = torch.from_numpy(img.transpose(2, 0, 1))

        # Mesh path (actual loading depends on pipeline needs)
        mesh_path = sample["mesh"]

        return {
            "images": img,
            "mesh_path": mesh_path,
            "filename": os.path.basename(sample["image"]),
        }
