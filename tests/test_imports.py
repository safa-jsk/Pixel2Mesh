#!/usr/bin/env python3
"""
Smoke-test that all pixel2mesh subpackages are importable.

Run:
    PYTHONPATH=src python tests/test_imports.py
"""
import sys
import os
import importlib

# Ensure src/ is on the path
_src = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_src))

MODULES = [
    "pixel2mesh",
    "pixel2mesh.config",
    "pixel2mesh.options",
    "pixel2mesh.logger",
    "pixel2mesh.datasets.base_dataset",
    "pixel2mesh.datasets.shapenet",
    "pixel2mesh.datasets.imagenet",
    "pixel2mesh.models.p2m",
    "pixel2mesh.models.classifier",
    "pixel2mesh.models.backbones",
    "pixel2mesh.models.layers.gbottleneck",
    "pixel2mesh.models.layers.gconv",
    "pixel2mesh.models.layers.gpooling",
    "pixel2mesh.models.layers.gprojection",
    "pixel2mesh.losses.p2m",
    "pixel2mesh.losses.classifier",
    "pixel2mesh.engine.base",
    "pixel2mesh.engine.trainer",
    "pixel2mesh.engine.evaluator",
    "pixel2mesh.engine.predictor",
    "pixel2mesh.engine.saver",
    "pixel2mesh.utils.average_meter",
    "pixel2mesh.utils.mesh",
    "pixel2mesh.utils.tensor",
]


def main():
    passed = 0
    failed = 0
    errors = []
    for mod_name in MODULES:
        try:
            importlib.import_module(mod_name)
            passed += 1
            print(f"  OK  {mod_name}")
        except Exception as e:
            failed += 1
            errors.append((mod_name, e))
            print(f"  FAIL {mod_name}: {e}")

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(MODULES)}")
    if errors:
        print("\nFailed imports:")
        for mod_name, err in errors:
            print(f"  {mod_name}: {err}")
        sys.exit(1)
    else:
        print("All imports OK.")
        sys.exit(0)


if __name__ == "__main__":
    main()
