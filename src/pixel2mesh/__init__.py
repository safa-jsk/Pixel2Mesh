"""
Pixel2Mesh: Generating 3D Mesh Vertices from Single RGB Images
================================================================

PyTorch implementation of Pixel2Mesh (Wang et al., ECCV 2018).
Restructured for thesis Model Pipeline evaluation across four designs:

- DesignA_CPU: Original CPU baseline (Docker/pinned environment)
- DesignA_GPU: GPU-enabled baseline (NVIDIA Docker runtime)
- DesignB: Optimized GPU pipeline with Model Pipeline controls
- DesignC: FaceScape integration with Data Pipeline adapter

All model math/algorithmic behavior is preserved from upstream.
"""

__version__ = "2.0.0"
