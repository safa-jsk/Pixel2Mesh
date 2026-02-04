"""
Design B Performance Optimization Utilities
============================================

This module provides utilities for GPU performance optimization following
Design B methodology for Pixel2Mesh evaluation.

Features:
- GPU warmup iterations to avoid cold-start timing artifacts
- AMP (Automatic Mixed Precision) for faster inference
- torch.compile support for PyTorch 2.x graph optimization
- cuDNN/TF32 flag configuration for optimal GPU performance
- CUDA-correct timing with synchronization barriers

Usage:
    from utils.perf import (
        setup_cuda_optimizations,
        warmup_model,
        get_autocast_context,
        compile_model_safe
    )
"""

import logging
import time
import torch
from contextlib import nullcontext


def setup_cuda_optimizations(
    cudnn_benchmark: bool = True,
    tf32: bool = True,
    logger: logging.Logger = None
):
    """
    Configure cuDNN and TF32 settings for optimal GPU performance.
    
    Design B Methodology Note:
    - cudnn.benchmark=True enables cuDNN autotuner for fixed input sizes,
      selecting the fastest convolution algorithms. Disable if input sizes vary.
    - TF32 (TensorFloat-32) allows tensor cores to use reduced precision
      for faster computation with minimal accuracy loss on Ampere+ GPUs.
    
    Args:
        cudnn_benchmark: Enable cuDNN benchmark mode (best for fixed input shapes)
        tf32: Enable TF32 tensor core math (Ampere+ GPUs only)
        logger: Optional logger for status messages
    
    Returns:
        dict: Applied settings for logging purposes
    """
    settings = {}
    
    if torch.cuda.is_available():
        # cuDNN benchmark mode - autotuner for fastest conv algorithms
        torch.backends.cudnn.benchmark = cudnn_benchmark
        settings['cudnn_benchmark'] = cudnn_benchmark
        
        # TF32 settings for Ampere+ GPUs (compute capability >= 8.0)
        # These have no effect on older GPUs but are safe to set
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        settings['matmul_tf32'] = tf32
        settings['cudnn_tf32'] = tf32
        
        if logger:
            logger.info(f"CUDA optimizations: cudnn.benchmark={cudnn_benchmark}, TF32={tf32}")
            if tf32:
                # Check if GPU supports TF32
                device_cap = torch.cuda.get_device_capability()
                if device_cap[0] >= 8:
                    logger.info(f"  GPU supports TF32 (compute capability {device_cap[0]}.{device_cap[1]})")
                else:
                    logger.info(f"  Note: GPU compute capability {device_cap[0]}.{device_cap[1]} < 8.0, TF32 has no effect")
    else:
        if logger:
            logger.info("CUDA not available, skipping GPU optimizations")
    
    return settings


def warmup_model(
    model: torch.nn.Module,
    input_shape: tuple,
    warmup_iters: int = 15,
    device: str = "cuda",
    amp_enabled: bool = False,
    logger: logging.Logger = None
):
    """
    Perform GPU warmup iterations to eliminate cold-start timing artifacts.
    
    Design B Methodology Note:
    Warmup is essential for accurate GPU timing because:
    1. CUDA context initialization on first kernel launch
    2. cuDNN autotuner runs on first forward pass (if benchmark=True)
    3. JIT compilation of fused kernels
    4. Memory allocation overhead on first use
    
    After warmup, subsequent forward passes have stable, representative timing.
    
    Args:
        model: The PyTorch model to warm up
        input_shape: Tuple of (batch_size, channels, height, width) for input
        warmup_iters: Number of warmup iterations (default: 15)
        device: Device to run on ("cuda" or "cpu")
        amp_enabled: Whether to use AMP during warmup (should match eval)
        logger: Optional logger for status messages
    
    Returns:
        float: Average warmup iteration time (for diagnostic purposes)
    """
    if warmup_iters <= 0:
        if logger:
            logger.info("Warmup disabled (warmup_iters=0)")
        return 0.0
    
    if not torch.cuda.is_available() and device == "cuda":
        if logger:
            logger.info("CUDA not available, skipping warmup")
        return 0.0
    
    if logger:
        logger.info(f"Running {warmup_iters} GPU warmup iterations...")
    
    model.eval()
    
    # Create representative dummy input
    dummy_input = torch.randn(input_shape, device=device)
    
    warmup_times = []
    
    with torch.inference_mode():
        autocast_ctx = get_autocast_context(amp_enabled, device)
        
        for i in range(warmup_iters):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            with autocast_ctx:
                _ = model(dummy_input)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            warmup_times.append(elapsed)
    
    avg_time = sum(warmup_times) / len(warmup_times)
    
    if logger:
        logger.info(f"  Warmup complete. Avg iteration: {avg_time*1000:.2f}ms")
        logger.info(f"  First iter: {warmup_times[0]*1000:.2f}ms, Last iter: {warmup_times[-1]*1000:.2f}ms")
    
    # Clear any cached memory from warmup
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return avg_time


def get_autocast_context(amp_enabled: bool, device: str = "cuda"):
    """
    Get the appropriate autocast context manager for AMP.
    
    Design B Methodology Note:
    AMP (Automatic Mixed Precision) uses FP16/BF16 for faster computation
    while maintaining FP32 for numerically sensitive operations.
    For inference-only, we use autocast without GradScaler.
    
    Note: Some sparse operations (like addmm_sparse_cuda) don't support BF16,
    so we default to FP16 for broader compatibility.
    
    Args:
        amp_enabled: Whether AMP is enabled
        device: Device type ("cuda" or "cpu")
    
    Returns:
        Context manager for autocast (or nullcontext if disabled)
    """
    if not amp_enabled:
        return nullcontext()
    
    if device == "cuda" and torch.cuda.is_available():
        # Use float16 for broader compatibility (sparse ops don't support bf16)
        # BFloat16 has better range but FP16 has wider hardware/op support
        dtype = torch.float16
        # Use the new torch.amp.autocast API (PyTorch 2.x compatible)
        return torch.amp.autocast(device_type='cuda', enabled=True, dtype=dtype)
    else:
        # CPU autocast (PyTorch 2.0+)
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            return torch.amp.autocast(device_type='cpu', enabled=amp_enabled, dtype=torch.bfloat16)
        return nullcontext()


def compile_model_safe(
    model: torch.nn.Module,
    compile_enabled: bool = False,
    compile_mode: str = "max-autotune",
    logger: logging.Logger = None
):
    """
    Safely compile model with torch.compile (PyTorch 2.x).
    
    Design B Methodology Note:
    torch.compile provides graph-mode optimization via Dynamo+Inductor,
    which can significantly speed up inference through:
    - Kernel fusion
    - Memory planning
    - Operator-level optimizations
    
    Falls back gracefully if compilation fails or PyTorch < 2.0.
    
    Args:
        model: PyTorch model to compile
        compile_enabled: Whether to attempt compilation
        compile_mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
        logger: Optional logger for status messages
    
    Returns:
        Compiled model (or original model if compilation disabled/failed)
    """
    if not compile_enabled:
        if logger:
            logger.info("torch.compile disabled")
        return model
    
    # Check PyTorch version
    pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if pytorch_version < (2, 0):
        if logger:
            logger.warning(f"torch.compile requires PyTorch >= 2.0 (found {torch.__version__}), skipping")
        return model
    
    try:
        if logger:
            logger.info(f"Compiling model with torch.compile(mode='{compile_mode}')...")
        
        # For DataParallel models, compile the underlying module
        if isinstance(model, torch.nn.DataParallel):
            model.module = torch.compile(model.module, mode=compile_mode)
            if logger:
                logger.info("  Compiled DataParallel.module successfully")
        else:
            model = torch.compile(model, mode=compile_mode)
            if logger:
                logger.info("  Compiled model successfully")
        
        return model
    
    except Exception as e:
        if logger:
            logger.warning(f"torch.compile failed: {e}")
            logger.warning("  Continuing with uncompiled model (eager mode)")
        return model


def cuda_sync_time():
    """
    Get current time with CUDA synchronization for accurate GPU timing.
    
    Design B Methodology Note:
    CUDA operations are asynchronous. Without synchronization, timing
    measurements will be incorrect (measuring kernel launch, not execution).
    
    Returns:
        float: Current time after GPU synchronization
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class CudaTimer:
    """
    Context manager for CUDA-correct timing with synchronization barriers.
    
    Design B Methodology Note:
    This ensures accurate GPU timing by:
    1. Synchronizing before starting the timer (clear pending ops)
    2. Synchronizing after the timed region (wait for completion)
    
    Usage:
        with CudaTimer() as timer:
            output = model(input)
        print(f"Inference took {timer.elapsed_ms:.2f}ms")
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
    
    @property
    def elapsed_ms(self):
        return self.elapsed * 1000


def get_perf_config_summary(
    warmup_iters: int,
    amp_enabled: bool,
    compile_enabled: bool,
    cudnn_benchmark: bool,
    tf32: bool
) -> dict:
    """
    Get a summary of performance configuration for logging/documentation.
    
    Returns:
        dict: Performance configuration summary
    """
    return {
        "warmup_iterations": warmup_iters,
        "amp_enabled": amp_enabled,
        "torch_compile": compile_enabled,
        "cudnn_benchmark": cudnn_benchmark,
        "tf32_enabled": tf32,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
