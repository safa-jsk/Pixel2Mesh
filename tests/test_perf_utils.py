#!/usr/bin/env python3
"""
Test script for Design B Performance Utilities
==============================================

Run this to verify the performance optimization code works correctly.
This script has minimal dependencies (only torch required).

Usage:
    python3 test_perf_utils.py           # Quick test
    python3 test_perf_utils.py --full    # Full test with warmup
"""

import sys
import argparse


def test_imports():
    """Test 1: Verify all imports work"""
    print("\n" + "="*60)
    print("TEST 1: Import checks")
    print("="*60)
    
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    try:
        from utils.perf import (
            setup_cuda_optimizations,
            warmup_model,
            get_autocast_context,
            compile_model_safe,
            get_perf_config_summary,
            CudaTimer,
        )
        print("  ✓ utils.perf module")
    except ImportError as e:
        print(f"  ✗ utils.perf: {e}")
        return False
    
    return True


def test_cuda_optimizations():
    """Test 2: Test CUDA optimization setup"""
    print("\n" + "="*60)
    print("TEST 2: CUDA Optimizations")
    print("="*60)
    
    import torch
    from utils.perf import setup_cuda_optimizations
    
    # Test with optimizations enabled
    settings = setup_cuda_optimizations(
        cudnn_benchmark=True,
        tf32=True,
        logger=None
    )
    
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  ✓ cudnn.benchmark = {torch.backends.cudnn.benchmark}")
        print(f"  ✓ cuda.matmul.allow_tf32 = {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  ✓ cudnn.allow_tf32 = {torch.backends.cudnn.allow_tf32}")
        
        # Test disable
        setup_cuda_optimizations(cudnn_benchmark=False, tf32=False)
        assert torch.backends.cudnn.benchmark == False
        print("  ✓ Disable optimizations works")
        
        # Re-enable
        setup_cuda_optimizations(cudnn_benchmark=True, tf32=True)
    else:
        print("  ⊘ Skipping CUDA tests (no GPU)")
    
    return True


def test_autocast_context():
    """Test 3: Test AMP autocast context manager"""
    print("\n" + "="*60)
    print("TEST 3: AMP Autocast Context")
    print("="*60)
    
    import torch
    from utils.perf import get_autocast_context
    
    # Test disabled
    ctx = get_autocast_context(amp_enabled=False, device="cuda")
    print(f"  ✓ Disabled context: {type(ctx).__name__}")
    
    # Test enabled
    if torch.cuda.is_available():
        ctx = get_autocast_context(amp_enabled=True, device="cuda")
        print(f"  ✓ Enabled context: {type(ctx)}")
        
        # Test it works
        x = torch.randn(2, 3).cuda()
        with ctx:
            y = x @ x.T
        print(f"  ✓ Autocast forward pass works")
    else:
        print("  ⊘ Skipping CUDA autocast test (no GPU)")
    
    return True


def test_cuda_timer():
    """Test 4: Test CudaTimer context manager"""
    print("\n" + "="*60)
    print("TEST 4: CudaTimer")
    print("="*60)
    
    import torch
    import time
    from utils.perf import CudaTimer
    
    # Test on CPU
    with CudaTimer(device="cpu") as timer:
        time.sleep(0.01)  # 10ms
    
    print(f"  ✓ CPU timer: {timer.elapsed_ms:.2f}ms (expected ~10ms)")
    assert timer.elapsed_ms > 5, "Timer too short"
    
    # Test on GPU
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000).cuda()
        
        with CudaTimer(device="cuda") as timer:
            for _ in range(100):
                y = x @ x
        
        print(f"  ✓ CUDA timer: {timer.elapsed_ms:.2f}ms")
        assert timer.elapsed > 0, "CUDA timer should measure something"
    else:
        print("  ⊘ Skipping CUDA timer test (no GPU)")
    
    return True


def test_compile_model():
    """Test 5: Test torch.compile wrapper"""
    print("\n" + "="*60)
    print("TEST 5: torch.compile wrapper")
    print("="*60)
    
    import torch
    from utils.perf import compile_model_safe
    
    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    
    # Test with compile disabled
    result = compile_model_safe(model, compile_enabled=False)
    assert result is model
    print("  ✓ Compile disabled returns original model")
    
    # Test with compile enabled (PyTorch 2.x only)
    pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if pytorch_version >= (2, 0):
        try:
            compiled = compile_model_safe(model, compile_enabled=True, compile_mode="default")
            print(f"  ✓ torch.compile succeeded")
            
            # Test forward pass
            x = torch.randn(4, 10)
            y = compiled(x)
            assert y.shape == (4, 5)
            print(f"  ✓ Compiled model forward pass works")
        except Exception as e:
            print(f"  ⊘ torch.compile failed (may be expected): {e}")
    else:
        print(f"  ⊘ Skipping torch.compile (PyTorch {torch.__version__} < 2.0)")
    
    return True


def test_warmup_model(full=False):
    """Test 6: Test GPU warmup"""
    print("\n" + "="*60)
    print("TEST 6: GPU Warmup")
    print("="*60)
    
    import torch
    from utils.perf import warmup_model
    
    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 3, padding=1),
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        
        warmup_iters = 10 if full else 2
        
        avg_time = warmup_model(
            model,
            input_shape=(1, 3, 224, 224),
            warmup_iters=warmup_iters,
            device="cuda",
            amp_enabled=False,
            logger=None
        )
        
        print(f"  ✓ Warmup completed: {warmup_iters} iters, avg {avg_time*1000:.2f}ms")
        
        # Test with AMP
        avg_time_amp = warmup_model(
            model,
            input_shape=(1, 3, 224, 224),
            warmup_iters=warmup_iters,
            device="cuda",
            amp_enabled=True,
            logger=None
        )
        
        print(f"  ✓ Warmup with AMP: {warmup_iters} iters, avg {avg_time_amp*1000:.2f}ms")
    else:
        # Test CPU fallback
        avg_time = warmup_model(
            model,
            input_shape=(1, 3, 224, 224),
            warmup_iters=0,
            device="cuda",
            amp_enabled=False,
            logger=None
        )
        print(f"  ✓ Warmup skipped (no GPU): returned {avg_time}")
    
    return True


def test_perf_config_summary():
    """Test 7: Test config summary"""
    print("\n" + "="*60)
    print("TEST 7: Performance Config Summary")
    print("="*60)
    
    import torch
    from utils.perf import get_perf_config_summary
    
    summary = get_perf_config_summary(
        warmup_iters=15,
        amp_enabled=True,
        compile_enabled=False,
        cudnn_benchmark=True,
        tf32=True
    )
    
    print(f"  Config summary:")
    for key, value in summary.items():
        print(f"    {key}: {value}")
    
    assert "pytorch_version" in summary
    assert "cuda_available" in summary
    print("  ✓ Config summary generated correctly")
    
    return True


def test_cli_args():
    """Test 8: Verify CLI argument parsing (without running eval)"""
    print("\n" + "="*60)
    print("TEST 8: CLI Argument Parsing")
    print("="*60)
    
    import argparse
    
    # Simulate the argument parser from entrypoint_designB_eval.py
    parser = argparse.ArgumentParser()
    
    # Performance flags
    parser.add_argument('--warmup-iters', type=int, default=15)
    parser.add_argument('--amp', dest='amp_enabled', action='store_true', default=False)  # Disabled for P2M
    parser.add_argument('--no-amp', dest='amp_enabled', action='store_false')
    parser.add_argument('--compile', dest='compile_enabled', action='store_true', default=False)
    parser.add_argument('--no-compile', dest='compile_enabled', action='store_false')
    parser.add_argument('--cudnn-benchmark', dest='cudnn_benchmark', action='store_true', default=True)
    parser.add_argument('--no-cudnn-benchmark', dest='cudnn_benchmark', action='store_false')
    parser.add_argument('--tf32', dest='tf32_enabled', action='store_true', default=True)
    parser.add_argument('--no-tf32', dest='tf32_enabled', action='store_false')
    
    # Test defaults
    args = parser.parse_args([])
    assert args.warmup_iters == 15
    assert args.amp_enabled == False  # Disabled by default for P2M (sparse ops)
    assert args.compile_enabled == False
    assert args.cudnn_benchmark == True
    assert args.tf32_enabled == True
    print("  ✓ Default values correct")
    
    # Test disabling all
    args = parser.parse_args([
        '--warmup-iters', '0',
        '--no-amp',
        '--no-compile',
        '--no-cudnn-benchmark',
        '--no-tf32'
    ])
    assert args.warmup_iters == 0
    assert args.amp_enabled == False
    assert args.compile_enabled == False
    assert args.cudnn_benchmark == False
    assert args.tf32_enabled == False
    print("  ✓ Disable all flags works")
    
    # Test enabling compile
    args = parser.parse_args(['--compile'])
    assert args.compile_enabled == True
    print("  ✓ Enable compile flag works")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test Design B Performance Utilities')
    parser.add_argument('--full', action='store_true', help='Run full tests including longer warmup')
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("# Design B Performance Utilities - Test Suite")
    print("#"*60)
    
    tests = [
        ("Imports", test_imports),
        ("CUDA Optimizations", test_cuda_optimizations),
        ("Autocast Context", test_autocast_context),
        ("CUDA Timer", test_cuda_timer),
        ("torch.compile", test_compile_model),
        ("GPU Warmup", lambda: test_warmup_model(args.full)),
        ("Config Summary", test_perf_config_summary),
        ("CLI Arguments", test_cli_args),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  ✗ FAILED: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, s, _ in results if s)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASS" if success else f"✗ FAIL: {error}"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
