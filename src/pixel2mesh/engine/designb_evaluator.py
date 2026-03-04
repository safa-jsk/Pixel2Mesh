"""
Design B: Full Dataset Baseline Evaluation with Mesh Generation
================================================================

This script evaluates Pixel2Mesh on the full train_tf dataset (43,784 samples)
and generates meshes for 26 specific samples (2 per category).

Metrics logged per sample/batch:
- Chamfer Distance
- F1-Score @ tau (1e-4)
- F1-Score @ 2*tau (2e-4)
- Timing (per sample, per batch, total)

## Design B Performance Methodology

This script implements the following GPU performance optimizations:

1. **GPU Warmup** (--warmup-iters): Run warmup iterations before timing to avoid
   cold-start artifacts from CUDA context init, cuDNN autotuner, and JIT compilation.

2. **AMP Mixed Precision** (--amp): Use torch.cuda.amp.autocast for faster FP16/BF16
   inference on supported GPUs. No GradScaler needed for inference-only.

3. **torch.compile** (--compile): PyTorch 2.x graph optimization via Dynamo+Inductor
   for kernel fusion and operator optimization.

4. **cuDNN Benchmark** (--cudnn-benchmark): Enable cuDNN autotuner for optimal
   convolution algorithms with fixed input sizes.

5. **TF32 Tensor Cores** (--tf32): Enable TensorFloat-32 on Ampere+ GPUs for faster
   matrix operations with minimal accuracy impact.

6. **Inference Mode**: Uses torch.inference_mode() for maximum inference performance.

7. **CUDA-Correct Timing**: torch.cuda.synchronize() at timing boundaries ensures
   accurate measurement of asynchronous GPU operations.

Usage:
    python entrypoint_designB_eval.py --options experiments/designB_baseline.yml \
        --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
        --name designB_full_eval

    # With performance optimizations:
    python entrypoint_designB_eval.py --options experiments/designB_baseline.yml \
        --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
        --name designB_full_eval_optimized \
        --warmup-iters 20 --amp --compile --cudnn-benchmark --tf32

    # Disable all optimizations for reproducibility comparison:
    python entrypoint_designB_eval.py --options experiments/designB_baseline.yml \
        --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
        --name designB_full_eval_baseline \
        --warmup-iters 0 --no-amp --no-compile --no-cudnn-benchmark --no-tf32
"""

import argparse
import csv
import json
import os
import sys
import time

# Bootstrap: ensure src/ is on sys.path when run directly
_src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from contextlib import nullcontext
from datetime import datetime
from logging import Logger

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pixel2mesh.engine.base import CheckpointRunner
from pixel2mesh.models.layers.chamfer_wrapper import ChamferDist
from pixel2mesh.models.p2m import P2MModel
from pixel2mesh.options import update_options, options, reset_options
from pixel2mesh.utils.average_meter import AverageMeter
from pixel2mesh.utils.mesh import Ellipsoid
from pixel2mesh.utils.perf import (
    setup_cuda_optimizations,
    warmup_model,
    get_autocast_context,
    compile_model_safe,
    get_perf_config_summary,
    CudaTimer,
)

# MeshRenderer is optional (requires neural_renderer)
try:
    from utils.vis.renderer import MeshRenderer
    HAS_RENDERER = True
except ImportError:
    HAS_RENDERER = False
    MeshRenderer = None


# 26 samples from Design A (2 per category)
DESIGN_A_SAMPLES = {
    "02691156": ["1b171503", "1954754c"],  # Airplane
    "02828884": ["715445f1", "84aa9117"],  # Bench
    "02933112": ["14c527e2", "4b80db7a"],  # Cabinet
    "02958343": ["3b56b3bd", "5cc5d027"],  # Car
    "03001627": ["c7953284", "854f3cc9"],  # Chair
    "03211117": ["3351a012", "d9b7d9a4"],  # Display
    "03636649": ["e6b34319", "cef0caa6"],  # Lamp
    "03691459": ["6fcb50de", "26778511"],  # Loudspeaker
    "04090263": ["8aff17e0", "3af4f08a"],  # Rifle
    "04256520": ["82495323", "f0808072"],  # Sofa
    "04379243": ["ea9e7db4", "38e83df8"],  # Table
    "04401088": ["f2245c0f", "fb1e1826"],  # Telephone
    "04530566": ["573c6998", "8fdc3288"],  # Watercraft
}

CATEGORY_NAMES = {
    "02691156": "Airplane",
    "02828884": "Bench",
    "02933112": "Cabinet",
    "02958343": "Car",
    "03001627": "Chair",
    "03211117": "Display",
    "03636649": "Lamp",
    "03691459": "Loudspeaker",
    "04090263": "Rifle",
    "04256520": "Sofa",
    "04379243": "Table",
    "04401088": "Telephone",
    "04530566": "Watercraft",
}


class DesignBEvaluator(CheckpointRunner):
    """
    Design B Evaluator: Full dataset evaluation with comprehensive logging
    
    Performance Features (Design B Methodology):
    - GPU warmup iterations for stable timing
    - AMP mixed precision inference
    - torch.compile graph optimization
    - cuDNN/TF32 configuration
    - CUDA-synchronized timing
    """

    # [DESIGN.B][CAMFM.A5_METHOD] Reproducible evaluation configuration with explicit performance flags
    def __init__(self, options, logger: Logger, writer, shared_model=None,
                 warmup_iters: int = 15,
                 amp_enabled: bool = False,  # Disabled: P2M sparse ops don't support half
                 compile_enabled: bool = False,
                 cudnn_benchmark: bool = True,
                 tf32_enabled: bool = True):
        
        # Store performance configuration before parent init (which calls init_fn)
        self.warmup_iters = warmup_iters
        self.amp_enabled = amp_enabled
        self.compile_enabled = compile_enabled
        self.cudnn_benchmark = cudnn_benchmark
        self.tf32_enabled = tf32_enabled
        
        # [DESIGN.B][CAMFM.A2d_OPTIONAL_ACCEL] Setup CUDA optimizations early (before model creation)
        setup_cuda_optimizations(
            cudnn_benchmark=cudnn_benchmark,
            tf32=tf32_enabled,
            logger=logger
        )
        
        super().__init__(options, logger, writer, training=False, shared_model=shared_model)
        
        self.sample_results = []  # Store per-sample results
        self.batch_results = []   # Store per-batch results
        self.mesh_output_dir = options.dataset.predict.folder
        os.makedirs(self.mesh_output_dir, exist_ok=True)
        
        # Apply torch.compile after model is loaded (in init_fn via parent __init__)
        if self.compile_enabled:
            self.model = compile_model_safe(
                self.model,
                compile_enabled=True,
                compile_mode="max-autotune",
                logger=logger
            )

    def init_fn(self, shared_model=None, **kwargs):
        # Renderer for visualization (optional)
        if HAS_RENDERER:
            self.renderer = MeshRenderer(
                self.options.dataset.camera_f,
                self.options.dataset.camera_c,
                self.options.dataset.mesh_pos
            )
        else:
            self.renderer = None
            self.logger.info("MeshRenderer not available (neural_renderer not installed)")
        # Initialize distance module
        self.chamfer = ChamferDist()
        # Create ellipsoid
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
        self.num_classes = self.options.dataset.num_classes

        if shared_model is not None:
            self.model = shared_model
        else:
            # [DESIGN.B][CAMFM.A2a_GPU_RESIDENCY] Model fully on GPU (no CPU fallbacks)
            self.model = P2MModel(
                self.options.model,
                self.ellipsoid,
                self.options.dataset.camera_f,
                self.options.dataset.camera_c,
                self.options.dataset.mesh_pos
            )
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

        self.evaluate_step_count = 0
        self.total_step_count = 0

        # Build list of samples to generate meshes for
        self.samples_to_save = set()
        for cat_id, obj_ids in DESIGN_A_SAMPLES.items():
            for obj_id in obj_ids:
                self.samples_to_save.add((cat_id, obj_id))

    def models_dict(self):
        return {'model': self.model}

    def evaluate_f1(self, dis_to_pred, dis_to_gt, pred_length, gt_length, thresh):
        recall = np.sum(dis_to_gt < thresh) / gt_length
        prec = np.sum(dis_to_pred < thresh) / pred_length
        return 2 * prec * recall / (prec + recall + 1e-8)

    def compute_sample_metrics(self, pred_vertices, gt_points, label):
        """Compute chamfer distance and F1 scores for a single sample"""
        pred_length = pred_vertices.size(0)
        gt_length = gt_points.size(0)
        
        d1, d2, i1, i2 = self.chamfer(pred_vertices.unsqueeze(0), gt_points.unsqueeze(0))
        d1, d2 = d1.cpu().numpy(), d2.cpu().numpy()
        
        chamfer_dist = np.mean(d1) + np.mean(d2)
        f1_tau = self.evaluate_f1(d1, d2, pred_length, gt_length, 1e-4)
        f1_2tau = self.evaluate_f1(d1, d2, pred_length, gt_length, 2e-4)
        
        return chamfer_dist, f1_tau, f1_2tau

    def should_save_mesh(self, filename):
        """Check if this sample is in the 26 samples to save"""
        # filename format: category_id/object_id/rendering/XX.dat
        parts = filename.split("/")
        if len(parts) >= 2:
            cat_id = parts[0]
            obj_id = parts[1]
            # Check if obj_id starts with any of our target IDs
            for target_cat, target_objs in DESIGN_A_SAMPLES.items():
                if cat_id == target_cat:
                    for target_obj in target_objs:
                        if obj_id.startswith(target_obj):
                            return True, cat_id, obj_id
        return False, None, None

    def save_mesh(self, vertices, cat_id, obj_id, stage):
        """Save mesh to OBJ file"""
        cat_name = CATEGORY_NAMES.get(cat_id, cat_id)
        filename = f"{cat_name.lower()}_{obj_id}.{stage}.obj"
        filepath = os.path.join(self.mesh_output_dir, filename)
        
        verts = vertices.cpu().numpy()
        vert_v = np.hstack((np.full([verts.shape[0], 1], "v"), verts))
        mesh = np.vstack((vert_v, self.ellipsoid.obj_fmt_faces[stage - 1]))
        np.savetxt(filepath, mesh, fmt='%s', delimiter=" ")
        
        return filepath

    def evaluate_step(self, input_batch, batch_idx):
        """
        Evaluate a single batch with detailed logging.
        
        Design B Performance Notes:
        - Uses torch.inference_mode() for maximum inference performance
        - AMP autocast for FP16/BF16 forward pass (if enabled)
        - CUDA synchronization at timing boundaries for accurate measurement
        - Chamfer/F1 metrics computed outside autocast to ensure FP32 precision
        """
        self.model.eval()
        batch_size = input_batch['images'].size(0)
        batch_metrics = {
            'chamfer_distances': [],
            'f1_tau_scores': [],
            'f1_2tau_scores': [],
            'sample_times': [],
            'meshes_saved': []
        }

        # Use inference_mode for maximum performance (disables autograd entirely)
        with torch.inference_mode():
            images = input_batch['images']
            
            # Get autocast context based on AMP setting
            # Note: Keep metric computation outside autocast to ensure FP32 precision
            autocast_ctx = get_autocast_context(self.amp_enabled, "cuda")
            
            # [DESIGN.B][CAMFM.A2b_STEADY_STATE] Time the forward pass with CUDA synchronization
            # Design B: Sync before and after to measure actual GPU execution time
            with CudaTimer() as timer:
                with autocast_ctx:
                    out = self.model(images)
            
            batch_inference_time = timer.elapsed
            
            # [DESIGN.B][CAMFM.A2c_MEM_LAYOUT] Ensure output is in FP32 for metric computation
            # (autocast may produce FP16 outputs on some layers)
            pred_vertices = out["pred_coord"][-1].float()
            gt_points = input_batch["points_orig"]
            if isinstance(gt_points, list):
                gt_points = [pts.cuda().float() for pts in gt_points]

            # Process each sample in the batch
            for i in range(batch_size):
                sample_start = time.time()
                
                # Get sample info
                filename = input_batch["filename"][i]
                label = input_batch["labels"][i].cpu().item()
                
                # [DESIGN.B][CAMFM.A3_METRICS] Compute metrics (chamfer, F1)
                cd, f1_tau, f1_2tau = self.compute_sample_metrics(
                    pred_vertices[i],
                    gt_points[i],
                    label
                )
                
                sample_time = time.time() - sample_start + (batch_inference_time / batch_size)
                
                # Store metrics
                batch_metrics['chamfer_distances'].append(cd)
                batch_metrics['f1_tau_scores'].append(f1_tau)
                batch_metrics['f1_2tau_scores'].append(f1_2tau)
                batch_metrics['sample_times'].append(sample_time)
                
                # Update per-class accumulators
                self.chamfer_distance[label].update(cd)
                self.f1_tau[label].update(f1_tau)
                self.f1_2tau[label].update(f1_2tau)
                
                # [DESIGN.B][CAMFM.A3_METRICS] Check if we should save mesh for this sample (26 samples)
                should_save, cat_id, obj_id = self.should_save_mesh(filename)
                if should_save:
                    # Save all 3 stages
                    for stage in range(1, 4):
                        stage_verts = out["pred_coord"][stage - 1][i]
                        mesh_path = self.save_mesh(stage_verts, cat_id, obj_id, stage)
                        batch_metrics['meshes_saved'].append(mesh_path)
                    self.logger.info(f"  -> Saved mesh for {CATEGORY_NAMES.get(cat_id, cat_id)}/{obj_id}")
                
                # Store sample result
                self.sample_results.append({
                    'sample_idx': self.evaluate_step_count * batch_size + i,
                    'filename': filename,
                    'category': label,
                    'chamfer_distance': cd,
                    'f1_tau': f1_tau,
                    'f1_2tau': f1_2tau,
                    'time_seconds': sample_time
                })

        return batch_metrics, batch_inference_time

    def evaluate(self):
        """
        Run full evaluation with comprehensive logging.
        
        Design B Performance Methodology:
        1. Log performance configuration for reproducibility
        2. Run GPU warmup iterations before timing
        3. Use inference_mode() throughout evaluation
        4. Apply AMP autocast during forward passes
        5. CUDA-synchronize at all timing boundaries
        """
        self.logger.info("=" * 80)
        self.logger.info("DESIGN B: Full Dataset Baseline Evaluation")
        self.logger.info("=" * 80)
        self.logger.info(f"Dataset: train_tf ({len(self.dataset)} samples)")
        self.logger.info(f"Batch size: {self.options.test.batch_size}")
        self.logger.info(f"Mesh output directory: {self.mesh_output_dir}")
        self.logger.info(f"Samples to generate meshes: 26 (2 per category)")
        self.logger.info("=" * 80)
        
        # Log performance configuration (Design B)
        perf_config = get_perf_config_summary(
            self.warmup_iters,
            self.amp_enabled,
            self.compile_enabled,
            self.cudnn_benchmark,
            self.tf32_enabled
        )
        self.logger.info("PERFORMANCE CONFIGURATION:")
        for key, value in perf_config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("-" * 80)

        # Initialize accumulators
        self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes)]
        self.f1_tau = [AverageMeter() for _ in range(self.num_classes)]
        self.f1_2tau = [AverageMeter() for _ in range(self.num_classes)]
        self.inference_time = AverageMeter()
        self.batch_time = AverageMeter()

        test_data_loader = DataLoader(
            self.dataset,
            batch_size=self.options.test.batch_size * self.options.num_gpus,
            num_workers=self.options.num_workers,
            pin_memory=self.options.pin_memory,
            shuffle=False,
            collate_fn=self.dataset_collate_fn
        )

        total_batches = len(test_data_loader)
        
        # ===== GPU WARMUP (Design B Performance) =====
        # Run warmup iterations to eliminate cold-start timing artifacts
        # This ensures cuDNN autotuner has run and CUDA context is initialized
        if self.warmup_iters > 0 and torch.cuda.is_available():
            # Determine input shape from dataset
            sample_batch = next(iter(test_data_loader))
            input_shape = sample_batch['images'].shape
            self.logger.info(f"Input shape for warmup: {input_shape}")
            
            warmup_model(
                self.model,
                input_shape=input_shape,
                warmup_iters=self.warmup_iters,
                device="cuda",
                amp_enabled=self.amp_enabled,
                logger=self.logger
            )
            
            # Recreate dataloader iterator after consuming one batch for warmup
            test_data_loader = DataLoader(
                self.dataset,
                batch_size=self.options.test.batch_size * self.options.num_gpus,
                num_workers=self.options.num_workers,
                pin_memory=self.options.pin_memory,
                shuffle=False,
                collate_fn=self.dataset_collate_fn
            )
        
        eval_start_time = time.time()
        meshes_saved_count = 0

        self.logger.info(f"Starting evaluation: {total_batches} batches")
        self.logger.info("-" * 80)

        for batch_idx, batch in enumerate(test_data_loader):
            batch_start = time.time()
            
            # Send to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Evaluate batch
            batch_metrics, inference_time = self.evaluate_step(batch, batch_idx)
            
            batch_end = time.time()
            batch_total_time = batch_end - batch_start
            
            # Update timing
            self.inference_time.update(inference_time, batch['images'].size(0))
            self.batch_time.update(batch_total_time, batch['images'].size(0))
            
            # Store batch result
            batch_result = {
                'batch_idx': batch_idx,
                'batch_size': len(batch_metrics['chamfer_distances']),
                'avg_chamfer_distance': np.mean(batch_metrics['chamfer_distances']),
                'avg_f1_tau': np.mean(batch_metrics['f1_tau_scores']),
                'avg_f1_2tau': np.mean(batch_metrics['f1_2tau_scores']),
                'batch_time_seconds': batch_total_time,
                'inference_time_seconds': inference_time,
                'meshes_saved': len(batch_metrics['meshes_saved'])
            }
            self.batch_results.append(batch_result)
            meshes_saved_count += len(batch_metrics['meshes_saved'])
            
            # Log progress
            if batch_idx % self.options.test.summary_steps == 0 or batch_idx == total_batches - 1:
                elapsed = time.time() - eval_start_time
                eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1)
                
                avg_cd = self.average_of_average_meters(self.chamfer_distance).avg
                avg_f1_tau = self.average_of_average_meters(self.f1_tau).avg
                avg_f1_2tau = self.average_of_average_meters(self.f1_2tau).avg
                
                self.logger.info(
                    f"Batch [{batch_idx+1:5d}/{total_batches}] "
                    f"CD: {avg_cd:.6f} | "
                    f"F1@τ: {avg_f1_tau:.4f} | "
                    f"F1@2τ: {avg_f1_2tau:.4f} | "
                    f"Time: {batch_total_time:.3f}s | "
                    f"Meshes: {meshes_saved_count}/78 | "
                    f"ETA: {eta/60:.1f}min"
                )
            
            self.evaluate_step_count += 1
            self.total_step_count += 1

        # Calculate final metrics
        total_time = time.time() - eval_start_time
        
        # Log final results
        self.logger.info("=" * 80)
        self.logger.info("EVALUATION COMPLETE")
        self.logger.info("=" * 80)
        
        # Per-category results
        self.logger.info("\nPER-CATEGORY RESULTS:")
        self.logger.info("-" * 80)
        self.logger.info(f"{'Category':<15} {'Samples':>8} {'CD':>12} {'F1@τ':>10} {'F1@2τ':>10}")
        self.logger.info("-" * 80)
        
        for label_idx, cat_id in enumerate(sorted(CATEGORY_NAMES.keys())):
            cat_name = CATEGORY_NAMES[cat_id]
            cd = self.chamfer_distance[label_idx]
            f1_t = self.f1_tau[label_idx]
            f1_2t = self.f1_2tau[label_idx]
            self.logger.info(
                f"{cat_name:<15} {cd.count:>8} {cd.avg:>12.6f} {f1_t.avg:>10.4f} {f1_2t.avg:>10.4f}"
            )
        
        # Overall results
        self.logger.info("-" * 80)
        avg_cd = self.average_of_average_meters(self.chamfer_distance)
        avg_f1_tau = self.average_of_average_meters(self.f1_tau)
        avg_f1_2tau = self.average_of_average_meters(self.f1_2tau)
        
        total_samples = sum(m.count for m in self.chamfer_distance)
        self.logger.info(
            f"{'OVERALL':<15} {total_samples:>8} {avg_cd.avg:>12.6f} {avg_f1_tau.avg:>10.4f} {avg_f1_2tau.avg:>10.4f}"
        )
        
        # Timing results
        self.logger.info("\nTIMING RESULTS:")
        self.logger.info("-" * 80)
        self.logger.info(f"Total evaluation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        self.logger.info(f"Total samples processed: {total_samples}")
        self.logger.info(f"Average time per sample: {total_time/total_samples*1000:.2f} ms")
        self.logger.info(f"Average inference time per batch: {self.inference_time.avg*1000:.2f} ms")
        self.logger.info(f"Throughput: {total_samples/total_time:.2f} samples/second")
        
        # Mesh generation results
        self.logger.info("\nMESH GENERATION:")
        self.logger.info("-" * 80)
        self.logger.info(f"Meshes saved: {meshes_saved_count} files (26 samples × 3 stages)")
        self.logger.info(f"Output directory: {self.mesh_output_dir}")
        
        # Save detailed results to CSV
        self.save_results_to_csv(total_time, total_samples)
        
        # Save summary to JSON
        self.save_summary_json(total_time, total_samples, avg_cd.avg, avg_f1_tau.avg, avg_f1_2tau.avg)
        
        self.logger.info("=" * 80)

    def average_of_average_meters(self, average_meters):
        """Compute weighted average of average meters"""
        s = sum([meter.sum for meter in average_meters])
        c = sum([meter.count for meter in average_meters])
        weighted_avg = s / c if c > 0 else 0.
        ret = AverageMeter()
        ret.avg = weighted_avg
        ret.count = c
        ret.sum = s
        return ret

    def save_results_to_csv(self, total_time, total_samples):
        """Save detailed results to CSV files"""
        # Sample-level results
        sample_csv = os.path.join(self.options.log_dir, self.options.name, "sample_results.csv")
        os.makedirs(os.path.dirname(sample_csv), exist_ok=True)
        
        with open(sample_csv, 'w', newline='') as f:
            if self.sample_results:
                writer = csv.DictWriter(f, fieldnames=self.sample_results[0].keys())
                writer.writeheader()
                writer.writerows(self.sample_results)
        self.logger.info(f"Sample results saved to: {sample_csv}")
        
        # Batch-level results
        batch_csv = os.path.join(self.options.log_dir, self.options.name, "batch_results.csv")
        with open(batch_csv, 'w', newline='') as f:
            if self.batch_results:
                writer = csv.DictWriter(f, fieldnames=self.batch_results[0].keys())
                writer.writeheader()
                writer.writerows(self.batch_results)
        self.logger.info(f"Batch results saved to: {batch_csv}")

    def save_summary_json(self, total_time, total_samples, avg_cd, avg_f1_tau, avg_f1_2tau):
        """Save evaluation summary to JSON including performance configuration"""
        summary = {
            "design": "B",
            "timestamp": datetime.now().isoformat(),
            "dataset": {
                "name": "train_tf",
                "total_samples": total_samples,
                "num_classes": self.num_classes
            },
            "metrics": {
                "chamfer_distance": avg_cd,
                "f1_tau": avg_f1_tau,
                "f1_2tau": avg_f1_2tau
            },
            "per_category": {},
            "timing": {
                "total_seconds": total_time,
                "total_minutes": total_time / 60,
                "samples_per_second": total_samples / total_time,
                "ms_per_sample": total_time / total_samples * 1000,
                "avg_batch_inference_ms": self.inference_time.avg * 1000
            },
            "mesh_generation": {
                "samples_generated": 26,
                "files_generated": 78,
                "output_directory": self.mesh_output_dir
            },
            "configuration": {
                "batch_size": self.options.test.batch_size,
                "num_gpus": self.options.num_gpus,
                "checkpoint": self.options.checkpoint
            },
            # Design B Performance Configuration
            "performance": get_perf_config_summary(
                self.warmup_iters,
                self.amp_enabled,
                self.compile_enabled,
                self.cudnn_benchmark,
                self.tf32_enabled
            )
        }
        
        # Add per-category metrics
        for label_idx, cat_id in enumerate(sorted(CATEGORY_NAMES.keys())):
            cat_name = CATEGORY_NAMES[cat_id]
            summary["per_category"][cat_name] = {
                "category_id": cat_id,
                "samples": self.chamfer_distance[label_idx].count,
                "chamfer_distance": self.chamfer_distance[label_idx].avg,
                "f1_tau": self.f1_tau[label_idx].avg,
                "f1_2tau": self.f1_2tau[label_idx].avg
            }
        
        summary_json = os.path.join(self.options.log_dir, self.options.name, "evaluation_summary.json")
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Summary saved to: {summary_json}")


def parse_args():
    """
    Parse command-line arguments for Design B evaluation.
    
    Performance Flags (Design B Methodology):
    - --warmup-iters: GPU warmup iterations before timing (default: 15)
    - --amp/--no-amp: Enable/disable AMP mixed precision (default: enabled)
    - --compile/--no-compile: Enable/disable torch.compile (default: disabled)
    - --cudnn-benchmark/--no-cudnn-benchmark: cuDNN autotuner (default: enabled)
    - --tf32/--no-tf32: TF32 tensor cores on Ampere+ (default: enabled)
    """
    parser = argparse.ArgumentParser(
        description='Design B: Full Dataset Baseline Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    # Standard evaluation arguments
    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', help='trained checkpoint file', type=str, required=True)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    parser.add_argument('--name', help='subfolder name of this experiment', required=True, type=str)
    parser.add_argument('--gpus', help='number of GPUs to use', type=int)
    parser.add_argument('--output-dir', help='directory to save generated meshes', type=str)
    parser.add_argument('--num-workers', help='number of DataLoader worker processes (0=main process only, avoids fork+CUDA conflicts)', type=int, default=0)

    # ===== DESIGN B PERFORMANCE FLAGS =====
    # These flags control GPU performance optimizations.
    # Each can be disabled for reproducibility comparisons.
    
    # GPU Warmup: Eliminates cold-start timing artifacts
    parser.add_argument(
        '--warmup-iters', 
        type=int, 
        default=15,
        help='Number of GPU warmup iterations before timing (0 to disable)'
    )
    
    # AMP Mixed Precision: FP16/BF16 for faster inference
    # NOTE: Disabled by default for Pixel2Mesh because sparse graph convolutions
    # (addmm_sparse_cuda) don't support half precision. Enable only if model is modified.
    parser.add_argument(
        '--amp', 
        dest='amp_enabled',
        action='store_true',
        default=False,
        help='Enable AMP mixed precision inference (disabled by default - sparse ops unsupported)'
    )
    parser.add_argument(
        '--no-amp', 
        dest='amp_enabled',
        action='store_false',
        help='Disable AMP mixed precision'
    )
    
    # torch.compile: PyTorch 2.x graph optimization
    parser.add_argument(
        '--compile', 
        dest='compile_enabled',
        action='store_true',
        default=False,
        help='Enable torch.compile (PyTorch 2.x, may increase first-run time)'
    )
    parser.add_argument(
        '--no-compile', 
        dest='compile_enabled',
        action='store_false',
        help='Disable torch.compile'
    )
    
    # cuDNN Benchmark: Autotuner for optimal conv algorithms
    parser.add_argument(
        '--cudnn-benchmark', 
        dest='cudnn_benchmark',
        action='store_true',
        default=True,
        help='Enable cuDNN benchmark mode (best for fixed input sizes)'
    )
    parser.add_argument(
        '--no-cudnn-benchmark', 
        dest='cudnn_benchmark',
        action='store_false',
        help='Disable cuDNN benchmark mode'
    )
    
    # TF32: TensorFloat-32 on Ampere+ GPUs
    parser.add_argument(
        '--tf32', 
        dest='tf32_enabled',
        action='store_true',
        default=True,
        help='Enable TF32 tensor core math (Ampere+ GPUs)'
    )
    parser.add_argument(
        '--no-tf32', 
        dest='tf32_enabled',
        action='store_false',
        help='Disable TF32'
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    
    # Override output directory if specified
    if args.output_dir:
        options.dataset.predict.folder = args.output_dir
    
    logger, writer = reset_options(options, args, phase='eval')

    logger.info("Design B: Full Dataset Baseline Evaluation")
    logger.info(f"Options file: {args.options}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {options.dataset.predict.folder}")
    
    # Log performance settings
    logger.info("Performance settings:")
    logger.info(f"  warmup_iters: {args.warmup_iters}")
    logger.info(f"  amp_enabled: {args.amp_enabled}")
    logger.info(f"  compile_enabled: {args.compile_enabled}")
    logger.info(f"  cudnn_benchmark: {args.cudnn_benchmark}")
    logger.info(f"  tf32_enabled: {args.tf32_enabled}")

    # Create evaluator with performance settings
    evaluator = DesignBEvaluator(
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
