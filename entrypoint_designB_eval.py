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

Usage:
    python entrypoint_designB_eval.py --options experiments/designB_baseline.yml \
        --checkpoint datasets/data/pretrained/tensorflow.pth.tar \
        --name designB_full_eval
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from logging import Logger

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from functions.base import CheckpointRunner
from models.layers.chamfer_wrapper import ChamferDist
from models.p2m import P2MModel
from options import update_options, options, reset_options
from utils.average_meter import AverageMeter
from utils.mesh import Ellipsoid

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
    """

    def __init__(self, options, logger: Logger, writer, shared_model=None):
        super().__init__(options, logger, writer, training=False, shared_model=shared_model)
        self.sample_results = []  # Store per-sample results
        self.batch_results = []   # Store per-batch results
        self.mesh_output_dir = options.dataset.predict.folder
        os.makedirs(self.mesh_output_dir, exist_ok=True)

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
        """Evaluate a single batch with detailed logging"""
        self.model.eval()
        batch_size = input_batch['images'].size(0)
        batch_metrics = {
            'chamfer_distances': [],
            'f1_tau_scores': [],
            'f1_2tau_scores': [],
            'sample_times': [],
            'meshes_saved': []
        }

        with torch.no_grad():
            images = input_batch['images']
            
            # Time the forward pass
            torch.cuda.synchronize()
            batch_start = time.time()
            out = self.model(images)
            torch.cuda.synchronize()
            batch_inference_time = time.time() - batch_start
            
            pred_vertices = out["pred_coord"][-1]
            gt_points = input_batch["points_orig"]
            if isinstance(gt_points, list):
                gt_points = [pts.cuda() for pts in gt_points]

            # Process each sample in the batch
            for i in range(batch_size):
                sample_start = time.time()
                
                # Get sample info
                filename = input_batch["filename"][i]
                label = input_batch["labels"][i].cpu().item()
                
                # Compute metrics
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
                
                # Check if we should save mesh for this sample
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
        """Run full evaluation with comprehensive logging"""
        self.logger.info("=" * 80)
        self.logger.info("DESIGN B: Full Dataset Baseline Evaluation")
        self.logger.info("=" * 80)
        self.logger.info(f"Dataset: train_tf ({len(self.dataset)} samples)")
        self.logger.info(f"Batch size: {self.options.test.batch_size}")
        self.logger.info(f"Mesh output directory: {self.mesh_output_dir}")
        self.logger.info(f"Samples to generate meshes: 26 (2 per category)")
        self.logger.info("=" * 80)

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
        """Save evaluation summary to JSON"""
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
            }
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
    parser = argparse.ArgumentParser(description='Design B: Full Dataset Baseline Evaluation')
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', help='trained checkpoint file', type=str, required=True)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    parser.add_argument('--name', help='subfolder name of this experiment', required=True, type=str)
    parser.add_argument('--gpus', help='number of GPUs to use', type=int)
    parser.add_argument('--output-dir', help='directory to save generated meshes', type=str)

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

    evaluator = DesignBEvaluator(options, logger, writer)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
