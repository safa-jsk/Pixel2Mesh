"""Pixel2Mesh Evaluator with Design B Performance Optimizations"""
from logging import Logger
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from functions.base import CheckpointRunner
from models.classifier import Classifier
from models.layers.chamfer_wrapper import ChamferDist
from models.p2m import P2MModel
from utils.average_meter import AverageMeter
from utils.mesh import Ellipsoid
from utils.vis.renderer import MeshRenderer
from utils.perf import (
    setup_cuda_optimizations,
    warmup_model,
    get_autocast_context,
    compile_model_safe,
    get_perf_config_summary,
    CudaTimer,
)


class Evaluator(CheckpointRunner):
    """
    Pixel2Mesh Evaluator with Design B Performance Optimizations.
    
    Performance Features:
    - GPU warmup iterations for stable timing
    - AMP mixed precision inference
    - torch.compile graph optimization (PyTorch 2.x)
    - cuDNN/TF32 configuration
    - CUDA-synchronized timing
    """

    def __init__(self, options, logger: Logger, writer, shared_model=None,
                 warmup_iters: int = 15,
                 amp_enabled: bool = False,  # Disabled: P2M sparse ops don't support half
                 compile_enabled: bool = False,
                 cudnn_benchmark: bool = True,
                 tf32_enabled: bool = True):
        
        # Store performance configuration before parent init
        self.warmup_iters = warmup_iters
        self.amp_enabled = amp_enabled
        self.compile_enabled = compile_enabled
        self.cudnn_benchmark = cudnn_benchmark
        self.tf32_enabled = tf32_enabled
        
        # Setup CUDA optimizations early
        setup_cuda_optimizations(
            cudnn_benchmark=cudnn_benchmark,
            tf32=tf32_enabled,
            logger=logger
        )
        
        super().__init__(options, logger, writer, training=False, shared_model=shared_model)
        
        # Apply torch.compile after model is loaded
        if self.compile_enabled:
            self.model = compile_model_safe(
                self.model,
                compile_enabled=True,
                compile_mode="max-autotune",
                logger=logger
            )

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        if self.options.model.name == "pixel2mesh":
            # Renderer for visualization
            self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
                                         self.options.dataset.mesh_pos)
            # Initialize distance module
            self.chamfer = ChamferDist()
            # create ellipsoid
            self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
            # use weighted mean evaluation metrics or not
            self.weighted_mean = self.options.test.weighted_mean
        else:
            self.renderer = None
        self.num_classes = self.options.dataset.num_classes

        if shared_model is not None:
            self.model = shared_model
        else:
            if self.options.model.name == "pixel2mesh":
                # create model
                self.model = P2MModel(self.options.model, self.ellipsoid,
                                      self.options.dataset.camera_f, self.options.dataset.camera_c,
                                      self.options.dataset.mesh_pos)
            elif self.options.model.name == "classifier":
                self.model = Classifier(self.options.model, self.options.dataset.num_classes)
            else:
                raise NotImplementedError("Your model is not found")
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

        # Evaluate step count, useful in summary
        self.evaluate_step_count = 0
        self.total_step_count = 0

    def models_dict(self):
        return {'model': self.model}

    def evaluate_f1(self, dis_to_pred, dis_to_gt, pred_length, gt_length, thresh):
        recall = np.sum(dis_to_gt < thresh) / gt_length
        prec = np.sum(dis_to_pred < thresh) / pred_length
        return 2 * prec * recall / (prec + recall + 1e-8)

    def evaluate_chamfer_and_f1(self, pred_vertices, gt_points, labels):
        # calculate accurate chamfer distance; ground truth points with different lengths;
        # therefore cannot be batched
        batch_size = pred_vertices.size(0)
        pred_length = pred_vertices.size(1)
        for i in range(batch_size):
            gt_length = gt_points[i].size(0)
            label = labels[i].cpu().item()
            d1, d2, i1, i2 = self.chamfer(pred_vertices[i].unsqueeze(0), gt_points[i].unsqueeze(0))
            d1, d2 = d1.cpu().numpy(), d2.cpu().numpy()  # convert to millimeter
            self.chamfer_distance[label].update(np.mean(d1) + np.mean(d2))
            self.f1_tau[label].update(self.evaluate_f1(d1, d2, pred_length, gt_length, 1E-4))
            self.f1_2tau[label].update(self.evaluate_f1(d1, d2, pred_length, gt_length, 2E-4))

    def evaluate_accuracy(self, output, target):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        top_k = [1, 5]
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(1.0 / batch_size)
            if k == 1:
                self.acc_1.update(acc)
            elif k == 5:
                self.acc_5.update(acc)

    def evaluate_step(self, input_batch):
        """
        Run single evaluation step with Design B performance optimizations.
        
        Uses torch.inference_mode() and optional AMP autocast for maximum
        inference performance. CUDA synchronization ensures accurate timing.
        """
        self.model.eval()

        # Run inference with inference_mode for maximum performance
        with torch.inference_mode():
            # Get ground truth
            images = input_batch['images']
            
            # Get autocast context based on AMP setting
            autocast_ctx = get_autocast_context(self.amp_enabled, "cuda")

            # Time the forward pass with CUDA synchronization
            with CudaTimer() as timer:
                with autocast_ctx:
                    out = self.model(images)
            
            self.inference_time.update(timer.elapsed, images.size(0))

            if self.options.model.name == "pixel2mesh":
                # Ensure FP32 for metric computation
                pred_vertices = out["pred_coord"][-1].float()
                gt_points = input_batch["points_orig"]
                if isinstance(gt_points, list):
                    gt_points = [pts.cuda().float() for pts in gt_points]
                self.evaluate_chamfer_and_f1(pred_vertices, gt_points, input_batch["labels"])
            elif self.options.model.name == "classifier":
                self.evaluate_accuracy(out, input_batch["labels"])

        return out

    # noinspection PyAttributeOutsideInit
    def evaluate(self):
        """
        Run full evaluation with Design B performance optimizations.
        
        Includes GPU warmup, performance configuration logging, and
        CUDA-correct timing throughout.
        """
        self.logger.info("Running evaluations...")
        
        # Log performance configuration
        perf_config = get_perf_config_summary(
            self.warmup_iters,
            self.amp_enabled,
            self.compile_enabled,
            self.cudnn_benchmark,
            self.tf32_enabled
        )
        self.logger.info("Performance configuration:")
        for key, value in perf_config.items():
            self.logger.info(f"  {key}: {value}")

        # clear evaluate_step_count, but keep total count uncleared
        self.evaluate_step_count = 0

        test_data_loader = DataLoader(self.dataset,
                                      batch_size=self.options.test.batch_size * self.options.num_gpus,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=self.options.test.shuffle,
                                      collate_fn=self.dataset_collate_fn)

        if self.options.model.name == "pixel2mesh":
            self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_tau = [AverageMeter() for _ in range(self.num_classes)]
            self.f1_2tau = [AverageMeter() for _ in range(self.num_classes)]
        elif self.options.model.name == "classifier":
            self.acc_1 = AverageMeter()
            self.acc_5 = AverageMeter()
        
        # Timing metrics
        self.inference_time = AverageMeter()  # Time for forward pass only
        self.batch_time = AverageMeter()  # Time for entire batch processing
        
        # ===== GPU WARMUP (Design B Performance) =====
        if self.warmup_iters > 0 and torch.cuda.is_available():
            # Get input shape from first batch
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
            
            # Recreate dataloader iterator
            test_data_loader = DataLoader(self.dataset,
                                          batch_size=self.options.test.batch_size * self.options.num_gpus,
                                          num_workers=self.options.num_workers,
                                          pin_memory=self.options.pin_memory,
                                          shuffle=self.options.test.shuffle,
                                          collate_fn=self.dataset_collate_fn)
        
        self.eval_start_time = time.time()

        # Iterate over all batches in an epoch
        batch_start = time.time()
        for step, batch in enumerate(test_data_loader):
            # Send input to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Run evaluation step
            out = self.evaluate_step(batch)
            
            # Record batch processing time
            batch_end = time.time()
            self.batch_time.update(batch_end - batch_start, batch['images'].size(0))

            # Tensorboard logging every summary_steps steps
            if self.evaluate_step_count % self.options.test.summary_steps == 0:
                self.evaluate_summaries(batch, out)

            # add later to log at step 0
            self.evaluate_step_count += 1
            self.total_step_count += 1
            batch_start = time.time()

        # Log timing information
        eval_total_time = time.time() - self.eval_start_time
        self.logger.info("="*60)
        self.logger.info("TIMING RESULTS:")
        self.logger.info("Total evaluation time: %.2f seconds (%.2f minutes)" % (eval_total_time, eval_total_time/60))
        self.logger.info("Average inference time per image: %.4f seconds (%.2f ms)" % (self.inference_time.avg, self.inference_time.avg*1000))
        self.logger.info("Average batch processing time: %.4f seconds" % self.batch_time.avg)
        self.logger.info("Throughput: %.2f images/second" % (1.0 / self.inference_time.avg if self.inference_time.avg > 0 else 0))
        self.logger.info("="*60)
        
        for key, val in self.get_result_summary().items():
            scalar = val
            if isinstance(val, AverageMeter):
                scalar = val.avg
            self.logger.info("Test [%06d] %s: %.6f" % (self.total_step_count, key, scalar))
            self.summary_writer.add_scalar("eval_" + key, scalar, self.total_step_count + 1)
        
        # Add timing to tensorboard
        self.summary_writer.add_scalar("eval_inference_time_ms", self.inference_time.avg*1000, self.total_step_count + 1)
        self.summary_writer.add_scalar("eval_throughput_imgs_per_sec", 1.0 / self.inference_time.avg if self.inference_time.avg > 0 else 0, self.total_step_count + 1)

    def average_of_average_meters(self, average_meters):
        s = sum([meter.sum for meter in average_meters])
        c = sum([meter.count for meter in average_meters])
        weighted_avg = s / c if c > 0 else 0.
        avg = sum([meter.avg for meter in average_meters]) / len(average_meters)
        ret = AverageMeter()
        if self.weighted_mean:
            ret.val, ret.avg = avg, weighted_avg
        else:
            ret.val, ret.avg = weighted_avg, avg
        return ret

    def get_result_summary(self):
        if self.options.model.name == "pixel2mesh":
            return {
                "cd": self.average_of_average_meters(self.chamfer_distance),
                "f1_tau": self.average_of_average_meters(self.f1_tau),
                "f1_2tau": self.average_of_average_meters(self.f1_2tau),
            }
        elif self.options.model.name == "classifier":
            return {
                "acc_1": self.acc_1,
                "acc_5": self.acc_5,
            }

    def evaluate_summaries(self, input_batch, out_summary):
        self.logger.info("Test Step %06d/%06d (%06d) " % (self.evaluate_step_count,
                                                          len(self.dataset) // (
                                                                  self.options.num_gpus * self.options.test.batch_size),
                                                          self.total_step_count,) \
                         + ", ".join([key + " " + (str(val) if isinstance(val, AverageMeter) else "%.6f" % val)
                                      for key, val in self.get_result_summary().items()]))

        self.summary_writer.add_histogram("eval_labels", input_batch["labels"].cpu().numpy(),
                                          self.total_step_count)
        if self.renderer is not None:
            # Do visualization for the first 2 images of the batch
            render_mesh = self.renderer.p2m_batch_visualize(input_batch, out_summary, self.ellipsoid.faces)
            self.summary_writer.add_image("eval_render_mesh", render_mesh, self.total_step_count)
