#!/usr/bin/env python3
"""
BEVFormer Multi-GPU Distributed Training Script
Uses BEVFormerTrainer class with DistributedDataParallel for scalable training
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity, schedule

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.config_parser import load_config
from trainer import BEVFormerTrainer, setup_logging, setup_device
from trainer.utils import cleanup_distributed, is_main_process, get_rank


class DistributedBEVFormerTrainer(BEVFormerTrainer):
    """BEVFormer trainer with distributed training support."""

    def __init__(self, config, device, logger, log_dir, checkpoint_dir, rank, world_size,
                 profiler_config=None, compile_config=None, amp_config=None):
        """Initialize distributed trainer.

        Args:
            config: Training configuration
            device: Device for training
            logger: Logger instance
            log_dir: Log directory
            checkpoint_dir: Checkpoint directory
            rank: Process rank
            world_size: Total number of processes
            profiler_config: Optional profiler configuration dict
            compile_config: Optional torch.compile configuration dict
            amp_config: Optional AMP configuration dict
        """
        super().__init__(config, device, logger, log_dir, checkpoint_dir)
        self.rank = rank
        self.world_size = world_size
        self.profiler_config = profiler_config or {}
        self.compile_config = compile_config or {}
        self.amp_config = amp_config or {}
        self.profiler = None
        self.profiling_step = 0

        # Setup AMP scaler if enabled
        self.use_amp = self.amp_config.get('enabled', False)
        self.scaler = None
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            if is_main_process():
                amp_dtype = self.amp_config.get('dtype', 'float16')
                self.logger.info("=" * 80)
                self.logger.info("AUTOMATIC MIXED PRECISION ENABLED")
                self.logger.info("=" * 80)
                self.logger.info(f"AMP dtype: {amp_dtype}")
                self.logger.info("Expected: 2-3x faster training, 40-50% memory savings")
                self.logger.info("=" * 80)

        # Setup GPU augmentation (disabled by default, can be enabled)
        self.use_gpu_aug = True  # Set to True to enable GPU augmentation
        self.gpu_augmentation = None
        if self.use_gpu_aug and torch.cuda.is_available():
            try:
                from augmentation_gpu import GPUAugmentation
                self.gpu_augmentation = GPUAugmentation(training=True).to(self.device)
                if is_main_process():
                    self.logger.info("=" * 80)
                    self.logger.info("GPU AUGMENTATION ENABLED (Experimental)")
                    self.logger.info("=" * 80)
                    self.logger.info("Photometric distortion will run on GPU")
                    self.logger.info("Expected: 20-30% faster data pipeline")
                    self.logger.info("=" * 80)
            except ImportError:
                if is_main_process():
                    self.logger.warning("GPU augmentation requested but Kornia not installed")
                    self.logger.warning("Install with: pip install kornia")
                self.gpu_augmentation = None

    def setup_model(self):
        """Create and wrap model with DDP."""
        model = super().setup_model()

        # Apply torch.compile() if enabled (before DDP wrapping)
        if hasattr(self, 'compile_config') and self.compile_config.get('enabled', False):
            if is_main_process():
                self.logger.info("=" * 80)
                self.logger.info("TORCH.COMPILE ENABLED")
                self.logger.info("=" * 80)
                self.logger.info(f"Compile Mode: {self.compile_config.get('mode', 'reduce-overhead')}")
                self.logger.info(f"Compile Backend: {self.compile_config.get('backend', 'inductor')}")
                self.logger.info("Note: First iteration will be slow due to compilation...")
                self.logger.info("=" * 80)

            try:
                import torch
                if hasattr(torch, 'compile'):
                    # Use fullgraph=False to allow graph breaks for complex operations
                    # This is necessary for BEVFormer's temporal modeling (obtain_history_bev)
                    model = torch.compile(
                        model,
                        mode=self.compile_config.get('mode', 'reduce-overhead'),
                        backend=self.compile_config.get('backend', 'inductor'),
                        fullgraph=False,  # Allow graph breaks for complex operations
                        dynamic=True  # Handle dynamic shapes better
                    )
                    if is_main_process():
                        self.logger.info("✓ Model compiled successfully (fullgraph=False, dynamic=True)")
                        self.logger.info("  Note: Graph breaks allowed for temporal BEV operations")
                else:
                    if is_main_process():
                        self.logger.warning("torch.compile() not available (requires PyTorch 2.0+). Skipping compilation.")
            except Exception as e:
                if is_main_process():
                    self.logger.error(f"Failed to compile model: {e}")
                    self.logger.warning("Continuing without compilation...")

        # Wrap model with DDP - enable find_unused_parameters for complex models like BEVFormer
        model = DDP(model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)
        self.model = model
        return model

    def setup_data(self):
        """Create dataloaders with distributed samplers."""
        if is_main_process():
            self.logger.info("Creating distributed datasets and dataloaders...")

        # Create datasets
        train_loader, train_dataset = self._create_dataloader(self.config, training=True)
        val_loader, val_dataset = self._create_dataloader(self.config, training=False)

        # Replace samplers with distributed versions
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )

        # Recreate dataloaders with distributed samplers
        training_config = self.config.get('training', {})
        train_batch_size = training_config.get('batch_size', 1)
        val_batch_size = training_config.get('val_batch_size', 1)
        num_workers = training_config.get('num_workers', 32)

        # Optimize num_workers for better performance (use more workers if available)
        # # Recommended: 8-16 workers for faster data loading
        # if num_workers < 8:
        #     num_workers = min(8, os.cpu_count() or 4)

        from dataset import custom_collate_fn

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
            prefetch_factor=2 if num_workers > 0 else None  # Prefetch 2 batches per worker
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )

        if is_main_process():
            self.logger.info(f"Training samples: {len(train_dataset)}")
            self.logger.info(f"Validation samples: {len(val_dataset)}")
            self.logger.info(f"Training batches per GPU: {len(train_loader)}")
            self.logger.info(f"Validation batches per GPU: {len(val_loader)}")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

        return train_loader, val_loader

    def train_epoch(self, epoch, total_epochs=None):
        """Train one epoch with distributed sampling and optional profiling."""
        # Set epoch for distributed sampler
        self.train_sampler.set_epoch(epoch)

        # Setup profiler if configured and this is the first epoch
        if self.profiler_config.get('enabled', False) and epoch == 0 and self.rank == 0:
            self._setup_profiler()

        # Call parent train_epoch with profiler
        avg_loss = self._train_epoch_with_profiler(epoch, total_epochs)

        # Cleanup profiler after epoch if it was active
        if self.profiler is not None and epoch == 0:
            self._export_profiler_results()
            self.profiler = None

        return avg_loss

    def _setup_profiler(self):
        """Setup PyTorch profiler for performance analysis."""
        profile_dir = self.profiler_config.get('profile_dir', './profiling')
        wait_steps = self.profiler_config.get('wait', 2)
        warmup_steps = self.profiler_config.get('warmup', 1)
        active_steps = self.profiler_config.get('active', 5)
        repeat = self.profiler_config.get('repeat', 1)

        # Create profiling directory
        Path(profile_dir).mkdir(parents=True, exist_ok=True)

        if is_main_process():
            self.logger.info("=" * 80)
            self.logger.info("PROFILER ENABLED")
            self.logger.info("=" * 80)
            self.logger.info(f"Profile Directory: {profile_dir}")
            self.logger.info(f"Wait Steps: {wait_steps}")
            self.logger.info(f"Warmup Steps: {warmup_steps}")
            self.logger.info(f"Active Steps: {active_steps}")
            self.logger.info(f"Total Profiling Steps: {wait_steps + warmup_steps + active_steps}")
            self.logger.info("=" * 80)

        # Create profiler with comprehensive settings
        self.profiler = profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            schedule=schedule(
                wait=wait_steps,
                warmup=warmup_steps,
                active=active_steps,
                repeat=repeat
            ),
            on_trace_ready=self._on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True
        )

        # Start profiler
        self.profiler.__enter__()
        self.profiling_step = 0

    def _on_trace_ready(self, prof):
        """Callback when profiler trace is ready."""
        if not is_main_process():
            return

        profile_dir = self.profiler_config.get('profile_dir', './profiling')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Export Chrome trace (JSON format)
        trace_path = os.path.join(profile_dir, f'trace_{timestamp}.json')
        prof.export_chrome_trace(trace_path)
        self.logger.info(f"Exported Chrome trace to: {trace_path}")

        # Export TensorBoard trace
        tb_path = os.path.join(profile_dir, 'tensorboard')
        Path(tb_path).mkdir(parents=True, exist_ok=True)
        prof.export_stacks(os.path.join(tb_path, f'profiler_stacks_{timestamp}.txt'), "self_cuda_time_total")
        self.logger.info(f"Exported TensorBoard stacks to: {tb_path}")

    def _export_profiler_results(self):
        """Export profiler summary and results."""
        if not is_main_process() or self.profiler is None:
            return

        profile_dir = self.profiler_config.get('profile_dir', './profiling')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        try:
            # Stop profiler
            self.profiler.__exit__(None, None, None)

            # Export summary statistics
            summary_path = os.path.join(profile_dir, f'profiler_summary_{timestamp}.txt')
            with open(summary_path, 'w') as f:
                # Key averages by CPU time
                f.write("=" * 80 + "\n")
                f.write("TOP OPERATIONS BY CPU TIME\n")
                f.write("=" * 80 + "\n")
                f.write(str(self.profiler.key_averages().table(
                    sort_by="cpu_time_total", row_limit=20
                )))
                f.write("\n\n")

                # Key averages by CUDA time
                f.write("=" * 80 + "\n")
                f.write("TOP OPERATIONS BY CUDA TIME\n")
                f.write("=" * 80 + "\n")
                f.write(str(self.profiler.key_averages().table(
                    sort_by="cuda_time_total", row_limit=20
                )))
                f.write("\n\n")

                # Key averages by memory
                f.write("=" * 80 + "\n")
                f.write("TOP OPERATIONS BY MEMORY\n")
                f.write("=" * 80 + "\n")
                f.write(str(self.profiler.key_averages().table(
                    sort_by="self_cuda_memory_usage", row_limit=20
                )))
                f.write("\n\n")

                # Grouped by operation name
                f.write("=" * 80 + "\n")
                f.write("OPERATIONS GROUPED BY NAME\n")
                f.write("=" * 80 + "\n")
                f.write(str(self.profiler.key_averages(group_by_input_shape=True).table(
                    sort_by="cuda_time_total", row_limit=30
                )))

            self.logger.info(f"Exported profiler summary to: {summary_path}")
            self.logger.info("=" * 80)
            self.logger.info("PROFILING COMPLETED")
            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"Error exporting profiler results: {e}")

    def _train_epoch_with_profiler(self, epoch, total_epochs=None):
        """Train epoch with profiler integration."""
        if self.model is None or self.train_loader is None or self.optimizer is None:
            raise RuntimeError("Model, dataloader, or optimizer not initialized")

        self.model.train()
        total_loss = 0.0
        epoch_losses = {}

        log_interval = self.config.get('logging', {}).get('tensorboard', {}).get('log_interval', 50)

        # Check if profiling is active
        profiling_active = self.profiler is not None
        max_profiling_steps = 0
        if profiling_active:
            wait = self.profiler_config.get('wait', 2)
            warmup = self.profiler_config.get('warmup', 1)
            active = self.profiler_config.get('active', 5)
            max_profiling_steps = wait + warmup + active

        if is_main_process():
            if total_epochs:
                progress_percent = (epoch + 1) / total_epochs * 100
                self.logger.info("═" * 60)
                self.logger.info(f"EPOCH {epoch + 1}/{total_epochs} (Progress: {progress_percent:.1f}%)")
                if profiling_active:
                    self.logger.info(f"PROFILING: Will profile first {max_profiling_steps} iterations")
                self.logger.info("═" * 60)
            else:
                self.logger.info("═" * 60)
                self.logger.info(f"EPOCH {epoch + 1}")
                self.logger.info("═" * 60)

        import time
        epoch_start_time = time.time()
        self.epoch_start_times[epoch] = epoch_start_time

        for batch_idx, batch in enumerate(self.train_loader):
            iteration_start_time = time.time()
            # Profile only first few iterations if profiling enabled
            if profiling_active and batch_idx >= max_profiling_steps:
                if is_main_process() and batch_idx == max_profiling_steps:
                    self.logger.info(f"Profiling complete. Continuing normal training...")
                # Continue training without profiling
                profiling_active = False

            # Check if GT data is available
            if 'gt_bboxes_3d' not in batch or 'gt_labels_3d' not in batch:
                if is_main_process():
                    self.logger.warning(f"Training batch {batch_idx} missing GT data, skipping...")
                if self.profiler:
                    self.profiler.step()
                continue

            # Wrap batch processing in record_function for profiling
            with record_function("batch_processing"):
                # Move batch to device
                with record_function("data_to_device"):
                    batch_img = batch['img'].to(self.device)
                    batch_gt_bboxes_3d = [bbox.to(self.device) for bbox in batch['gt_bboxes_3d']]
                    batch_gt_labels_3d = [labels.to(self.device) for labels in batch['gt_labels_3d']]

                # Apply GPU augmentation if enabled (runs on GPU after data transfer)
                if self.gpu_augmentation is not None:
                    with record_function("gpu_augmentation"):
                        batch_img = self.gpu_augmentation(batch_img)

                # Determine AMP dtype
                amp_dtype = torch.float16
                if self.use_amp and self.amp_config.get('dtype') == 'bfloat16':
                    amp_dtype = torch.bfloat16

                # Forward pass with AMP
                with record_function("forward_pass"):
                    if self.use_amp:
                        with torch.cuda.amp.autocast(dtype=amp_dtype):
                            losses = self.model(
                                img=batch_img,
                                img_metas=batch['img_metas'],
                                gt_bboxes_3d=batch_gt_bboxes_3d,
                                gt_labels_3d=batch_gt_labels_3d,
                                return_loss=True
                            )
                    else:
                        losses = self.model(
                            img=batch_img,
                            img_metas=batch['img_metas'],
                            gt_bboxes_3d=batch_gt_bboxes_3d,
                            gt_labels_3d=batch_gt_labels_3d,
                            return_loss=True
                        )

                # Calculate total loss
                with record_function("loss_calculation"):
                    batch_total_loss = 0
                    for key, value in losses.items():
                        if torch.is_tensor(value) and value.requires_grad:
                            batch_total_loss += value
                            loss_val = value.item()
                            if key not in epoch_losses:
                                epoch_losses[key] = 0.0
                            epoch_losses[key] += loss_val

                # Backward pass with AMP
                with record_function("backward_pass"):
                    self.optimizer.zero_grad()
                    if self.use_amp:
                        self.scaler.scale(batch_total_loss).backward()
                    else:
                        batch_total_loss.backward()

                # Gradient clipping with AMP
                with record_function("gradient_clipping"):
                    grad_clip_norm = self.config.get('training', {}).get('grad_clip', {}).get('max_norm', 35.0)
                    if grad_clip_norm > 0:
                        if self.use_amp:
                            # Unscale before clipping
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)

                # Optimizer step with AMP
                with record_function("optimizer_step"):
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

            total_loss += batch_total_loss.item()

            # Calculate iteration time and metrics
            iteration_time = time.time() - iteration_start_time
            self.iteration_times.append(iteration_time)

            # Keep only recent iteration times (sliding window)
            if len(self.iteration_times) > 100:
                self.iteration_times = self.iteration_times[-100:]

            batch_size = batch_img.size(0)
            queue_length = batch_img.size(1)
            self.total_samples_processed += batch_size

            # Step profiler
            if self.profiler:
                self.profiler.step()
                self.profiling_step += 1

            # Enhanced logging with performance metrics
            if is_main_process() and batch_idx % log_interval == 0:
                from trainer.utils import calculate_training_metrics, format_duration, format_detailed_losses, get_world_size

                avg_loss = total_loss / (batch_idx + 1)

                # Calculate training metrics with world size
                avg_iteration_time = sum(self.iteration_times[-10:]) / len(self.iteration_times[-10:])
                world_size = get_world_size()
                metrics = calculate_training_metrics(batch_size, avg_iteration_time, queue_length, world_size)

                # Calculate ETA
                remaining_batches = len(self.train_loader) - batch_idx - 1
                eta_seconds = remaining_batches * avg_iteration_time
                eta_str = format_duration(eta_seconds)

                # Profile status
                profile_status = f" [PROFILING {self.profiling_step}/{max_profiling_steps}]" if profiling_active else ""

                # Main progress log
                self.logger.info(f"Batch {batch_idx + 1}/{len(self.train_loader)} | "
                               f"Loss: {batch_total_loss.item():.4f} | Avg: {avg_loss:.4f} | "
                               f"LR: {self.optimizer.param_groups[0]['lr']:.2e}{profile_status}")

                # Performance metrics log - show both total and per-GPU for distributed
                if world_size > 1:
                    self.logger.info(f"Speed: {metrics['samples_per_sec_total']:.1f} samples/sec total "
                                   f"({metrics['samples_per_sec_per_gpu']:.1f}/gpu) | "
                                   f"{metrics['images_per_sec_total']:.1f} images/sec total "
                                   f"({metrics['images_per_sec_per_gpu']:.1f}/gpu) | "
                                   f"Time/iter: {metrics['time_per_iteration']:.3f}s | ETA: {eta_str}")
                else:
                    self.logger.info(f"Speed: {metrics['samples_per_sec_total']:.1f} samples/sec | "
                                   f"{metrics['images_per_sec_total']:.1f} images/sec | "
                                   f"Time/iter: {metrics['time_per_iteration']:.3f}s | ETA: {eta_str}")

                # Detailed losses with better formatting
                formatted_losses = format_detailed_losses(losses)
                self.logger.info(f"\n{formatted_losses}")

                # TensorBoard logging - organized structure
                if self.writer:
                    global_step = epoch * len(self.train_loader) + batch_idx

                    # Performance metrics - both total and per-GPU
                    self.writer.add_scalar('performance/samples_per_sec_total', metrics['samples_per_sec_total'], global_step)
                    self.writer.add_scalar('performance/samples_per_sec_per_gpu', metrics['samples_per_sec_per_gpu'], global_step)
                    self.writer.add_scalar('performance/images_per_sec_total', metrics['images_per_sec_total'], global_step)
                    self.writer.add_scalar('performance/images_per_sec_per_gpu', metrics['images_per_sec_per_gpu'], global_step)
                    self.writer.add_scalar('performance/iteration_time', metrics['time_per_iteration'], global_step)
                    self.writer.add_scalar('performance/learning_rate', self.optimizer.param_groups[0]['lr'], global_step)

                    # Detailed train losses
                    self.writer.add_scalar('train/batch_total_loss', batch_total_loss.item(), global_step)
                    for key, value in losses.items():
                        if torch.is_tensor(value):
                            self.writer.add_scalar(f'train/{key}', value.item(), global_step)

        # Step scheduler
        if self.scheduler:
            self.scheduler.step()

        # Average losses for epoch
        avg_epoch_loss = total_loss / len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)

        # Average losses across all GPUs in distributed training
        if dist.is_available() and dist.is_initialized():
            loss_tensor = torch.tensor(avg_epoch_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_epoch_loss = loss_tensor.item()

            for key in epoch_losses:
                loss_tensor = torch.tensor(epoch_losses[key], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                epoch_losses[key] = loss_tensor.item()

        epoch_time = time.time() - epoch_start_time
        if is_main_process():
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s | Average Loss: {avg_epoch_loss:.4f}")

            # Log epoch averages to TensorBoard - only main losses
            if self.writer:
                self.writer.add_scalar('averages/train_loss', avg_epoch_loss, epoch)

                # Log epoch timing and performance
                self.writer.add_scalar('performance/epoch_time', epoch_time, epoch)
                if self.iteration_times:
                    avg_iter_time = sum(self.iteration_times) / len(self.iteration_times)
                    self.writer.add_scalar('performance/avg_iteration_time', avg_iter_time, epoch)

        return avg_epoch_loss

    def validate_epoch(self, epoch):
        """Validate one epoch with distributed sampling."""
        # Ensure all processes are synchronized
        dist.barrier()
        return super().validate_epoch(epoch)


def setup_distributed():
    """Initialize distributed training."""
    # Initialize the process group
    dist.init_process_group(backend='nccl')

    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))

    return rank, world_size, local_rank


def main():
    """Main distributed training function."""
    parser = argparse.ArgumentParser(description='BEVFormer Multi-GPU Distributed Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--exp-name', type=str,
                       help='Experiment name (overrides config)')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for logs')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory for checkpoints')

    # Profiling arguments
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling mode')
    parser.add_argument('--profile-dir', type=str, default='./profiling',
                       help='Directory to save profiling traces')
    parser.add_argument('--profile-wait', type=int, default=2,
                       help='Number of warmup iterations before profiling starts')
    parser.add_argument('--profile-warmup', type=int, default=1,
                       help='Number of warmup iterations within profiling')
    parser.add_argument('--profile-active', type=int, default=5,
                       help='Number of active profiling iterations')
    parser.add_argument('--profile-repeat', type=int, default=1,
                       help='Number of times to repeat profiling cycle')

    # Optimization arguments
    parser.add_argument('--compile', action='store_true',
                       help='Enable torch.compile() for model optimization (PyTorch 2.0+)')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='Compilation mode: default (balanced), reduce-overhead (faster), max-autotune (slowest compile, fastest run)')
    parser.add_argument('--compile-backend', type=str, default='inductor',
                       choices=['inductor', 'aot_eager', 'cudagraphs'],
                       help='Compilation backend (inductor is recommended)')

    # Mixed Precision Training
    parser.add_argument('--amp', action='store_true',
                       help='Enable Automatic Mixed Precision (AMP) training')
    parser.add_argument('--amp-dtype', type=str, default='float16',
                       choices=['float16', 'bfloat16'],
                       help='AMP dtype: float16 (default) or bfloat16 (better stability on A100/H100)')

    args = parser.parse_args()

    try:
        # Setup distributed training
        rank, world_size, local_rank = setup_distributed()

        # Load configuration
        config = load_config(args.config)

        # Extract experiment name
        exp_name = args.exp_name or config.get('experiment', {}).get('name', 'bevformer_multi_gpu')

        # Setup directories
        log_dir = os.path.join(args.log_dir, exp_name)
        checkpoint_dir = os.path.join(args.checkpoint_dir, exp_name)

        # Setup device for this process
        device = setup_device(rank=local_rank)

        # Setup logging (only main process logs)
        logger = setup_logging(log_dir, exp_name, rank=rank)

        if is_main_process():
            logger.info("=" * 80)
            logger.info("BEVFORMER MULTI-GPU DISTRIBUTED TRAINING")
            logger.info("=" * 80)
            logger.info(f"Experiment: {exp_name}")
            logger.info(f"Config: {args.config}")
            logger.info(f"Log directory: {log_dir}")
            logger.info(f"Checkpoint directory: {checkpoint_dir}")
            logger.info(f"World size: {world_size}")
            logger.info(f"Rank: {rank}")

        # Log device information
        if is_main_process():
            logger.info(f"Using device: {device}")
            if device.type == 'cuda':
                logger.info(f"GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
                memory_gb = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
                logger.info(f"GPU Memory: {memory_gb:.1f} GB")

        # Setup profiler configuration
        profiler_config = None
        if args.profile:
            profiler_config = {
                'enabled': True,
                'profile_dir': args.profile_dir,
                'wait': args.profile_wait,
                'warmup': args.profile_warmup,
                'active': args.profile_active,
                'repeat': args.profile_repeat
            }
            if is_main_process():
                logger.info("Profiling enabled - will profile first epoch only")

        # Setup compile configuration
        compile_config = None
        if args.compile:
            compile_config = {
                'enabled': True,
                'mode': args.compile_mode,
                'backend': args.compile_backend
            }
            if is_main_process():
                logger.info(f"torch.compile() enabled with mode='{args.compile_mode}', backend='{args.compile_backend}'")

        # Setup AMP configuration
        amp_config = None
        if args.amp:
            amp_config = {
                'enabled': True,
                'dtype': args.amp_dtype
            }
            if is_main_process():
                logger.info(f"Automatic Mixed Precision (AMP) enabled with dtype='{args.amp_dtype}'")

        # Create distributed trainer
        trainer = DistributedBEVFormerTrainer(
            config=config,
            device=device,
            logger=logger,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            rank=rank,
            world_size=world_size,
            profiler_config=profiler_config,
            compile_config=compile_config,
            amp_config=amp_config
        )

        # Setup training components
        if is_main_process():
            logger.info("Setting up distributed training components...")

        model = trainer.setup_model()
        train_loader, val_loader = trainer.setup_data()
        optimizer, scheduler = trainer.setup_optimizer_scheduler(model)
        writer = trainer.setup_tensorboard()

        # Handle resume (only on main process)
        start_epoch = 0
        if is_main_process():
            if args.resume and os.path.exists(args.resume):
                # Use explicit resume path
                start_epoch = trainer.load_checkpoint(args.resume)
            else:
                # Try auto-resume from latest checkpoint
                latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
                if os.path.exists(latest_checkpoint_path):
                    start_epoch = trainer.load_checkpoint(latest_checkpoint_path)

        # Broadcast start_epoch to all processes
        start_epoch_tensor = torch.tensor(start_epoch, device=device)
        dist.broadcast(start_epoch_tensor, src=0)
        start_epoch = start_epoch_tensor.item()

        # Run distributed training
        epochs = config.get('training', {}).get('epochs', 24)
        trainer.train(start_epoch=start_epoch, end_epoch=epochs)

        if is_main_process():
            logger.info("Distributed training completed successfully!")

    except KeyboardInterrupt:
        if is_main_process():
            logger.info("Training interrupted by user")
    except Exception as e:
        if is_main_process():
            logger.error(f"Distributed training failed: {e}")
        raise
    finally:
        # Clean up resources
        if 'trainer' in locals():
            trainer.cleanup()
        cleanup_distributed()


if __name__ == '__main__':
    main()