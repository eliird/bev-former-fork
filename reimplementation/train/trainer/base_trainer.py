"""BEVFormer Trainer Class"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(train_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, train_dir)

from dataset import NuScenesDataset, custom_collate_fn
from models import BEVFormer
from evaluation import calculate_nds_map, extract_detections_from_model_output
from .utils import count_parameters, is_main_process, get_rank


class BEVFormerTrainer:
    """BEVFormer training infrastructure"""

    def __init__(self, config: Dict[str, Any], device: torch.device, logger: logging.Logger,
                 log_dir: str, checkpoint_dir: str):
        """Initialize trainer.

        Args:
            config: Training configuration
            device: Compute device
            logger: Logger instance
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
        """
        self.config = config
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        self.best_loss = float('inf')

        # Setup directories
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def setup_model(self) -> BEVFormer:
        """Create and configure BEVFormer model."""
        model = self._create_model(self.config)
        model = model.to(self.device)

        # Log model information
        if is_main_process():
            count_parameters(model, self.logger)

        self.model = model
        return model

    def setup_data(self) -> Tuple[data.DataLoader, data.DataLoader]:
        """Create train and validation dataloaders."""
        if is_main_process():
            self.logger.info("Creating datasets and dataloaders...")

        train_loader, train_dataset = self._create_dataloader(self.config, training=True)
        val_loader, val_dataset = self._create_dataloader(self.config, training=False)

        if is_main_process():
            self.logger.info(f"Training samples: {len(train_dataset)}")
            self.logger.info(f"Validation samples: {len(val_dataset)}")
            self.logger.info(f"Training batches: {len(train_loader)}")
            self.logger.info(f"Validation batches: {len(val_loader)}")

        self.train_loader = train_loader
        self.val_loader = val_loader
        return train_loader, val_loader

    def setup_optimizer_scheduler(self, model: nn.Module) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler]]:
        """Setup optimizer and scheduler."""
        training_config = self.config.get('training', {})
        optimizer_config = training_config.get('optimizer', {})

        # Create optimizer
        lr = optimizer_config.get('lr', 2e-4)
        weight_decay = optimizer_config.get('weight_decay', 0.01)
        betas = optimizer_config.get('betas', [0.9, 0.999])
        eps = optimizer_config.get('eps', 1e-8)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

        if is_main_process():
            self.logger.info(f"Optimizer: AdamW, LR: {lr}, Weight Decay: {weight_decay}, Betas: {betas}, Eps: {eps}")

        # Create scheduler
        scheduler_config = training_config.get('scheduler', {})
        scheduler = None
        if scheduler_config.get('type') == 'CosineAnnealingLR':
            T_max = scheduler_config.get('T_max', training_config.get('epochs', 24))
            eta_min = scheduler_config.get('eta_min', 1e-7)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

            if is_main_process():
                self.logger.info(f"Scheduler: CosineAnnealingLR, T_max: {T_max}, eta_min: {eta_min}")

        self.optimizer = optimizer
        self.scheduler = scheduler
        return optimizer, scheduler

    def setup_tensorboard(self) -> Optional[SummaryWriter]:
        """Setup TensorBoard writer."""
        if is_main_process() and self.config.get('logging', {}).get('tensorboard', {}).get('enabled', True):
            writer = SummaryWriter(self.log_dir)
            self.writer = writer
            return writer
        return None

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch."""
        if self.model is None or self.train_loader is None or self.optimizer is None:
            raise RuntimeError("Model, dataloader, or optimizer not initialized")

        self.model.train()
        total_loss = 0.0
        epoch_losses = {}

        log_interval = self.config.get('logging', {}).get('tensorboard', {}).get('log_interval', 50)

        if is_main_process():
            self.logger.info(f"Starting epoch {epoch}")
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Check if GT data is available
            if 'gt_bboxes_3d' not in batch or 'gt_labels_3d' not in batch:
                if is_main_process():
                    self.logger.warning(f"Training batch {batch_idx} missing GT data, skipping...")
                continue

            # Move batch to device
            batch_img = batch['img'].to(self.device)
            batch_gt_bboxes_3d = [bbox.to(self.device) for bbox in batch['gt_bboxes_3d']]
            batch_gt_labels_3d = [labels.to(self.device) for labels in batch['gt_labels_3d']]

            # Forward pass
            losses = self.model(
                img=batch_img,
                img_metas=batch['img_metas'],
                gt_bboxes_3d=batch_gt_bboxes_3d,
                gt_labels_3d=batch_gt_labels_3d,
                return_loss=True
            )

            # Calculate total loss
            batch_total_loss = 0
            for key, value in losses.items():
                if torch.is_tensor(value) and value.requires_grad:
                    batch_total_loss += value

                    # Track individual losses
                    loss_val = value.item()
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += loss_val

            # Backward pass
            self.optimizer.zero_grad()
            batch_total_loss.backward()

            # Gradient clipping
            grad_clip_norm = self.config.get('training', {}).get('grad_clip', {}).get('max_norm', 35.0)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)

            self.optimizer.step()

            total_loss += batch_total_loss.item()

            # Logging
            if is_main_process() and batch_idx % log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                self.logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | "
                               f"Loss: {batch_total_loss.item():.4f} | Avg Loss: {avg_loss:.4f} | "
                               f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

                # Log individual losses
                loss_info = " | ".join([f"{k}: {v.item():.4f}" for k, v in losses.items() if torch.is_tensor(v)])
                self.logger.info(f"  Detailed losses: {loss_info}")

                # TensorBoard logging
                if self.writer:
                    global_step = epoch * len(self.train_loader) + batch_idx
                    self.writer.add_scalar('train/total_loss', batch_total_loss.item(), global_step)
                    self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
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

        epoch_time = time.time() - epoch_start_time
        if is_main_process():
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s | Average Loss: {avg_epoch_loss:.4f}")

        return avg_epoch_loss

    def validate_epoch(self, epoch: int) -> float:
        """Run validation with sequential scene processing."""
        if self.model is None or self.val_loader is None:
            raise RuntimeError("Model or validation dataloader not initialized")

        self.model.eval()
        total_loss = 0.0
        val_losses = {}

        # Check if we should compute NDS/mAP metrics
        compute_metrics = self.config.get('evaluation', {}).get('compute_metrics', False)
        predictions = [] if compute_metrics else None
        ground_truths = [] if compute_metrics else None

        # Get max evaluation samples from config
        max_eval_samples = self.config.get('evaluation', {}).get('max_eval_samples', -1)
        if max_eval_samples > 0 and max_eval_samples < len(self.val_loader):
            batches_to_process = max_eval_samples
            if is_main_process():
                self.logger.info(f"Running validation on {batches_to_process} sequential samples...")
        else:
            batches_to_process = len(self.val_loader)
            if is_main_process():
                self.logger.info("Running validation on all samples...")

        if compute_metrics and is_main_process():
            self.logger.info("Computing NDS and mAP metrics with sequential scene processing...")

        with torch.no_grad():
            processed_batches = 0
            current_scene_token = None

            # Reset temporal state at start
            self.model.prev_frame_info = {'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}

            for batch_idx, batch_data in enumerate(self.val_loader):
                if batch_idx >= batches_to_process:
                    break

                try:
                    # Check if GT data is available
                    if 'gt_bboxes_3d' not in batch_data or 'gt_labels_3d' not in batch_data:
                        if is_main_process():
                            self.logger.warning(f"Validation batch {batch_idx} missing GT data, skipping...")
                        continue

                    # Move batch to device
                    batch_img = batch_data['img'].to(self.device)
                    batch_gt_bboxes_3d = [bbox.to(self.device) for bbox in batch_data['gt_bboxes_3d']]
                    batch_gt_labels_3d = [labels.to(self.device) for labels in batch_data['gt_labels_3d']]

                    # Forward pass for loss calculation
                    losses = self.model(
                        img=batch_img,
                        img_metas=batch_data['img_metas'],
                        gt_bboxes_3d=batch_gt_bboxes_3d,
                        gt_labels_3d=batch_gt_labels_3d,
                        return_loss=True
                    )

                    # Accumulate losses
                    batch_total_loss = 0
                    for key, value in losses.items():
                        if torch.is_tensor(value):
                            loss_val = value.item()
                            batch_total_loss += loss_val

                            if key not in val_losses:
                                val_losses[key] = 0.0
                            val_losses[key] += loss_val

                    total_loss += batch_total_loss

                    # Sequential inference for metrics calculation
                    if compute_metrics:
                        try:
                            # Process each sample in the batch sequentially
                            batch_size = batch_img.size(0)
                            queue_length = batch_img.size(1)

                            for sample_idx in range(batch_size):
                                # Extract current frame for this sample
                                current_img = batch_img[sample_idx, -1, ...]

                                # Extract current frame meta
                                current_meta = batch_data['img_metas'][sample_idx][queue_length - 1]

                                # Check for scene change
                                scene_token = current_meta.get('scene_token')
                                if scene_token != current_scene_token:
                                    # New scene - reset temporal state
                                    self.model.prev_frame_info = {'prev_bev': None, 'scene_token': scene_token, 'prev_pos': 0, 'prev_angle': 0}
                                    current_scene_token = scene_token

                                # Forward test with proper format
                                model_output = self.model.forward_test(
                                    img=[current_img],
                                    img_metas=[[current_meta]]
                                )

                                # Extract detections
                                detections = extract_detections_from_model_output(model_output)
                                predictions.append(detections)

                                # Clear GPU cache after each forward_test
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                                # Process ground truths for this sample
                                if len(batch_gt_bboxes_3d) > sample_idx and len(batch_gt_labels_3d) > sample_idx:
                                    gt_dict = {
                                        'gt_bboxes_3d': batch_gt_bboxes_3d[sample_idx],
                                        'gt_labels_3d': batch_gt_labels_3d[sample_idx]
                                    }
                                    ground_truths.append(gt_dict)
                                else:
                                    # Empty ground truth
                                    gt_dict = {
                                        'gt_bboxes_3d': torch.zeros((0, 9)),
                                        'gt_labels_3d': torch.zeros(0, dtype=torch.long)
                                    }
                                    ground_truths.append(gt_dict)

                            # Clear GPU cache after each batch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        except Exception as e:
                            if is_main_process():
                                self.logger.warning(f"Error collecting predictions for batch {batch_idx}: {e}")
                            # Clear GPU cache even on error
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            # Add empty prediction/GT to keep lists aligned
                            if compute_metrics:
                                predictions.append({
                                    'boxes_3d': torch.zeros((0, 9)),
                                    'scores_3d': torch.zeros(0),
                                    'labels_3d': torch.zeros(0, dtype=torch.long)
                                })
                                ground_truths.append({
                                    'gt_bboxes_3d': torch.zeros((0, 9)),
                                    'gt_labels_3d': torch.zeros(0, dtype=torch.long)
                                })

                    val_log_interval = self.config.get('logging', {}).get('val_log_interval', 50)
                    if is_main_process() and batch_idx % val_log_interval == 0:
                        self.logger.info(f"Validation batch {batch_idx}: loss={batch_total_loss:.4f}")

                    processed_batches += 1

                except Exception as e:
                    if is_main_process():
                        self.logger.warning(f"Error processing batch {batch_idx}: {e}")
                    continue

        # Average losses
        avg_total_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
        for key in val_losses:
            val_losses[key] /= processed_batches

        # Calculate metrics if enabled
        metrics_results = {}
        if compute_metrics and predictions and ground_truths and is_main_process():
            class_names = self.config.get('data', {}).get('class_names', [])
            distance_thresholds = self.config.get('evaluation', {}).get('distance_thresholds', [0.5, 1.0, 2.0, 4.0])

            self.logger.info("Computing NDS and mAP metrics...")
            try:
                metrics_results = calculate_nds_map(
                    predictions,
                    ground_truths,
                    class_names,
                    distance_thresholds
                )
                self.logger.info(f"Validation Metrics:")
                self.logger.info(f"  NDS: {metrics_results['NDS']:.4f}")
                self.logger.info(f"  mAP: {metrics_results['mAP']:.4f}")

                # Log per-class AP if available
                if 'per_class_AP' in metrics_results:
                    for class_name, ap in metrics_results['per_class_AP'].items():
                        self.logger.info(f"  {class_name}_AP: {ap:.4f}")
            except Exception as e:
                self.logger.warning(f"Failed to compute metrics: {e}")

        if is_main_process():
            self.logger.info(f"Validation Results (processed {processed_batches} batches):")
            self.logger.info(f"  Total Loss: {avg_total_loss:.4f}")
            for key, value in val_losses.items():
                self.logger.info(f"  {key}: {value:.4f}")

        # Log to TensorBoard
        if self.writer and is_main_process():
            self.writer.add_scalar('val/total_loss', avg_total_loss, epoch)
            for key, value in val_losses.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)

            # Log metrics to TensorBoard
            if metrics_results:
                self.writer.add_scalar('val/NDS', metrics_results['NDS'], epoch)
                self.writer.add_scalar('val/mAP', metrics_results['mAP'], epoch)

                # Log per-class AP
                if 'per_class_AP' in metrics_results:
                    for class_name, ap in metrics_results['per_class_AP'].items():
                        self.writer.add_scalar(f'val/{class_name}_AP', ap, epoch)

        self.model.train()
        return avg_total_loss

    def save_checkpoint(self, epoch: int, step: int = 0, is_best: bool = False,
                       filename: Optional[str] = None) -> str:
        """Save training checkpoint."""
        if not is_main_process():
            return ""

        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if filename is None:
            filename = f'checkpoint_epoch_{epoch:03d}.pth'

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)

        # Also save as latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_loss = checkpoint.get('best_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', 0) + 1

        if is_main_process():
            self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
            self.logger.info(f"Resuming from epoch: {start_epoch}")

        return start_epoch

    def train(self, start_epoch: int = 0, end_epoch: Optional[int] = None) -> None:
        """Main training loop."""
        if end_epoch is None:
            end_epoch = self.config.get('training', {}).get('epochs', 24)

        eval_interval = self.config.get('evaluation', {}).get('eval_interval', 1)
        save_interval = self.config.get('checkpoint', {}).get('save_interval', 5)

        if is_main_process():
            self.logger.info("Starting training...")
        training_start_time = time.time()

        for epoch in range(start_epoch, end_epoch):
            # Training
            avg_train_loss = self.train_epoch(epoch)

            # Validation
            if (epoch + 1) % eval_interval == 0:
                avg_val_loss = self.validate_epoch(epoch)

                # Save best model
                if avg_val_loss < self.best_loss:
                    self.best_loss = avg_val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    if is_main_process():
                        self.logger.info(f"New best model saved! Val loss: {self.best_loss:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch)
                if is_main_process():
                    self.logger.info(f"Checkpoint saved at epoch {epoch}")

        total_training_time = time.time() - training_start_time
        if is_main_process():
            self.logger.info(f"Training completed! Total time: {total_training_time / 3600:.2f} hours")
            self.logger.info(f"Best validation loss: {self.best_loss:.4f}")

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.writer:
            self.writer.close()

    def _create_model(self, config: Dict[str, Any]) -> BEVFormer:
        """Create BEVFormer model with configuration from YAML."""
        # Extract parameters from config
        model_config = config.get('model', {})
        data_config = config.get('data', {})

        # Basic model parameters
        embed_dims = model_config.get('embed_dims', 256)
        encoder_layers = model_config.get('encoder_layers', 3)
        decoder_layers = model_config.get('decoder_layers', 3)
        num_query = model_config.get('num_query', 900)
        bev_h = model_config.get('bev_h', 200)
        bev_w = model_config.get('bev_w', 200)
        use_grid_mask = model_config.get('use_grid_mask', False)

        # Backbone configuration
        backbone_config = model_config.get('backbone', {})
        backbone_depth = backbone_config.get('depth', 50)
        num_stages = backbone_config.get('num_stages', 4)
        out_indices = tuple(backbone_config.get('out_indices', [1, 2, 3]))
        frozen_stages = backbone_config.get('frozen_stages', 1)
        use_checkpoint = backbone_config.get('with_cp', False)

        # Neck configuration
        neck_config = model_config.get('neck', {})
        in_channels = neck_config.get('in_channels', [512, 1024, 2048])
        out_channels = neck_config.get('out_channels', embed_dims)
        num_outs = neck_config.get('num_outs', 4)
        start_level = neck_config.get('start_level', 0)
        add_extra_convs = neck_config.get('add_extra_convs', 'on_output')
        relu_before_extra_convs = neck_config.get('relu_before_extra_convs', True)

        # Transformer configuration
        transformer_config = model_config.get('transformer', {})
        num_points_in_pillar = transformer_config.get('num_points_in_pillar', 4)
        return_intermediate_encoder = transformer_config.get('return_intermediate_encoder', False)
        return_intermediate_decoder = transformer_config.get('return_intermediate_decoder', True)
        rotate_prev_bev = transformer_config.get('rotate_prev_bev', True)
        use_shift = transformer_config.get('use_shift', True)
        use_can_bus = transformer_config.get('use_can_bus', True)

        # Bbox coder configuration
        bbox_coder_config = model_config.get('bbox_coder', {})
        post_center_range = bbox_coder_config.get('post_center_range', [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0])
        max_num = bbox_coder_config.get('max_num', 300)

        # Loss configuration
        loss_config = model_config.get('loss', {})
        cls_loss_config = loss_config.get('cls', {})
        bbox_loss_config = loss_config.get('bbox', {})
        iou_loss_config = loss_config.get('iou', {})

        # Point cloud range and class names from data config
        point_cloud_range = data_config.get('pc_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        class_names = data_config.get('class_names', [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ])

        # Create complete model configuration
        model_cfg = dict(
            img_backbone=dict(
                depth=backbone_depth,
                num_stages=num_stages,
                out_indices=out_indices,
                frozen_stages=frozen_stages,
                with_cp=use_checkpoint
            ),
            img_neck=dict(
                in_channels=in_channels,
                out_channels=out_channels,
                num_outs=num_outs,
                start_level=start_level,
                add_extra_convs=add_extra_convs,
                relu_before_extra_convs=relu_before_extra_convs
            ),
            pts_bbox_head=dict(
                num_classes=len(class_names),
                in_channels=embed_dims,
                num_query=num_query,
                bev_h=bev_h,
                bev_w=bev_w,
                transformer=dict(
                    embed_dims=embed_dims,
                    encoder=dict(
                        num_layers=encoder_layers,
                        pc_range=point_cloud_range,
                        num_points_in_pillar=num_points_in_pillar,
                        return_intermediate=return_intermediate_encoder
                    ),
                    decoder=dict(
                        num_layers=decoder_layers,
                        return_intermediate=return_intermediate_decoder
                    ),
                    rotate_prev_bev=rotate_prev_bev,
                    use_shift=use_shift,
                    use_can_bus=use_can_bus
                ),
                bbox_coder=dict(
                    pc_range=point_cloud_range,
                    post_center_range=post_center_range,
                    max_num=max_num,
                    num_classes=len(class_names)
                ),
                loss_cls=dict(
                    use_sigmoid=cls_loss_config.get('use_sigmoid', True),
                    gamma=cls_loss_config.get('gamma', 2.0),
                    alpha=cls_loss_config.get('alpha', 0.25),
                    loss_weight=cls_loss_config.get('loss_weight', 2.0)
                ),
                loss_bbox=dict(
                    loss_weight=bbox_loss_config.get('loss_weight', 0.25)
                ),
                loss_iou=dict(
                    loss_weight=iou_loss_config.get('loss_weight', 0.0)
                ),
                train_cfg=dict(
                    pts=dict(
                        assigner=dict(
                            pc_range=point_cloud_range
                        )
                    )
                )
            ),
            use_grid_mask=use_grid_mask,
            video_test_mode=False
        )

        model = BEVFormer(**model_cfg)
        return model

    def _create_dataloader(self, config: Dict[str, Any], training: bool = True) -> Tuple[data.DataLoader, NuScenesDataset]:
        """Create train/val dataloader from config."""
        data_config = config.get('data', {})
        training_config = config.get('training', {})

        data_root = data_config.get('data_root', '../data/nuscenes')

        if training:
            data_file = os.path.join(data_root, data_config.get('train_pkl', 'nuscenes_infos_temporal_train.pkl'))
            batch_size = training_config.get('batch_size', 1)
        else:
            data_file = os.path.join(data_root, data_config.get('val_pkl', 'nuscenes_infos_temporal_val.pkl'))
            batch_size = training_config.get('val_batch_size', 1)

        queue_length = data_config.get('queue_length', 4)
        num_workers = training_config.get('num_workers', 4)
        point_cloud_range = data_config.get('pc_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        class_names = data_config.get('class_names', [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ])

        dataset = NuScenesDataset(
            data_file=data_file,
            queue_length=queue_length,
            training=training,
            point_cloud_range=point_cloud_range,
            class_names=class_names
        )

        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=training
        )

        return dataloader, dataset