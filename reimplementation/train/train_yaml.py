#!/usr/bin/env python3
"""
BEVFormer Training Script with YAML Configuration
Based on working train_bevformer.py with YAML config support and proper logging
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset import NuScenesDataset, custom_collate_fn
from models import BEVFormer
from utils.config_parser import load_config


def setup_logging(log_dir: str, exp_name: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger('BEVFormer')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    log_file = log_dir / f'{exp_name}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


def create_model(config) -> BEVFormer:
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


def create_dataloader(config, training: bool = True):
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
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=training
    )

    return dataloader, dataset


def save_checkpoint(model, optimizer, scheduler, epoch, iteration, best_loss, checkpoint_dir, filename=None):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f'checkpoint_epoch_{epoch:03d}.pth'

    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_loss': best_loss,
    }

    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)

    # Also save as latest checkpoint
    latest_path = checkpoint_dir / 'latest_checkpoint.pth'
    torch.save(checkpoint, latest_path)

    return checkpoint_path


def validate(model, val_loader, logger, writer, epoch, config):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    val_losses = {}

    # Get max evaluation samples from config
    max_eval_samples = config.get('evaluation', {}).get('max_eval_samples', -1)
    if max_eval_samples > 0 and max_eval_samples < len(val_loader):
        logger.info(f"Running validation on {max_eval_samples} random samples...")
    else:
        max_eval_samples = len(val_loader)
        logger.info("Running validation on all samples...")

    with torch.no_grad():
        # Create random indices if we're using a subset
        if max_eval_samples < len(val_loader):
            import random
            val_indices = random.sample(range(len(val_loader)), max_eval_samples)
            val_indices.sort()  # Keep some order for consistent logging
        else:
            val_indices = range(len(val_loader))

        processed_batches = 0
        for batch_idx_in_loader, batch in enumerate(val_loader):
            # Skip this batch if it's not in our selected indices
            if max_eval_samples < len(val_loader) and batch_idx_in_loader not in val_indices:
                continue

            batch_idx = processed_batches  # Use processed batch count for logging
            # Check if GT data is available
            if 'gt_bboxes_3d' not in batch or 'gt_labels_3d' not in batch:
                logger.warning(f"Validation batch {batch_idx} missing GT data, skipping...")
                continue

            # Move batch to GPU
            if torch.cuda.is_available():
                batch_img = batch['img'].cuda()
                batch_gt_bboxes_3d = [bbox.cuda() for bbox in batch['gt_bboxes_3d']]
                batch_gt_labels_3d = [labels.cuda() for labels in batch['gt_labels_3d']]
            else:
                batch_img = batch['img']
                batch_gt_bboxes_3d = batch['gt_bboxes_3d']
                batch_gt_labels_3d = batch['gt_labels_3d']

            # Forward pass
            losses = model.forward_train(
                img=batch_img,
                img_metas=batch['img_metas'],
                gt_bboxes_3d=batch_gt_bboxes_3d,
                gt_labels_3d=batch_gt_labels_3d
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

            val_log_interval = config.get('logging', {}).get('val_log_interval', 50)
            if batch_idx % val_log_interval == 0:
                logger.info(f"Validation batch {batch_idx}/{max_eval_samples}: loss={batch_total_loss:.4f}")

            processed_batches += 1

    # Average losses
    avg_total_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
    for key in val_losses:
        val_losses[key] /= processed_batches

    logger.info(f"Validation Results (processed {processed_batches} batches):")
    logger.info(f"  Total Loss: {avg_total_loss:.4f}")
    for key, value in val_losses.items():
        logger.info(f"  {key}: {value:.4f}")

    # Log to TensorBoard
    if writer:
        writer.add_scalar('val/total_loss', avg_total_loss, epoch)
        for key, value in val_losses.items():
            writer.add_scalar(f'val/{key}', value, epoch)

    model.train()
    return avg_total_loss


def train_epoch(model, train_loader, optimizer, scheduler, epoch, logger, writer, config):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    epoch_losses = {}

    log_interval = config.get('logging', {}).get('tensorboard', {}).get('log_interval', 50)

    logger.info(f"Starting epoch {epoch}")
    epoch_start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        # Check if GT data is available
        if 'gt_bboxes_3d' not in batch or 'gt_labels_3d' not in batch:
            logger.warning(f"Training batch {batch_idx} missing GT data, skipping...")
            continue

        # Move batch to GPU
        if torch.cuda.is_available():
            batch_img = batch['img'].cuda()
            batch_gt_bboxes_3d = [bbox.cuda() for bbox in batch['gt_bboxes_3d']]
            batch_gt_labels_3d = [labels.cuda() for labels in batch['gt_labels_3d']]
        else:
            batch_img = batch['img']
            batch_gt_bboxes_3d = batch['gt_bboxes_3d']
            batch_gt_labels_3d = batch['gt_labels_3d']

        # Forward pass
        losses = model.forward_train(
            img=batch_img,
            img_metas=batch['img_metas'],
            gt_bboxes_3d=batch_gt_bboxes_3d,
            gt_labels_3d=batch_gt_labels_3d
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
        optimizer.zero_grad()
        batch_total_loss.backward()

        # Gradient clipping
        grad_clip_norm = config.get('training', {}).get('grad_clip', {}).get('max_norm', 35.0)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        total_loss += batch_total_loss.item()

        # Logging
        if batch_idx % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                       f"Loss: {batch_total_loss.item():.4f} | Avg Loss: {avg_loss:.4f} | "
                       f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Log individual losses
            loss_info = " | ".join([f"{k}: {v.item():.4f}" for k, v in losses.items() if torch.is_tensor(v)])
            logger.info(f"  Detailed losses: {loss_info}")

            # TensorBoard logging
            if writer:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/total_loss', batch_total_loss.item(), global_step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                for key, value in losses.items():
                    if torch.is_tensor(value):
                        writer.add_scalar(f'train/{key}', value.item(), global_step)

    # Step scheduler
    if scheduler:
        scheduler.step()

    # Average losses for epoch
    avg_epoch_loss = total_loss / len(train_loader)
    for key in epoch_losses:
        epoch_losses[key] /= len(train_loader)

    epoch_time = time.time() - epoch_start_time
    logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s | Average Loss: {avg_epoch_loss:.4f}")

    return avg_epoch_loss


def main():
    parser = argparse.ArgumentParser(description='BEVFormer Training with YAML Config')
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

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Extract experiment name
    exp_name = args.exp_name or config.get('experiment', {}).get('name', 'bevformer_experiment')

    # Setup directories
    log_dir = os.path.join(args.log_dir, exp_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(log_dir, exp_name)
    logger.info("=" * 80)
    logger.info("BEVFORMER TRAINING WITH YAML CONFIG")
    logger.info("=" * 80)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create datasets and dataloaders
    logger.info("Creating datasets and dataloaders...")
    train_loader, train_dataset = create_dataloader(config, training=True)
    val_loader, val_dataset = create_dataloader(config, training=False)

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    training_config = config.get('training', {})
    optimizer_config = training_config.get('optimizer', {})
    lr = optimizer_config.get('lr', 2e-4)
    weight_decay = optimizer_config.get('weight_decay', 0.01)
    betas = optimizer_config.get('betas', [0.9, 0.999])
    eps = optimizer_config.get('eps', 1e-8)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    logger.info(f"Optimizer: AdamW, LR: {lr}, Weight Decay: {weight_decay}, Betas: {betas}, Eps: {eps}")

    # Create scheduler
    scheduler_config = training_config.get('scheduler', {})
    if scheduler_config.get('type') == 'CosineAnnealingLR':
        T_max = scheduler_config.get('T_max', training_config.get('epochs', 24))
        eta_min = scheduler_config.get('eta_min', 1e-7)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        logger.info(f"Scheduler: CosineAnnealingLR, T_max: {T_max}, eta_min: {eta_min}")
    else:
        scheduler = None

    # Setup TensorBoard
    writer = SummaryWriter(log_dir) if config.get('logging', {}).get('tensorboard', {}).get('enabled', True) else None

    # Auto-resume from latest checkpoint if available
    start_epoch = 0
    best_loss = float('inf')
    latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')

    if args.resume and os.path.exists(args.resume):
        # Use explicit resume path if provided
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"Resumed from checkpoint: {args.resume}")
        logger.info(f"Starting from epoch: {start_epoch}")
    elif os.path.exists(latest_checkpoint_path):
        # Auto-resume from latest checkpoint
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"Auto-resumed from latest checkpoint: {latest_checkpoint_path}")
        logger.info(f"Starting from epoch: {start_epoch}")

    # Training loop
    epochs = training_config.get('epochs', 24)
    eval_interval = config.get('evaluation', {}).get('eval_interval', 1)

    logger.info("Starting training...")
    training_start_time = time.time()

    for epoch in range(start_epoch, epochs):
        # Train epoch
        avg_train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, logger, writer, config)

        # Validation
        if (epoch + 1) % eval_interval == 0:
            avg_val_loss = validate(model, val_loader, logger, writer, epoch, config)

            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, 0, best_loss,
                              checkpoint_dir, f'best_model.pth')
                logger.info(f"New best model saved! Val loss: {best_loss:.4f}")

        # Save periodic checkpoint
        save_interval = config.get('checkpoint', {}).get('save_interval', 5)
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, 0, best_loss, checkpoint_dir)
            logger.info(f"Checkpoint saved at epoch {epoch}")

    total_training_time = time.time() - training_start_time
    logger.info(f"Training completed! Total time: {total_training_time / 3600:.2f} hours")
    logger.info(f"Best validation loss: {best_loss:.4f}")

    # Clean up
    if writer:
        writer.close()


if __name__ == '__main__':
    main()