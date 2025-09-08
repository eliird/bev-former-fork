#!/usr/bin/env python3
"""
BEVFormer Training Script
Pure PyTorch implementation for training BEVFormer on nuScenes dataset
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

# Add paths for our reimplementation
sys.path.append('/home/irdali.durrani/po-pi/BEVFormer/reimplementation')
sys.path.append('/home/irdali.durrani/po-pi/BEVFormer/reimplementation/models')
sys.path.append('/home/irdali.durrani/po-pi/BEVFormer/reimplementation/dataset')

from dataset.nuscenes_dataset import NuScenesDataset
from dataset.collate_fn import custom_collate_fn
from bevformer import BEVFormer


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


def create_model(args) -> BEVFormer:
    """Create BEVFormer model with configuration."""
    embed_dims = args.embed_dims
    
    model_cfg = dict(
        img_backbone=dict(
            depth=args.backbone_depth,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            with_cp=args.use_checkpoint
        ),
        img_neck=dict(
            in_channels=[512, 1024, 2048] if args.backbone_depth == 50 else [256, 512, 1024, 2048],
            out_channels=embed_dims,
            num_outs=4,
            start_level=0,
            add_extra_convs='on_output',
            relu_before_extra_convs=True
        ),
        pts_bbox_head=dict(
            num_classes=len(args.class_names),
            in_channels=embed_dims,
            num_query=args.num_query,
            bev_h=args.bev_h,
            bev_w=args.bev_w,
            transformer=dict(
                embed_dims=embed_dims,
                encoder=dict(
                    num_layers=args.encoder_layers,
                    pc_range=args.point_cloud_range,
                    num_points_in_pillar=4,
                    return_intermediate=False
                ),
                decoder=dict(
                    num_layers=args.decoder_layers,
                    return_intermediate=True
                ),
                rotate_prev_bev=True,
                use_shift=True,
                use_can_bus=True
            ),
            bbox_coder=dict(
                pc_range=args.point_cloud_range,
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=300,
                num_classes=len(args.class_names)
            ),
            train_cfg=dict(
                pts=dict(
                    assigner=dict(
                        pc_range=args.point_cloud_range
                    )
                )
            )
        ),
        use_grid_mask=args.use_grid_mask,
        video_test_mode=False
    )
    
    model = BEVFormer(**model_cfg)
    return model


def create_dataloader(args, training: bool = True):
    """Create train/val dataloader."""
    if training:
        data_file = os.path.join(args.data_root, args.train_pkl)
    else:
        data_file = os.path.join(args.data_root, args.val_pkl)
    
    dataset = NuScenesDataset(
        data_file=data_file,
        queue_length=args.queue_length,
        training=training,
        point_cloud_range=args.point_cloud_range,
        class_names=args.class_names
    )
    
    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch_size if training else args.val_batch_size,
        shuffle=training,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=training
    )
    
    return dataloader, dataset


def save_checkpoint(model, optimizer, scheduler, epoch, iteration, best_loss, args, filename=None):
    """Save model checkpoint."""
    checkpoint_dir = Path(args.checkpoint_dir)
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
        'args': vars(args)
    }
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest checkpoint
    latest_path = checkpoint_dir / 'latest_checkpoint.pth'
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, logger):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0, 0, float('inf')
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    iteration = checkpoint.get('iteration', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    
    logger.info(f"Resumed from epoch {epoch}, iteration {iteration}, best_loss {best_loss:.4f}")
    
    return epoch, iteration, best_loss


def validate(model, val_loader, logger, writer, epoch):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    val_losses = {}
    
    logger.info("Running validation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
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
                if torch.is_tensor(value) and value.requires_grad:
                    loss_val = value.item()
                    batch_total_loss += loss_val
                    
                    if key not in val_losses:
                        val_losses[key] = 0.0
                    val_losses[key] += loss_val
            
            total_loss += batch_total_loss
            total_samples += batch_img.size(0)
            
            if batch_idx % 10 == 0:
                logger.info(f"Validation batch {batch_idx}/{len(val_loader)}: loss={batch_total_loss:.4f}")
    
    # Average losses
    avg_total_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    for key in val_losses:
        val_losses[key] /= len(val_loader)
    
    logger.info(f"Validation Results:")
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


def train_epoch(model, train_loader, optimizer, scheduler, epoch, logger, writer, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    epoch_losses = {}
    
    for batch_idx, batch in enumerate(train_loader):
        iteration = epoch * len(train_loader) + batch_idx
        
        # Move batch to GPU
        if torch.cuda.is_available():
            batch_img = batch['img'].cuda()
            batch_gt_bboxes_3d = [bbox.cuda() for bbox in batch['gt_bboxes_3d']]
            batch_gt_labels_3d = [labels.cuda() for labels in batch['gt_labels_3d']]
        else:
            batch_img = batch['img']
            batch_gt_bboxes_3d = batch['gt_bboxes_3d']
            batch_gt_labels_3d = batch['gt_labels_3d']
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        losses = model.forward_train(
            img=batch_img,
            img_metas=batch['img_metas'],
            gt_bboxes_3d=batch_gt_bboxes_3d,
            gt_labels_3d=batch_gt_labels_3d
        )
        
        # Compute total loss
        loss_total = 0
        batch_losses = {}
        for key, value in losses.items():
            if torch.is_tensor(value) and value.requires_grad:
                loss_total += value
                batch_losses[key] = value.item()
                
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value.item()
        
        # Backward pass
        loss_total.backward()
        
        # Gradient clipping
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
        
        # Optimizer step
        optimizer.step()
        
        # Scheduler step (if per-iteration)
        if scheduler and args.scheduler_step == 'iteration':
            scheduler.step()
        
        total_loss += loss_total.item()
        
        # Logging every 10 iterations
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            lr = optimizer.param_groups[0]['lr']
            
            logger.info(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Batch {batch_idx + 1:4d}/{len(train_loader)} | "
                f"Loss: {loss_total.item():.4f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e}"
            )
            
            # Individual loss components
            for key, value in batch_losses.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # TensorBoard logging
            if writer:
                writer.add_scalar('train/total_loss', loss_total.item(), iteration)
                writer.add_scalar('train/avg_loss', avg_loss, iteration)
                writer.add_scalar('train/learning_rate', lr, iteration)
                
                for key, value in batch_losses.items():
                    writer.add_scalar(f'train/{key}', value, iteration)
    
    # Scheduler step (if per-epoch)
    if scheduler and args.scheduler_step == 'epoch':
        scheduler.step()
    
    # Average epoch losses
    avg_epoch_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    for key in epoch_losses:
        epoch_losses[key] /= len(train_loader)
    
    return avg_epoch_loss, epoch_losses


def main():
    parser = argparse.ArgumentParser(description='BEVFormer Training Script')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='/home/irdali.durrani/po-pi/BEVFormer/reimplementation/data/nuscenes',
                        help='Root directory of the dataset')
    parser.add_argument('--train_pkl', type=str, default='nuscenes_infos_temporal_train.pkl',
                        help='Training dataset pickle file')
    parser.add_argument('--val_pkl', type=str, default='nuscenes_infos_temporal_val.pkl',
                        help='Validation dataset pickle file')
    
    # Model arguments
    parser.add_argument('--backbone_depth', type=int, default=50, choices=[50, 101],
                        help='ResNet backbone depth')
    parser.add_argument('--embed_dims', type=int, default=256,
                        help='Embedding dimensions')
    parser.add_argument('--encoder_layers', type=int, default=6,
                        help='Number of encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=6,
                        help='Number of decoder layers')
    parser.add_argument('--num_query', type=int, default=900,
                        help='Number of object queries')
    parser.add_argument('--bev_h', type=int, default=200,
                        help='BEV height')
    parser.add_argument('--bev_w', type=int, default=200,
                        help='BEV width')
    parser.add_argument('--use_grid_mask', action='store_true',
                        help='Use grid mask augmentation')
    parser.add_argument('--use_checkpoint', action='store_true',
                        help='Use gradient checkpointing')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='Validation batch size')
    parser.add_argument('--epochs', type=int, default=24,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_clip_norm', type=float, default=35.0,
                        help='Gradient clipping norm (0 to disable)')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--scheduler_step', type=str, default='epoch', choices=['epoch', 'iteration'],
                        help='Scheduler step frequency')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='Warmup epochs')
    
    # Data loading arguments
    parser.add_argument('--queue_length', type=int, default=4,
                        help='Temporal sequence length')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Logging and checkpointing
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Logging directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (iterations)')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Validation interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Checkpoint save interval (epochs)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--auto_resume', action='store_true',
                        help='Auto resume from latest checkpoint')
    
    args = parser.parse_args()
    
    # Setup experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f'bevformer_{timestamp}'
    
    # Setup class names and point cloud range
    args.class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    args.point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.exp_name)
    logger.info("=" * 80)
    logger.info("BEVFORMER TRAINING")
    logger.info("=" * 80)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, train_dataset = create_dataloader(args, training=True)
    val_loader, val_dataset = create_dataloader(args, training=False)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(args)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    
    logger.info(f"Optimizer: AdamW(lr={args.learning_rate}, weight_decay={args.weight_decay})")
    logger.info(f"Scheduler: {args.scheduler}")
    
    # Setup TensorBoard
    tensorboard_dir = Path(args.log_dir) / args.exp_name
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard logs: {tensorboard_dir}")
    
    # Resume training if requested
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        start_epoch, _, best_loss = load_checkpoint(model, optimizer, scheduler, args.resume, logger)
    elif args.auto_resume:
        latest_checkpoint = Path(args.checkpoint_dir) / 'latest_checkpoint.pth'
        if latest_checkpoint.exists():
            start_epoch, _, best_loss = load_checkpoint(model, optimizer, scheduler, 
                                                      str(latest_checkpoint), logger)
    
    # Training loop
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Training
        avg_train_loss, train_losses = train_epoch(
            model, train_loader, optimizer, scheduler, epoch, logger, writer, args
        )
        
        # Validation
        avg_val_loss = None
        if (epoch + 1) % args.val_interval == 0:
            avg_val_loss = validate(model, val_loader, logger, writer, epoch)
            
            # Update best loss
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                # Save best checkpoint
                save_checkpoint(model, optimizer, scheduler, epoch, 0, best_loss, args, 'best_checkpoint.pth')
                logger.info(f"New best validation loss: {best_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = save_checkpoint(model, optimizer, scheduler, epoch, 0, best_loss, args)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch:3d} completed in {epoch_time:.1f}s")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        if avg_val_loss is not None:
            logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  Best Loss: {best_loss:.4f}")
        logger.info("-" * 80)
    
    # Final checkpoint
    final_checkpoint = save_checkpoint(model, optimizer, scheduler, args.epochs - 1, 0, best_loss, args, 'final_checkpoint.pth')
    logger.info(f"Training completed! Final checkpoint: {final_checkpoint}")
    
    # Close writer
    writer.close()


if __name__ == '__main__':
    main()