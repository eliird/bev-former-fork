#!/usr/bin/env python3
"""
BEVFormer Multi-GPU Distributed Training Script
Uses BEVFormerTrainer class with DistributedDataParallel for scalable training
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.config_parser import load_config
from trainer import BEVFormerTrainer, setup_logging, setup_device
from trainer.utils import cleanup_distributed, is_main_process, get_rank


class DistributedBEVFormerTrainer(BEVFormerTrainer):
    """BEVFormer trainer with distributed training support."""

    def __init__(self, config, device, logger, log_dir, checkpoint_dir, rank, world_size):
        """Initialize distributed trainer."""
        super().__init__(config, device, logger, log_dir, checkpoint_dir)
        self.rank = rank
        self.world_size = world_size

    def setup_model(self):
        """Create and wrap model with DDP."""
        model = super().setup_model()

        # Wrap model with DDP
        model = DDP(model, device_ids=[self.rank], output_device=self.rank)
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
        num_workers = training_config.get('num_workers', 4)

        from dataset import custom_collate_fn

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=False
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

    def train_epoch(self, epoch):
        """Train one epoch with distributed sampling."""
        # Set epoch for distributed sampler
        self.train_sampler.set_epoch(epoch)
        return super().train_epoch(epoch)

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

        # Create distributed trainer
        trainer = DistributedBEVFormerTrainer(
            config=config,
            device=device,
            logger=logger,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            rank=rank,
            world_size=world_size
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