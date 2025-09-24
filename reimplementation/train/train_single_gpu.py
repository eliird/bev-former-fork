#!/usr/bin/env python3
"""
BEVFormer Single GPU Training Script
Uses BEVFormerTrainer class for clean and maintainable training
"""

import os
import sys
import argparse
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.config_parser import load_config
from trainer import BEVFormerTrainer, setup_logging, setup_device


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='BEVFormer Single GPU Training')
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
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda:0, cpu, etc.). Auto-detect if not specified.')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Extract experiment name
    exp_name = args.exp_name or config.get('experiment', {}).get('name', 'bevformer_single_gpu')

    # Setup directories
    log_dir = os.path.join(args.log_dir, exp_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, exp_name)

    # Setup device
    device = setup_device(args.device)

    # Setup logging
    logger = setup_logging(log_dir, exp_name, rank=0)

    logger.info("=" * 80)
    logger.info("BEVFORMER SINGLE GPU TRAINING")
    logger.info("=" * 80)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Log device information
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Memory: {memory_gb:.1f} GB")

    # Create trainer
    trainer = BEVFormerTrainer(
        config=config,
        device=device,
        logger=logger,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir
    )

    try:
        # Setup training components
        logger.info("Setting up training components...")

        model = trainer.setup_model()
        train_loader, val_loader = trainer.setup_data()
        optimizer, scheduler = trainer.setup_optimizer_scheduler(model)
        writer = trainer.setup_tensorboard()

        # Handle resume
        start_epoch = 0
        if args.resume and os.path.exists(args.resume):
            # Use explicit resume path
            start_epoch = trainer.load_checkpoint(args.resume)
        else:
            # Try auto-resume from latest checkpoint
            latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            if os.path.exists(latest_checkpoint_path):
                start_epoch = trainer.load_checkpoint(latest_checkpoint_path)

        # Run training
        epochs = config.get('training', {}).get('epochs', 24)
        trainer.train(start_epoch=start_epoch, end_epoch=epochs)

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Clean up resources
        trainer.cleanup()


if __name__ == '__main__':
    main()