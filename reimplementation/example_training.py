#!/usr/bin/env python3
"""
Example script demonstrating BEVFormer training usage
Shows different training configurations and command examples
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_training_example(config_name: str):
    """Run a specific training configuration."""
    
    base_cmd = [
        sys.executable, 
        "train_bevformer.py"
    ]
    
    # Different training configurations
    configs = {
        'quick_test': [
            '--data_root', '/home/irdali.durrani/po-pi/BEVFormer/reimplementation/data/nuscenes',
            '--train_pkl', 'nuscenes_infos_temporal_val.pkl',  # Use val as train for quick test
            '--val_pkl', 'nuscenes_infos_temporal_val.pkl',
            '--batch_size', '1',
            '--epochs', '2',
            '--encoder_layers', '1',  # Minimal layers for testing
            '--decoder_layers', '1',
            '--log_interval', '5',
            '--val_interval', '1',
            '--exp_name', 'quick_test'
        ],
        'small_training': [
            '--data_root', '/home/irdali.durrani/po-pi/BEVFormer/reimplementation/data/nuscenes',
            '--train_pkl', 'nuscenes_infos_temporal_train.pkl',
            '--val_pkl', 'nuscenes_infos_temporal_val.pkl',
            '--batch_size', '2',
            '--epochs', '5',
            '--encoder_layers', '2',
            '--decoder_layers', '2',
            '--learning_rate', '1e-4',
            '--log_interval', '10',
            '--val_interval', '2',
            '--exp_name', 'small_training'
        ],
        'full_training': [
            '--data_root', '/home/irdali.durrani/po-pi/BEVFormer/reimplementation/data/nuscenes',
            '--train_pkl', 'nuscenes_infos_temporal_train.pkl',
            '--val_pkl', 'nuscenes_infos_temporal_val.pkl',
            '--batch_size', '4',
            '--epochs', '24',
            '--encoder_layers', '6',
            '--decoder_layers', '6',
            '--learning_rate', '2e-4',
            '--use_grid_mask',
            '--log_interval', '10',
            '--val_interval', '5',
            '--exp_name', 'full_bevformer'
        ]
    }
    
    if config_name not in configs:
        print(f"Unknown config: {config_name}")
        print(f"Available configs: {list(configs.keys())}")
        return False
    
    cmd = base_cmd + configs[config_name]
    
    print("=" * 80)
    print(f"RUNNING TRAINING CONFIG: {config_name}")
    print("=" * 80)
    print("Command:")
    print(" ".join(cmd))
    print("=" * 80)
    
    # Run the training
    try:
        result = subprocess.run(cmd, cwd='/home/irdali.durrani/po-pi/BEVFormer/reimplementation')
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False
    except Exception as e:
        print(f"Error running training: {e}")
        return False


def show_help():
    """Show help information."""
    print("BEVFormer Training Examples")
    print("=" * 50)
    print()
    print("Available configurations:")
    print("  quick_test    - Minimal training for testing (2 epochs)")
    print("  small_training - Small scale training (5 epochs)")  
    print("  full_training - Full BEVFormer training (24 epochs)")
    print()
    print("Usage:")
    print("  python example_training.py quick_test")
    print("  python example_training.py small_training")
    print("  python example_training.py full_training")
    print()
    print("Direct training script usage:")
    print("  python train_bevformer.py --help")
    print()
    print("Custom training examples:")
    print("  # Basic training with default settings")
    print("  python train_bevformer.py")
    print()
    print("  # Custom batch size and learning rate")
    print("  python train_bevformer.py --batch_size 2 --learning_rate 1e-4")
    print()
    print("  # Training with checkpointing")
    print("  python train_bevformer.py --checkpoint_dir ./my_checkpoints --auto_resume")
    print()
    print("  # Resume from specific checkpoint")
    print("  python train_bevformer.py --resume ./checkpoints/checkpoint_epoch_010.pth")


def main():
    parser = argparse.ArgumentParser(description='BEVFormer Training Examples')
    parser.add_argument('config', nargs='?', help='Training configuration name')
    parser.add_argument('--help-examples', action='store_true', help='Show example usage')
    
    args = parser.parse_args()
    
    if args.help_examples or not args.config:
        show_help()
        return
    
    success = run_training_example(args.config)
    
    if success:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed or was interrupted")


if __name__ == '__main__':
    main()