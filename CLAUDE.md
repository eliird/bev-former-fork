# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a pure PyTorch reimplementation of BEVFormer, a bird's-eye-view (BEV) transformer model for 3D object detection from multi-camera images. The repository removes dependencies on mmdetection/mmcv for easier integration with modern PyTorch versions.

## Key Architecture Components

### Model Structure
- **BEVFormer**: Main model combining backbone, neck, and transformer components for BEV representation learning
- **Spatial Cross-Attention**: Extracts features from multi-view camera images into BEV space
- **Temporal Self-Attention**: Fuses historical BEV features across time frames
- **Detection Head**: Performs 3D object detection using learned BEV queries

### Data Pipeline
- Processes nuScenes dataset with 6 camera views
- Temporal sequences of 4 frames for motion modeling
- Custom data augmentation including GridMask
- CAN bus data integration for ego-motion compensation

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n open-mmlab python=3.10 -y
conda activate open-mmlab

# Install PyTorch (adjust CUDA version as needed)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install dependencies
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13 typing-extensions==4.5.0 pylint ipython==8.12 numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Optional: Install mmcv for custom kernels (not required)
pip install mim
mim install mmengine mmcv
```

### Dataset Preparation
```bash
# Download nuScenes mini dataset
wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz
tar -xzf v1.0-mini.tgz -C data/

# Download CAN bus data
wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/can_bus.zip
unzip can_bus.zip && mv can_bus data/

# Generate annotations
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini --canbus ./data

# Move data to reimplementation directory
mv ./data reimplementation/data
```

### Training Commands
```bash
cd reimplementation

# Quick test (2 epochs, minimal config)
python example_training.py quick_test

# Small training run (5 epochs)
python example_training.py small_training

# Full training (24 epochs, standard BEVFormer)
python example_training.py full_training

# Custom training with specific parameters
python train_bevformer.py --batch_size 2 --learning_rate 1e-4 --epochs 10 --exp_name my_experiment

# Resume from checkpoint
python train_bevformer.py --resume ./checkpoints/checkpoint_epoch_010.pth --auto_resume
```

### Testing Components
```bash
cd reimplementation

# Test individual model components
python test_components.py

# Test data loading pipeline
python dataset/test_dataloader.py

# Test full transformation pipeline
python transforms/complete_pipeline_test.py
```

### Model Evaluation
```bash
# Run evaluation on validation set (from tools directory)
python tools/test.py [config_file] [checkpoint_file] --eval mAP

# Distributed testing
bash tools/dist_test.sh [config_file] [checkpoint_file] [num_gpus]
```

## Important Implementation Details

### Training Script Arguments
The main training script `reimplementation/train_bevformer.py` accepts:
- `--data_root`: Path to nuScenes data
- `--train_pkl` / `--val_pkl`: Pickle files with temporal annotations
- `--batch_size`: Batch size per GPU
- `--epochs`: Number of training epochs
- `--encoder_layers` / `--decoder_layers`: Transformer depth
- `--learning_rate`: Initial learning rate
- `--use_grid_mask`: Enable GridMask augmentation
- `--checkpoint_dir`: Directory to save checkpoints
- `--exp_name`: Experiment name for logging

### Model Configurations
- **BEVFormer-tiny**: 3 encoder/decoder layers, lightweight for testing
- **BEVFormer-small**: 2 encoder/decoder layers, reduced memory usage
- **BEVFormer-base**: 6 encoder/decoder layers, standard configuration

### Key Differences from Original
- Pure PyTorch implementation without mmdetection dependencies
- Simplified configuration system using argparse instead of config files
- Direct checkpoint loading/saving without mmcv wrappers
- Custom data loading pipeline optimized for readability

## User Preferences and Guidelines

### Command Execution
- **DO NOT run commands directly** - always ask the user to run them
- Provide the command and ask for confirmation before execution
- Wait for user to confirm results before proceeding

## Directory Structure
```
bev-former-fork/
├── reimplementation/       # Pure PyTorch implementation
│   ├── models/            # Model components
│   ├── dataset/           # Data loading and processing
│   ├── transforms/        # Data augmentations
│   ├── tools/             # Reimplemented data creation scripts
│   ├── train_bevformer.py # Main training script
│   └── example_training.py # Training examples
├── tools/                 # Dataset creation and utilities
│   ├── create_data.py     # Generate nuScenes annotations
│   ├── train.py          # Original mmdet training (deprecated)
│   └── test.py           # Original mmdet testing (deprecated)
├── docs/                  # Documentation
└── data/                  # Dataset location (after setup)
```