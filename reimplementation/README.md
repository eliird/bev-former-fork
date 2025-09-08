# BEVFormer Reimplementation

Pure PyTorch implementation of BEVFormer without MMDetection dependencies.

## Quick Start

### 1. Download Data
```bash
# Download nuScenes v1.0-mini
wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz
tar -xzf v1.0-mini.tgz -C data/

# Download CAN bus expansion
wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/can_bus.zip
unzip can_bus.zip && mv can_bus data/
```

### 2. Prepare Dataset
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini --canbus ./data
```

### 3. Train
```bash
# Quick test (2 epochs)
python example_training.py quick_test

# Full training
python train_bevformer.py
```

## Features
- ✅ Pure PyTorch (no MMDetection)
- ✅ Temporal modeling with 4-frame sequences  
- ✅ Multi-view camera input (6 cameras)
- ✅ Complete loss functions (FocalLoss + L1Loss + GIoULoss)
- ✅ TensorBoard logging
- ✅ Checkpoint saving/loading