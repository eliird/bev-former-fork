# BEVFormer Training

Training infrastructure for BEVFormer 3D object detection on nuScenes dataset.

## Usage

### Single GPU Training
```bash
python train_single_gpu.py --config configs/bevformer_tiny_clean.yaml
```

### Multi-GPU Training (4 GPUs)
```bash
torchrun --nproc_per_node=4 train_multi_gpu.py --config configs/bevformer_tiny_clean.yaml
```

### With Custom Experiment Name
```bash
python train_single_gpu.py --config configs/bevformer_tiny_clean.yaml --exp-name my_experiment
```

### Resume Training
```bash
python train_single_gpu.py --config configs/bevformer_tiny_clean.yaml --resume checkpoints/my_exp/best_model.pth
```

## Arguments

- `--config`: Path to YAML configuration file (required)
- `--exp-name`: Experiment name for logs and checkpoints (optional)
- `--resume`: Resume from specific checkpoint (optional)
- `--log-dir`: Directory for logs (default: `./logs`)
- `--checkpoint-dir`: Directory for checkpoints (default: `./checkpoints`)
- `--device`: Device to use, e.g., `cuda:0`, `cpu` (single GPU only, optional)

## Features

- NDS and mAP metrics calculation
- TensorBoard logging
- Auto-resume from latest checkpoint
- Sequential scene processing for proper temporal evaluation
- Memory-efficient validation with GPU cache management

## Configuration

Edit `configs/bevformer_tiny_clean.yaml` to modify training parameters.