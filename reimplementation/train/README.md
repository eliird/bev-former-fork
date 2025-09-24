# BEVFormer Training

Simple training script for BEVFormer 3D object detection on nuScenes dataset.

## Training

```bash
python train_yaml.py --config configs/bevformer_tiny_clean.yaml --exp-name my_experiment
```

### Arguments

- `--config`: Path to YAML configuration file (required)
- `--exp-name`: Experiment name for logs and checkpoints (optional, uses config default)
- `--resume`: Path to specific checkpoint to resume from (optional)
- `--log-dir`: Directory for logs (default: `./logs`)
- `--checkpoint-dir`: Directory for checkpoints (default: `./checkpoints`)

### Auto-Resume

The training automatically resumes from `latest_checkpoint.pth` if found in the checkpoint directory.

## Configuration

Edit `configs/bevformer_tiny_clean.yaml` to modify training parameters.