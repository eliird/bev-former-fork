"""Training utilities for BEVFormer"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import torch


def setup_logging(log_dir: str, exp_name: str, rank: int = 0) -> logging.Logger:
    """Setup enhanced logging configuration with multiple file handlers.

    Args:
        log_dir: Directory for log files
        exp_name: Experiment name
        rank: Process rank for distributed training

    Returns:
        Configured logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger_name = 'BEVFormer' if rank == 0 else f'BEVFormer_rank{rank}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Only setup handlers for rank 0 in distributed training
    if rank == 0:
        # Console handler - clean format for readability
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # Main log file handler - detailed format
        main_log_file = log_dir / f'{exp_name}.log'
        file_handler = logging.FileHandler(main_log_file)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # Training metrics file handler - for performance data
        metrics_log_file = log_dir / f'{exp_name}_metrics.log'
        metrics_handler = logging.FileHandler(metrics_log_file)
        metrics_handler.setLevel(logging.INFO)
        # Create a custom logger for metrics
        metrics_logger = logging.getLogger(f'{logger_name}_metrics')
        metrics_logger.setLevel(logging.INFO)
        metrics_logger.addHandler(metrics_handler)

        # Store reference to metrics logger
        logger.metrics_logger = metrics_logger

    return logger


def setup_device(device: Optional[Union[str, torch.device]] = None, rank: Optional[int] = None) -> torch.device:
    """Setup compute device for training.

    Args:
        device: Specific device to use (optional)
        rank: GPU rank for distributed training (optional)

    Returns:
        Configured device
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        if rank is not None:
            # Distributed training - use specific GPU
            device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(device)
        else:
            # Single GPU training - use first available GPU
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    return device


def get_device_info(device: torch.device, logger: logging.Logger) -> None:
    """Log device information.

    Args:
        device: Compute device
        logger: Logger instance
    """
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        gpu_id = device.index if device.index is not None else 0
        logger.info(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
        memory_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
        logger.info(f"GPU Memory: {memory_gb:.1f} GB")


def count_parameters(model: torch.nn.Module, logger: logging.Logger) -> None:
    """Log model parameter counts.

    Args:
        model: PyTorch model
        logger: Logger instance
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")


def cleanup_distributed() -> None:
    """Clean up distributed training resources."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process in distributed training."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size() -> int:
    """Get world size for distributed training."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string (e.g., "2h 15m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def format_detailed_losses(losses: dict, indent: str = "    ") -> str:
    """Format detailed losses in a hierarchical, readable format.

    Args:
        losses: Dictionary of loss values
        indent: Indentation string for formatting

    Returns:
        Formatted loss string
    """
    if not losses:
        return "No losses to display"

    # Separate primary losses from decoder losses
    primary_losses = {}
    decoder_losses = {}

    for key, value in losses.items():
        if torch.is_tensor(value):
            value = value.item()

        if key.startswith('d') and '.' in key:
            # Decoder loss (e.g., 'd0.loss_cls')
            decoder_losses[key] = value
        else:
            # Primary loss
            primary_losses[key] = value

    lines = ["Loss Components:"]

    # Format primary losses
    if primary_losses:
        lines.append(f"{indent}├── Primary Losses")
        primary_items = list(primary_losses.items())
        for i, (key, value) in enumerate(primary_items):
            is_last = (i == len(primary_items) - 1) and not decoder_losses
            prefix = "└──" if is_last else "├──"
            loss_name = key.replace('loss_', '').replace('_', ' ').title()
            lines.append(f"{indent}│   {prefix} {loss_name}: {value:.4f}")

    # Format decoder losses
    if decoder_losses:
        lines.append(f"{indent}└── Decoder Losses")
        # Group by decoder level
        decoder_groups = {}
        for key, value in decoder_losses.items():
            decoder_id, loss_type = key.split('.', 1)
            if decoder_id not in decoder_groups:
                decoder_groups[decoder_id] = {}
            decoder_groups[decoder_id][loss_type] = value

        decoder_items = list(decoder_groups.items())
        for i, (decoder_id, losses_dict) in enumerate(decoder_items):
            is_last = (i == len(decoder_items) - 1)
            prefix = "└──" if is_last else "├──"
            loss_strs = [f"{k.replace('loss_', '').title()}: {v:.4f}"
                        for k, v in losses_dict.items()]
            lines.append(f"{indent}    {prefix} {decoder_id.upper()} - {', '.join(loss_strs)}")

    return '\n'.join(lines)


def calculate_training_metrics(batch_size: int, iteration_time: float, queue_length: int = 1) -> dict:
    """Calculate training speed metrics.

    Args:
        batch_size: Number of samples per batch
        iteration_time: Time taken for one iteration in seconds
        queue_length: Number of images per sample (temporal queue)

    Returns:
        Dictionary with training metrics
    """
    if iteration_time <= 0:
        return {
            'samples_per_sec': 0.0,
            'images_per_sec': 0.0,
            'time_per_iteration': 0.0,
            'iterations_per_sec': 0.0
        }

    samples_per_sec = batch_size / iteration_time
    images_per_sec = samples_per_sec * queue_length
    iterations_per_sec = 1.0 / iteration_time

    return {
        'samples_per_sec': samples_per_sec,
        'images_per_sec': images_per_sec,
        'time_per_iteration': iteration_time,
        'iterations_per_sec': iterations_per_sec
    }