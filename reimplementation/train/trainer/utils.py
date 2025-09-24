"""Training utilities for BEVFormer"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import torch


def setup_logging(log_dir: str, exp_name: str, rank: int = 0) -> logging.Logger:
    """Setup logging configuration.

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
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

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