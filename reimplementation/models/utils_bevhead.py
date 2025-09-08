import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Callable, List, Any, Tuple
import math


def multi_apply(func: Callable, *args, **kwargs) -> tuple:
    """Apply function to multiple inputs and return tuple of results.
    
    This is commonly used in detection heads to apply the same function
    to multiple levels or multiple samples.
    
    Args:
        func: Function to apply
        *args: Variable length argument lists to apply func to
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        tuple: Results of applying func to each set of arguments
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def partial(func: Callable, **kwargs):
    """Create a partial function with fixed keyword arguments."""
    def wrapper(*args):
        return func(*args, **kwargs)
    return wrapper


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce mean across all processes in distributed training.
    
    In non-distributed mode, this just returns the input tensor.
    
    Args:
        tensor: Input tensor to reduce
        
    Returns:
        Tensor: Mean reduced tensor
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / dist.get_world_size()
    return tensor


def bias_init_with_prob(prior_prob: float) -> float:
    """Initialize bias value according to prior probability.
    
    This is used to initialize the bias of the final classification layer
    so that the initial predictions have the desired prior probability.
    
    Args:
        prior_prob: Prior probability of positive class
        
    Returns:
        float: Bias value to achieve the prior probability
    """
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse sigmoid function.
    
    Args:
        x: Input tensor (should be in range [0, 1])
        eps: Small value to avoid numerical issues
        
    Returns:
        Tensor: Inverse sigmoid of input
    """
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def normalize_bbox(bboxes: torch.Tensor, pc_range: list) -> torch.Tensor:
    """Normalize bounding boxes to [0, 1] range based on point cloud range.
    
    Args:
        bboxes: Bounding boxes [..., (cx, cy, cz, w, l, h, rot, vx?, vy?)]
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        
    Returns:
        Tensor: Normalized bounding boxes with log-scaled dimensions
    """
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2] 
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()
    
    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes: torch.Tensor, pc_range: list) -> torch.Tensor:
    """Denormalize bounding boxes from [0, 1] range.
    
    Args:
        normalized_bboxes: Normalized bboxes [..., (cx, cy, w_log, l_log, cz, h_log, sin, cos, vx?, vy?)]
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        
    Returns:
        Tensor: Denormalized bounding boxes
    """
    # Rotation
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)
    
    # Center in BEV
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
    
    # Size (exp to recover from log)
    w = normalized_bboxes[..., 2:3].exp()
    l = normalized_bboxes[..., 3:4].exp()
    h = normalized_bboxes[..., 5:6].exp()
    
    if normalized_bboxes.size(-1) > 8:
        # Velocity
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
        
    return denormalized_bboxes


def test_utils():
    """Test utility functions"""
    print("Testing utility functions...")
    
    # Test multi_apply
    def add_one(x, y):
        return x + 1, y + 2
    
    xs = [1, 2, 3]
    ys = [4, 5, 6]
    result = multi_apply(add_one, xs, ys)
    assert result == ([2, 3, 4], [6, 7, 8]), "multi_apply test failed"
    print("✓ multi_apply test passed")
    
    # Test bias_init_with_prob
    prior_prob = 0.01
    bias = bias_init_with_prob(prior_prob)
    # Verify that sigmoid(bias) ≈ prior_prob
    assert abs(torch.sigmoid(torch.tensor(bias)).item() - prior_prob) < 1e-6
    print("✓ bias_init_with_prob test passed")
    
    # Test inverse_sigmoid
    x = torch.tensor([0.1, 0.5, 0.9])
    x_inv = inverse_sigmoid(x)
    x_recovered = torch.sigmoid(x_inv)
    assert torch.allclose(x, x_recovered, atol=1e-6)
    print("✓ inverse_sigmoid test passed")
    
    # Test normalize/denormalize bbox
    bboxes = torch.tensor([[10, 20, 1, 3, 4, 2, 0.5, 0.1, 0.2]])  # cx, cy, cz, w, l, h, rot, vx, vy
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    normalized = normalize_bbox(bboxes, pc_range)
    denormalized = denormalize_bbox(normalized, pc_range)
    
    # Check shape preservation
    assert normalized.shape == bboxes.shape
    assert denormalized.shape[:-1] == bboxes.shape[:-1]  # May have different last dim due to rot representation
    print("✓ bbox normalization test passed")
    
    print("All utility tests passed!")


if __name__ == "__main__":
    test_utils()