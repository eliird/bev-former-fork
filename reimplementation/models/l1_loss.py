# from mmdet.models.losses import L1Loss
from collections.abc import Callable
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
import functools



def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss: Tensor,
                       weight: Optional[Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Optional[Tensor], optional): Element-wise weights.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (Optional[float], optional): Average factor when
            computing the mean of losses. Defaults to None.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func: Callable) -> Callable:
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                reduction: str = 'mean',
                avg_factor: Optional[int] = None,
                **kwargs) -> Tensor:
        """
        Args:
            pred (Tensor): The prediction.
            target (Tensor): Target bboxes.
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            reduction (str, optional): Options are "none", "mean" and "sum".
                Defaults to 'mean'.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.

        Returns:
            Tensor: Loss tensor.
        """
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


@weighted_loss
def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss


class L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


def test_l1_loss():
    """Test L1Loss implementation"""
    print("=" * 60)
    print("Testing L1Loss")
    print("=" * 60)
    
    try:
        # Test basic L1 loss
        l1_criterion = L1Loss(
            reduction='mean',
            loss_weight=1.0
        )
        
        print("✓ L1Loss created successfully")
        print(f"  - reduction: {l1_criterion.reduction}")
        print(f"  - loss_weight: {l1_criterion.loss_weight}")
        
        # Test inputs - regression scenario
        batch_size = 4
        num_dims = 8  # e.g., bbox coordinates: [x, y, w, h, z, dx, dy, dz]
        
        # Predictions and targets
        pred = torch.randn(batch_size, num_dims, requires_grad=True)
        target = torch.randn(batch_size, num_dims)
        
        print("✓ Test inputs created")
        print(f"  - pred shape: {pred.shape}")
        print(f"  - target shape: {target.shape}")
        
        # Forward pass
        loss = l1_criterion(pred, target)
        
        print("✓ Forward pass successful")
        print(f"  - loss value: {loss.item():.6f}")
        print(f"  - loss shape: {loss.shape}")
        
        # Test gradient flow
        loss.backward()
        assert pred.grad is not None, "Gradients should exist"
        print("✓ Gradient flow successful")
        
        # Test different reductions
        l1_none = L1Loss(reduction='none')
        loss_none = l1_none(pred, target)
        print(f"✓ Reduction 'none': shape {loss_none.shape}")
        
        l1_sum = L1Loss(reduction='sum')
        loss_sum = l1_sum(pred, target)
        print(f"✓ Reduction 'sum': value {loss_sum.item():.6f}")
        
        # Test with weights
        weight = torch.tensor([1.0, 2.0, 0.5, 1.5])
        # Reshape weight to match pred dimensions for element-wise weighting
        weight_expanded = weight.view(-1, 1).expand_as(pred)
        loss_weighted = l1_criterion(pred, target, weight=weight_expanded)
        print(f"✓ Weighted loss: {loss_weighted.item():.6f}")
        
        # Test with per-sample weights
        per_sample_weight = torch.tensor([1.0, 2.0, 0.5, 1.5])
        loss_per_sample = l1_criterion(pred, target, weight=per_sample_weight.view(-1, 1))
        print(f"✓ Per-sample weighted loss: {loss_per_sample.item():.6f}")
        
        # Test loss_weight scaling
        l1_scaled = L1Loss(loss_weight=2.0)
        loss_scaled = l1_scaled(pred, target)
        expected_scaled = loss * 2.0
        print(f"✓ Loss weight scaling: {loss_scaled.item():.6f} (expected: {expected_scaled.item():.6f})")
        assert torch.allclose(loss_scaled, expected_scaled), "Loss weight scaling failed"
        
        # Test reduction override
        loss_override = l1_criterion(pred, target, reduction_override='sum')
        print(f"✓ Reduction override: {loss_override.item():.6f}")
        
        # Test avg_factor
        loss_avg_factor = l1_criterion(pred, target, avg_factor=10.0)
        print(f"✓ Avg factor loss: {loss_avg_factor.item():.6f}")
        
        # Test edge cases
        print("\n--- Testing edge cases ---")
        
        # Zero target
        zero_target = torch.zeros_like(target)
        loss_zero_target = l1_criterion(pred, zero_target)
        print(f"✓ Zero target: {loss_zero_target.item():.6f}")
        
        # Zero prediction
        zero_pred = torch.zeros_like(pred, requires_grad=True)
        loss_zero_pred = l1_criterion(zero_pred, target)
        print(f"✓ Zero prediction: {loss_zero_pred.item():.6f}")
        
        # Perfect prediction (pred == target)
        perfect_pred = target.clone().requires_grad_()
        loss_perfect = l1_criterion(perfect_pred, target)
        print(f"✓ Perfect prediction: {loss_perfect.item():.6f}")
        assert loss_perfect.item() < 1e-6, "Perfect prediction should have near-zero loss"
        
        # Empty tensors
        empty_pred = torch.randn(0, num_dims, requires_grad=True)
        empty_target = torch.randn(0, num_dims)
        loss_empty = l1_criterion(empty_pred, empty_target)
        print(f"✓ Empty tensors: {loss_empty.item():.6f}")
        
        # Test all-zero weights
        zero_weights = torch.zeros_like(pred)
        loss_zero_weights = l1_criterion(pred, target, weight=zero_weights)
        print(f"✓ All-zero weights: {loss_zero_weights.item():.6f}")
        assert loss_zero_weights.item() == 0.0, "All-zero weights should produce zero loss"
        
        # Test direct l1_loss function
        print("\n--- Testing direct l1_loss function ---")
        
        # Test the decorated l1_loss function directly
        direct_loss = l1_loss(pred, target)
        print(f"✓ Direct l1_loss (mean): {direct_loss.item():.6f}")
        
        direct_loss_none = l1_loss(pred, target, reduction='none')
        print(f"✓ Direct l1_loss (none): shape {direct_loss_none.shape}")
        
        direct_loss_sum = l1_loss(pred, target, reduction='sum')
        print(f"✓ Direct l1_loss (sum): {direct_loss_sum.item():.6f}")
        
        # Test with weights
        direct_loss_weighted = l1_loss(pred, target, weight=weight_expanded)
        print(f"✓ Direct l1_loss weighted: {direct_loss_weighted.item():.6f}")
        
        # Test utility functions
        print("\n--- Testing utility functions ---")
        
        test_tensor = torch.randn(batch_size, num_dims)
        
        # Test reduce_loss
        reduced_mean = reduce_loss(test_tensor, 'mean')
        reduced_sum = reduce_loss(test_tensor, 'sum')
        reduced_none = reduce_loss(test_tensor, 'none')
        
        print(f"✓ reduce_loss mean: {reduced_mean.item():.6f}")
        print(f"✓ reduce_loss sum: {reduced_sum.item():.6f}")
        print(f"✓ reduce_loss none: shape {reduced_none.shape}")
        
        # Test weight_reduce_loss
        test_weight = torch.ones_like(test_tensor)
        weighted_reduced = weight_reduce_loss(test_tensor, test_weight, 'mean')
        print(f"✓ weight_reduce_loss: {weighted_reduced.item():.6f}")
        
        # Test weight_reduce_loss with avg_factor
        avg_factor_reduced = weight_reduce_loss(test_tensor, None, 'mean', avg_factor=5.0)
        print(f"✓ weight_reduce_loss with avg_factor: {avg_factor_reduced.item():.6f}")
        
        # Verify L1 loss computation manually
        manual_l1 = torch.mean(torch.abs(pred - target))
        auto_l1 = l1_criterion(pred, target)
        diff = abs(manual_l1.item() - auto_l1.item())
        print(f"✓ Manual vs automatic L1 difference: {diff:.8f}")
        assert diff < 1e-6, f"Manual and automatic L1 should be equal, got difference: {diff}"
        
        # Test different tensor shapes
        print("\n--- Testing different shapes ---")
        
        # 1D tensors
        pred_1d = torch.randn(10, requires_grad=True)
        target_1d = torch.randn(10)
        loss_1d = l1_criterion(pred_1d, target_1d)
        print(f"✓ 1D tensors: {loss_1d.item():.6f}")
        
        # 3D tensors (e.g., feature maps)
        pred_3d = torch.randn(2, 8, 16, requires_grad=True)
        target_3d = torch.randn(2, 8, 16)
        loss_3d = l1_criterion(pred_3d, target_3d)
        print(f"✓ 3D tensors: {loss_3d.item():.6f}")
        
        # Large tensors
        pred_large = torch.randn(100, 50, requires_grad=True)
        target_large = torch.randn(100, 50)
        loss_large = l1_criterion(pred_large, target_large)
        print(f"✓ Large tensors: {loss_large.item():.6f}")
        
        print("✓ All assertions passed")
        print("🎉 L1Loss test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_l1_loss()
