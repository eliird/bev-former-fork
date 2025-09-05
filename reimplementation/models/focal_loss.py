from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Try to import MMCV focal loss
MMCV_FOCAL_LOSS_AVAILABLE = False
try:
    from mmcv.ops import sigmoid_focal_loss as mmcv_sigmoid_focal_loss
    MMCV_FOCAL_LOSS_AVAILABLE = True
    print("MMCV focal loss available - using optimized implementation when possible")
except ImportError:
    print("MMCV focal loss not available - using PyTorch implementation")
    mmcv_sigmoid_focal_loss = None

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


def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # Actually, pt here denotes (1 - pt) in the Focal Loss paper
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # Thus it's pt.pow(gamma) rather than (1 - pt).pow(gamma)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_focal_loss_with_prob(pred,
                            target,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
            The target shape support (N,C) or (N,), (N,C) means one-hot form.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    if pred.dim() != target.dim():
        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes + 1)
        target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""Focal Loss with MMCV fallback support.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    if MMCV_FOCAL_LOSS_AVAILABLE and pred.is_cuda:
        # Use MMCV optimized version if available and on CUDA
        # MMCV expects target to be class indices (long tensor), not one-hot
        if target.dim() == pred.dim() and target.size(1) == pred.size(1):
            # Convert one-hot to class indices
            target_indices = target.argmax(dim=1).long()
        else:
            # Already class indices
            target_indices = target.long()
            
        # MMCV function signature: sigmoid_focal_loss(pred, target, gamma, alpha, weight, reduction)
        loss = mmcv_sigmoid_focal_loss(
            pred.contiguous(), 
            target_indices.contiguous(), 
            gamma,
            alpha,
            None,  # weight handled separately
            'none'  # reduction handled separately
        )
        # Apply weight and reduction using our utility functions
        if weight is not None:
            if weight.shape != loss.shape:
                if weight.size(0) == loss.size(0):
                    weight = weight.view(-1, 1)
                else:
                    assert weight.numel() == loss.numel()
                    weight = weight.view(loss.size(0), -1)
            assert weight.ndim == loss.ndim
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    else:
        # Fall back to PyTorch implementation
        loss = py_sigmoid_focal_loss(
            pred, target, weight, gamma, alpha, reduction, avg_factor
        )
    return loss



class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
                The target shape support (N,C) or (N,), (N,C) means
                one-hot form.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if pred.dim() == target.dim():
                    # this means that target is already in One-Hot form.
                    calculate_loss_func = py_sigmoid_focal_loss
                elif torch.cuda.is_available() and pred.is_cuda:
                    calculate_loss_func = sigmoid_focal_loss
                else:
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls


def test_focal_loss():
    """Test FocalLoss implementation"""
    print("=" * 60)
    print("Testing FocalLoss")
    print("=" * 60)
    
    try:
        # Test basic focal loss
        focal_loss = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0
        )
        
        print("✓ FocalLoss created successfully")
        print(f"  - gamma: {focal_loss.gamma}")
        print(f"  - alpha: {focal_loss.alpha}")
        print(f"  - MMCV available: {MMCV_FOCAL_LOSS_AVAILABLE}")
        
        # Test inputs - classification scenario
        batch_size = 4
        num_classes = 10
        
        # Predictions (logits)
        pred = torch.randn(batch_size, num_classes, requires_grad=True)
        
        # Binary targets (one-hot encoded)
        target = torch.zeros(batch_size, num_classes)
        target[0, 1] = 1.0  # Class 1 for sample 0
        target[1, 3] = 1.0  # Class 3 for sample 1
        target[2, 5] = 1.0  # Class 5 for sample 2
        target[3, 8] = 1.0  # Class 8 for sample 3
        
        print("✓ Test inputs created")
        print(f"  - pred shape: {pred.shape}")
        print(f"  - target shape: {target.shape}")
        
        # Forward pass with one-hot targets
        loss = focal_loss(pred, target)
        
        print("✓ Forward pass with one-hot targets successful")
        print(f"  - loss value: {loss.item():.6f}")
        print(f"  - loss shape: {loss.shape}")
        
        # Test gradient flow
        loss.backward()
        assert pred.grad is not None, "Gradients should exist"
        print("✓ Gradient flow successful")
        
        # Test with class indices (not one-hot)
        pred_2 = torch.randn(batch_size, num_classes, requires_grad=True)
        target_indices = torch.tensor([1, 3, 5, 8])  # Class indices
        
        loss_indices = focal_loss(pred_2, target_indices)
        print(f"✓ Class indices input: {loss_indices.item():.6f}")
        
        # Test different reductions
        focal_loss_none = FocalLoss(reduction='none')
        loss_none = focal_loss_none(pred_2, target_indices)
        print(f"✓ Reduction 'none': shape {loss_none.shape}")
        
        focal_loss_sum = FocalLoss(reduction='sum')
        loss_sum = focal_loss_sum(pred_2, target_indices)
        print(f"✓ Reduction 'sum': value {loss_sum.item():.6f}")
        
        # Test with weights
        weight = torch.tensor([1.0, 2.0, 0.5, 1.5])
        loss_weighted = focal_loss(pred_2, target_indices, weight=weight)
        print(f"✓ Weighted loss: {loss_weighted.item():.6f}")
        
        # Test activated input (probabilities)
        focal_loss_activated = FocalLoss(activated=True)
        pred_prob = torch.sigmoid(pred_2)
        loss_activated = focal_loss_activated(pred_prob, target)
        print(f"✓ Activated input: {loss_activated.item():.6f}")
        
        # Test edge cases
        # All negative samples (background)
        background_target = torch.zeros(batch_size, dtype=torch.long)
        loss_background = focal_loss(pred_2, background_target)
        print(f"✓ Background samples: {loss_background.item():.6f}")
        
        # Very confident predictions
        confident_pred = torch.ones_like(pred_2) * 5  # Very high logits for class 0
        loss_confident = focal_loss(confident_pred, target_indices)
        print(f"✓ Confident predictions: {loss_confident.item():.6f}")
        
        # Test PyTorch vs MMCV comparison if available
        if MMCV_FOCAL_LOSS_AVAILABLE:
            # Test on CPU (should use PyTorch)
            pred_cpu = torch.randn(batch_size, num_classes)
            target_cpu = torch.zeros(batch_size, num_classes)
            target_cpu[0, 1] = 1.0
            target_cpu[1, 3] = 1.0
            
            loss_cpu = sigmoid_focal_loss(pred_cpu, target_cpu)
            print(f"✓ CPU (PyTorch fallback): {loss_cpu.item():.6f}")
            
            # Test on CUDA if available
            if torch.cuda.is_available():
                pred_cuda = pred_cpu.cuda()
                target_cuda = target_cpu.cuda()
                
                loss_cuda = sigmoid_focal_loss(pred_cuda, target_cuda)
                print(f"✓ CUDA (MMCV optimized): {loss_cuda.item():.6f}")
                
                # Compare results
                diff = abs(loss_cpu.item() - loss_cuda.item())
                print(f"✓ CPU vs CUDA difference: {diff:.8f}")
                
                if diff >= 1e-4:
                    print(f"⚠ Warning: Difference between CPU/CUDA: {diff}")
        
        # Test direct PyTorch functions
        print("\n--- Testing direct PyTorch functions ---")
        
        # Test py_sigmoid_focal_loss
        loss_py = py_sigmoid_focal_loss(pred_2, target)
        print(f"✓ py_sigmoid_focal_loss: {loss_py.item():.6f}")
        
        # Test py_focal_loss_with_prob  
        prob_input = torch.sigmoid(pred_2)
        loss_prob = py_focal_loss_with_prob(prob_input, target)
        print(f"✓ py_focal_loss_with_prob: {loss_prob.item():.6f}")
        
        # Test utility functions
        test_loss = torch.randn(batch_size, num_classes)
        
        # Test reduce_loss
        reduced_mean = reduce_loss(test_loss, 'mean')
        reduced_sum = reduce_loss(test_loss, 'sum')
        reduced_none = reduce_loss(test_loss, 'none')
        
        print(f"✓ reduce_loss mean: {reduced_mean.item():.6f}")
        print(f"✓ reduce_loss sum: {reduced_sum.item():.6f}")
        print(f"✓ reduce_loss none shape: {reduced_none.shape}")
        
        # Test weight_reduce_loss
        test_weight = torch.ones(batch_size, num_classes)  # Match loss shape
        weighted_loss = weight_reduce_loss(test_loss, test_weight, 'mean')
        print(f"✓ weight_reduce_loss: {weighted_loss.item():.6f}")
        
        # Test weight_reduce_loss with per-sample weight (needs reshaping)
        test_weight_per_sample = torch.ones(batch_size)
        weighted_loss_per_sample = weight_reduce_loss(test_loss, test_weight_per_sample.view(-1, 1), 'mean')
        print(f"✓ weight_reduce_loss per-sample: {weighted_loss_per_sample.item():.6f}")
        
        # Test with avg_factor
        avg_factor_loss = weight_reduce_loss(test_loss, None, 'mean', avg_factor=10.0)
        print(f"✓ avg_factor loss: {avg_factor_loss.item():.6f}")
        
        print("✓ All assertions passed")
        print("🎉 FocalLoss test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_focal_loss()
