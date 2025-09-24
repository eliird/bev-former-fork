# from mmdet.models.losses import GIoULoss
from typing import Optional
import torch.nn as nn
from torch import Tensor
import torch

from .l1_loss import weighted_loss



def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


@weighted_loss
def giou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    # avoid fp16 overflow
    if pred.dtype == torch.float16:
        fp16 = True
        pred = pred.to(torch.float32)
    else:
        fp16 = False

    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)

    if fp16:
        gious = gious.to(torch.float16)

    loss = 1 - gious
    return loss

class GIoULoss(nn.Module):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


def test_giou_loss():
    """Test GIoULoss implementation"""
    print("=" * 60)
    print("Testing GIoULoss")
    print("=" * 60)
    
    try:
        # Test basic GIoU loss
        giou_criterion = GIoULoss(
            eps=1e-6,
            reduction='mean',
            loss_weight=1.0
        )
        
        print("✓ GIoULoss created successfully")
        print(f"  - eps: {giou_criterion.eps}")
        print(f"  - reduction: {giou_criterion.reduction}")
        print(f"  - loss_weight: {giou_criterion.loss_weight}")
        
        # Test inputs - bounding box regression scenario
        batch_size = 8
        
        # Bounding boxes in format [x1, y1, x2, y2]
        # Predicted boxes
        pred = torch.tensor([
            [10.0, 10.0, 50.0, 50.0],  # 40x40 box
            [0.0, 0.0, 20.0, 20.0],    # 20x20 box
            [30.0, 30.0, 70.0, 70.0],  # 40x40 box
            [5.0, 5.0, 15.0, 25.0],    # 10x20 box
            [100.0, 100.0, 120.0, 110.0],  # 20x10 box
            [0.0, 0.0, 10.0, 10.0],    # 10x10 box
            [50.0, 50.0, 60.0, 80.0],  # 10x30 box
            [200.0, 200.0, 220.0, 230.0]  # 20x30 box
        ], requires_grad=True)
        
        # Target boxes (ground truth)
        target = torch.tensor([
            [12.0, 8.0, 48.0, 52.0],   # Slightly offset from pred[0]
            [2.0, 2.0, 18.0, 18.0],    # Slightly smaller than pred[1]
            [25.0, 25.0, 75.0, 75.0],  # Slightly different from pred[2]
            [5.0, 5.0, 15.0, 25.0],    # Identical to pred[3] (perfect match)
            [95.0, 95.0, 125.0, 115.0], # Different from pred[4]
            [5.0, 5.0, 15.0, 15.0],    # Different from pred[5]
            [55.0, 45.0, 65.0, 75.0],  # Different from pred[6]
            [190.0, 190.0, 230.0, 240.0] # Different from pred[7]
        ])
        
        print("✓ Test inputs created")
        print(f"  - pred shape: {pred.shape}")
        print(f"  - target shape: {target.shape}")
        
        # Forward pass
        loss = giou_criterion(pred, target)
        
        print("✓ Forward pass successful")
        print(f"  - loss value: {loss.item():.6f}")
        print(f"  - loss shape: {loss.shape}")
        
        # Test gradient flow
        loss.backward()
        assert pred.grad is not None, "Gradients should exist"
        print("✓ Gradient flow successful")
        
        # Test different reductions
        giou_none = GIoULoss(reduction='none')
        loss_none = giou_none(pred, target)
        print(f"✓ Reduction 'none': shape {loss_none.shape}, values: {loss_none}")
        
        giou_sum = GIoULoss(reduction='sum')
        loss_sum = giou_sum(pred, target)
        print(f"✓ Reduction 'sum': value {loss_sum.item():.6f}")
        
        # Test with weights
        weight = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.3, 2.5])
        loss_weighted = giou_criterion(pred, target, weight=weight)
        print(f"✓ Weighted loss: {loss_weighted.item():.6f}")
        
        # Test loss_weight scaling
        giou_scaled = GIoULoss(loss_weight=2.0)
        loss_scaled = giou_scaled(pred, target)
        expected_scaled = loss * 2.0
        print(f"✓ Loss weight scaling: {loss_scaled.item():.6f} (expected: {expected_scaled.item():.6f})")
        assert torch.allclose(loss_scaled, expected_scaled, atol=1e-5), "Loss weight scaling failed"
        
        # Test reduction override
        loss_override = giou_criterion(pred, target, reduction_override='sum')
        print(f"✓ Reduction override: {loss_override.item():.6f}")
        
        # Test avg_factor
        loss_avg_factor = giou_criterion(pred, target, avg_factor=10.0)
        print(f"✓ Avg factor loss: {loss_avg_factor.item():.6f}")
        
        # Test edge cases
        print("\n--- Testing edge cases ---")
        
        # Perfect predictions (pred == target)
        perfect_pred = target.clone().requires_grad_()
        loss_perfect = giou_criterion(perfect_pred, target)
        print(f"✓ Perfect prediction: {loss_perfect.item():.6f}")
        assert loss_perfect.item() < 1e-5, "Perfect prediction should have near-zero loss"
        
        # Non-overlapping boxes (worst case)
        non_overlap_pred = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [100.0, 100.0, 110.0, 110.0]
        ], requires_grad=True)
        non_overlap_target = torch.tensor([
            [50.0, 50.0, 60.0, 60.0],
            [0.0, 0.0, 10.0, 10.0]
        ])
        loss_non_overlap = giou_criterion(non_overlap_pred, non_overlap_target)
        print(f"✓ Non-overlapping boxes: {loss_non_overlap.item():.6f}")
        
        # All-zero weights
        zero_weights = torch.zeros(batch_size)
        loss_zero_weights = giou_criterion(pred, target, weight=zero_weights)
        print(f"✓ All-zero weights: {loss_zero_weights.item():.6f}")
        assert loss_zero_weights.item() == 0.0, "All-zero weights should produce zero loss"
        
        # Test bbox_overlaps function directly
        print("\n--- Testing bbox_overlaps function ---")
        
        test_boxes1 = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 15.0],
            [20.0, 20.0, 30.0, 30.0]
        ])
        test_boxes2 = torch.tensor([
            [0.0, 0.0, 10.0, 20.0],
            [0.0, 10.0, 10.0, 19.0],
            [10.0, 10.0, 20.0, 20.0]
        ])
        
        # Test IoU
        iou_matrix = bbox_overlaps(test_boxes1, test_boxes2, mode='iou', is_aligned=False)
        print(f"✓ IoU matrix shape: {iou_matrix.shape}")
        print(f"✓ IoU matrix:\n{iou_matrix}")
        
        # Test aligned IoU
        iou_aligned = bbox_overlaps(test_boxes1, test_boxes2, mode='iou', is_aligned=True)
        print(f"✓ Aligned IoU shape: {iou_aligned.shape}")
        print(f"✓ Aligned IoU values: {iou_aligned}")
        
        # Test GIoU
        giou_matrix = bbox_overlaps(test_boxes1, test_boxes2, mode='giou', is_aligned=False)
        print(f"✓ GIoU matrix shape: {giou_matrix.shape}")
        print(f"✓ GIoU matrix:\n{giou_matrix}")
        
        # Test aligned GIoU
        giou_aligned = bbox_overlaps(test_boxes1, test_boxes2, mode='giou', is_aligned=True)
        print(f"✓ Aligned GIoU shape: {giou_aligned.shape}")
        print(f"✓ Aligned GIoU values: {giou_aligned}")
        
        # Test direct giou_loss function
        print("\n--- Testing direct giou_loss function ---")
        
        direct_loss = giou_loss(pred, target)
        print(f"✓ Direct giou_loss (mean): {direct_loss.item():.6f}")
        
        direct_loss_none = giou_loss(pred, target, reduction='none')
        print(f"✓ Direct giou_loss (none): shape {direct_loss_none.shape}")
        
        direct_loss_sum = giou_loss(pred, target, reduction='sum')
        print(f"✓ Direct giou_loss (sum): {direct_loss_sum.item():.6f}")
        
        # Test with weights in direct function
        direct_loss_weighted = giou_loss(pred, target, weight=weight)
        print(f"✓ Direct giou_loss weighted: {direct_loss_weighted.item():.6f}")
        
        # Test FP16 support
        print("\n--- Testing FP16 support ---")
        if torch.cuda.is_available():
            pred_fp16 = pred.clone().cuda().half().requires_grad_()
            target_fp16 = target.clone().cuda().half()
            
            giou_fp16 = GIoULoss().cuda()
            loss_fp16 = giou_fp16(pred_fp16, target_fp16)
            print(f"✓ FP16 CUDA loss: {loss_fp16.item():.6f}")
            
            # Test fp16_clamp function
            test_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0]).half()
            clamped = fp16_clamp(test_tensor, min=0.0, max=1.5)
            print(f"✓ fp16_clamp: {clamped}")
        else:
            # Test CPU FP16 
            pred_fp16_cpu = pred.clone().half().requires_grad_()
            target_fp16_cpu = target.clone().half()
            
            loss_fp16_cpu = giou_criterion(pred_fp16_cpu, target_fp16_cpu)
            print(f"✓ FP16 CPU loss: {loss_fp16_cpu.item():.6f}")
        
        # Test empty tensors
        print("\n--- Testing empty tensors ---")
        empty_pred = torch.empty(0, 4, requires_grad=True)
        empty_target = torch.empty(0, 4)
        
        overlaps_empty = bbox_overlaps(empty_pred, empty_target, is_aligned=True)
        print(f"✓ Empty tensors overlaps shape: {overlaps_empty.shape}")
        
        # Test various box configurations
        print("\n--- Testing various box configurations ---")
        
        # Identical boxes
        identical_pred = torch.tensor([[10.0, 10.0, 20.0, 20.0]], requires_grad=True)
        identical_target = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
        loss_identical = giou_criterion(identical_pred, identical_target)
        print(f"✓ Identical boxes: {loss_identical.item():.6f}")
        
        # Contained boxes (one inside another)
        container_pred = torch.tensor([[0.0, 0.0, 20.0, 20.0]], requires_grad=True)
        contained_target = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        loss_contained = giou_criterion(container_pred, contained_target)
        print(f"✓ Contained boxes: {loss_contained.item():.6f}")
        
        # Adjacent boxes (touching but not overlapping)
        adjacent_pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]], requires_grad=True)
        adjacent_target = torch.tensor([[10.0, 0.0, 20.0, 10.0]])
        loss_adjacent = giou_criterion(adjacent_pred, adjacent_target)
        print(f"✓ Adjacent boxes: {loss_adjacent.item():.6f}")
        
        # Test weight dimension handling
        print("\n--- Testing weight dimension handling ---")
        multi_dim_weight = torch.ones_like(pred)  # Same shape as pred (n, 4)
        loss_multi_dim_weight = giou_criterion(pred, target, weight=multi_dim_weight)
        print(f"✓ Multi-dimensional weight: {loss_multi_dim_weight.item():.6f}")
        
        # Verify GIoU properties
        print("\n--- Verifying GIoU properties ---")
        
        # GIoU should be <= IoU
        test_pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        test_target = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        
        iou_val = bbox_overlaps(test_pred, test_target, mode='iou', is_aligned=True)
        giou_val = bbox_overlaps(test_pred, test_target, mode='giou', is_aligned=True)
        
        print(f"✓ IoU: {iou_val.item():.6f}, GIoU: {giou_val.item():.6f}")
        assert giou_val.item() <= iou_val.item() + 1e-6, "GIoU should be <= IoU"
        
        print("✓ All assertions passed")
        print("🎉 GIoULoss test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_giou_loss()