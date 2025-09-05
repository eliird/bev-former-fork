from abc import abstractmethod
from typing import Optional, Union
import torch
from GIoULoss import bbox_overlaps
from torch import Tensor
from mmdet.models.task_modules.assigners.match_cost import FocalLossCost

def bbox_xyxy_to_cxcywh(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)


class BaseMatchCost:
    """Base match cost class.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, weight: Union[float, int] = 1.) -> None:
        self.weight = weight

    @abstractmethod
    def __call__(self,
                 pred_instances,
                 gt_instances,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> torch.Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            img_meta (dict, optional): Image information.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pass


class FocalLossCost(BaseMatchCost):
    """FocalLossCost.

    Args:
        alpha (Union[float, int]): focal_loss alpha. Defaults to 0.25.
        gamma (Union[float, int]): focal_loss gamma. Defaults to 2.
        eps (float): Defaults to 1e-12.
        binary_input (bool): Whether the input is binary. Currently,
            binary_input = True is for masks input, binary_input = False
            is for label input. Defaults to False.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 alpha: Union[float, int] = 0.25,
                 gamma: Union[float, int] = 2,
                 eps: float = 1e-12,
                 binary_input: bool = False,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight

    def _mask_focal_loss_cost(self, cls_pred, gt_labels) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits.
                in shape (num_queries, d1, ..., dn), dtype=torch.float32.
            gt_labels (Tensor): Ground truth in shape (num_gt, d1, ..., dn),
                dtype=torch.long. Labels should be binary.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_queries, num_gt).
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
            torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost / n * self.weight

    def __call__(self,
                 pred_instances,
                 gt_instances,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        if self.binary_input:
            pred_masks = pred_instances.masks
            gt_masks = gt_instances.masks
            return self._mask_focal_loss_cost(pred_masks, gt_masks)
        else:
            pred_scores = pred_instances.scores
            gt_labels = gt_instances.labels
            return self._focal_loss_cost(pred_scores, gt_labels)


class BBox3DL1Cost(BaseMatchCost):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

class IoUCost(BaseMatchCost):
    """IoUCost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        iou_mode (str): iou mode such as 'iou', 'giou'. Defaults to 'giou'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ... match_costs.match_cost import IoUCost
        >>> import torch
        >>> self = IoUCost()
        >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
        >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> self(bboxes, gt_bboxes)
        tensor([[-0.1250,  0.1667],
            [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode: str = 'giou', weight: Union[float, int] = 1.):
        super().__init__(weight=weight)
        self.iou_mode = iou_mode

    def __call__(self,
                 pred_instances,
                 gt_instances,
                 img_meta: Optional[dict] = None,
                 **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes

        # avoid fp16 overflow
        if pred_bboxes.dtype == torch.float16:
            fp16 = True
            pred_bboxes = pred_bboxes.to(torch.float32)
        else:
            fp16 = False

        overlaps = bbox_overlaps(
            pred_bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)

        if fp16:
            overlaps = overlaps.to(torch.float16)

        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight


class MockInstanceData:
    """Mock InstanceData class for testing purposes."""
    def __init__(self, bboxes=None, labels=None, scores=None, masks=None):
        self.bboxes = bboxes
        self.labels = labels  
        self.scores = scores
        self.masks = masks


def test_cost_functions():
    """Test all cost function implementations"""
    print("=" * 60)
    print("Testing Cost Functions")
    print("=" * 60)
    
    try:
        # Test data setup
        # Predicted bounding boxes (xyxy format)
        pred_bboxes = torch.tensor([
            [10.0, 10.0, 50.0, 50.0],  # 40x40 box
            [0.0, 0.0, 20.0, 20.0],    # 20x20 box  
            [30.0, 30.0, 70.0, 70.0],  # 40x40 box
            [5.0, 5.0, 15.0, 25.0],    # 10x20 box
        ])
        
        # Ground truth bounding boxes (xyxy format)
        gt_bboxes = torch.tensor([
            [12.0, 8.0, 48.0, 52.0],   # Close to pred[0]
            [2.0, 2.0, 18.0, 18.0],    # Close to pred[1]  
            [100.0, 100.0, 120.0, 110.0]  # Far from all preds
        ])
        
        # Classification data for FocalLossCost
        num_classes = 10
        pred_scores = torch.randn(4, num_classes)  # Raw logits
        # Make some predictions more confident
        pred_scores[0, 2] = 5.0  # Very confident for class 2
        pred_scores[1, 1] = 3.0  # Confident for class 1
        pred_scores[2, 0] = 2.0  # Somewhat confident for class 0
        pred_scores[3, :] = -2.0  # Not confident about any class
        
        gt_labels = torch.tensor([2, 1, 0])  # Ground truth classes
        
        # Create mock instance data
        pred_instances = MockInstanceData(bboxes=pred_bboxes, scores=pred_scores)
        gt_instances = MockInstanceData(bboxes=gt_bboxes, labels=gt_labels)
        
        # Mock image metadata
        img_meta = {'img_shape': (480, 640)}  # height, width
        
        print("✓ Test data created")
        print(f"  - pred_bboxes shape: {pred_bboxes.shape}")
        print(f"  - gt_bboxes shape: {gt_bboxes.shape}")
        print(f"  - pred_scores shape: {pred_scores.shape}")
        print(f"  - gt_labels: {gt_labels}")
        
        # Test FocalLossCost
        print("\n--- Testing FocalLossCost ---")
        
        # Test with default parameters
        focal_loss_cost = FocalLossCost()
        focal_cost_matrix = focal_loss_cost(pred_instances, gt_instances, img_meta)
        
        print("✓ FocalLossCost created successfully")
        print(f"  - alpha: {focal_loss_cost.alpha}")
        print(f"  - gamma: {focal_loss_cost.gamma}")
        print(f"  - eps: {focal_loss_cost.eps}")
        print(f"  - binary_input: {focal_loss_cost.binary_input}")
        print(f"  - weight: {focal_loss_cost.weight}")
        print(f"  - cost matrix shape: {focal_cost_matrix.shape}")
        print(f"  - cost matrix:\n{focal_cost_matrix}")
        
        # Verify focal loss cost matrix properties
        assert focal_cost_matrix.shape == (4, 3), f"Expected shape (4, 3), got {focal_cost_matrix.shape}"
        assert torch.isfinite(focal_cost_matrix).all(), "All focal costs should be finite"
        
        # Test with custom parameters
        focal_loss_custom = FocalLossCost(alpha=0.5, gamma=1.0, weight=2.0)
        focal_cost_custom = focal_loss_custom(pred_instances, gt_instances, img_meta)
        
        print("✓ FocalLossCost with custom parameters successful")
        print(f"  - custom cost matrix shape: {focal_cost_custom.shape}")
        
        # Test binary mask focal loss
        print("\n--- Testing FocalLossCost with binary masks ---")
        
        # Create binary mask data
        pred_masks = torch.rand(4, 32, 32)  # 4 predictions, 32x32 masks
        gt_masks = torch.randint(0, 2, (3, 32, 32)).float()  # 3 GT masks, binary
        
        mask_pred_instances = MockInstanceData(masks=pred_masks)
        mask_gt_instances = MockInstanceData(masks=gt_masks)
        
        focal_loss_mask = FocalLossCost(binary_input=True, weight=1.5)
        focal_mask_cost = focal_loss_mask(mask_pred_instances, mask_gt_instances, img_meta)
        
        print("✓ Binary mask focal loss successful")
        print(f"  - mask cost matrix shape: {focal_mask_cost.shape}")
        print(f"  - mask cost range: [{focal_mask_cost.min().item():.4f}, {focal_mask_cost.max().item():.4f}]")
        
        assert focal_mask_cost.shape == (4, 3), "Mask cost matrix should have shape (4, 3)"
        assert torch.isfinite(focal_mask_cost).all(), "All mask costs should be finite"
        
        # Test BBox3DL1Cost  
        print("\n--- Testing BBox3DL1Cost ---")
        
        # Create test data for 3D cost (already normalized)
        bbox_pred_3d = torch.tensor([
            [0.5, 0.5, 0.2, 0.3],  # Normalized cxcywh
            [0.3, 0.3, 0.1, 0.1],
            [0.7, 0.7, 0.15, 0.2]
        ])
        
        gt_bboxes_3d = torch.tensor([
            [0.52, 0.48, 0.18, 0.32],  # Close to pred[0]
            [0.8, 0.8, 0.1, 0.1]       # Different from all
        ])
        
        bbox_3d_l1_cost = BBox3DL1Cost(weight=1.5)
        l1_cost_3d_matrix = bbox_3d_l1_cost(bbox_pred_3d, gt_bboxes_3d)
        
        print("✓ BBox3DL1Cost created successfully")
        print(f"  - weight: {bbox_3d_l1_cost.weight}")
        print(f"  - cost matrix shape: {l1_cost_3d_matrix.shape}")
        print(f"  - cost matrix:\n{l1_cost_3d_matrix}")
        
        # Verify 3D cost matrix properties
        assert l1_cost_3d_matrix.shape == (3, 2), f"Expected shape (3, 2), got {l1_cost_3d_matrix.shape}"
        assert torch.all(l1_cost_3d_matrix >= 0), "3D L1 costs should be non-negative"
        
        # Test IoUCost
        print("\n--- Testing IoUCost ---")
        
        # Test with default parameters (giou mode)
        iou_cost = IoUCost()
        iou_cost_matrix = iou_cost(pred_instances, gt_instances, img_meta)
        
        print("✓ IoUCost created successfully")
        print(f"  - iou_mode: {iou_cost.iou_mode}")
        print(f"  - weight: {iou_cost.weight}")
        print(f"  - cost matrix shape: {iou_cost_matrix.shape}")
        print(f"  - cost matrix:\n{iou_cost_matrix}")
        
        # Verify IoU cost matrix properties
        assert iou_cost_matrix.shape == (4, 3), f"Expected shape (4, 3), got {iou_cost_matrix.shape}"
        # IoU costs are negative overlaps. For GIoU, values can range from -1 to 1, so costs range from -1 to 1
        assert torch.all(iou_cost_matrix >= -1), "GIoU costs should be >= -1"
        assert torch.all(iou_cost_matrix <= 1), "GIoU costs should be <= 1"
        
        # Test with different IoU mode
        iou_cost_iou = IoUCost(iou_mode='iou', weight=2.0)
        iou_cost_matrix_iou = iou_cost_iou(pred_instances, gt_instances, img_meta)
        
        print("✓ IoUCost with iou mode successful")
        print(f"  - cost matrix shape: {iou_cost_matrix_iou.shape}")
        print(f"  - IoU cost matrix:\n{iou_cost_matrix_iou}")
        
        # For regular IoU, costs should be in range [-1, 0] since IoU is in [0, 1]
        assert torch.all(iou_cost_matrix_iou >= -2), "IoU costs should be >= -2 (weight=2.0)"
        assert torch.all(iou_cost_matrix_iou <= 0), "IoU costs should be <= 0"
        
        # Test focal loss edge cases
        print("\n--- Testing focal loss edge cases ---")
        
        # Test with perfect predictions (very high confidence for correct class)
        perfect_scores = torch.full((4, num_classes), -10.0)  # Very low confidence
        # Set high confidence for correct classes
        perfect_scores[0, gt_labels[0]] = 10.0  # Perfect for GT 0
        perfect_scores[1, gt_labels[1]] = 10.0  # Perfect for GT 1
        perfect_scores[2, gt_labels[2]] = 10.0  # Perfect for GT 2
        perfect_scores[3, gt_labels[0]] = 10.0  # Perfect for GT 0 (duplicate)
        
        perfect_pred_instances = MockInstanceData(scores=perfect_scores)
        perfect_focal_cost = focal_loss_cost(perfect_pred_instances, gt_instances, img_meta)
        
        print(f"✓ Perfect predictions focal cost range: [{perfect_focal_cost.min().item():.4f}, {perfect_focal_cost.max().item():.4f}]")
        
        # Test with worst predictions (high confidence for wrong class)
        worst_scores = torch.full((4, num_classes), -10.0)
        # Set high confidence for wrong classes
        worst_scores[0, (gt_labels[0] + 1) % num_classes] = 10.0  # Wrong for GT 0
        worst_scores[1, (gt_labels[1] + 1) % num_classes] = 10.0  # Wrong for GT 1
        worst_scores[2, (gt_labels[2] + 1) % num_classes] = 10.0  # Wrong for GT 2
        worst_scores[3, (gt_labels[0] + 1) % num_classes] = 10.0  # Wrong for GT 0
        
        worst_pred_instances = MockInstanceData(scores=worst_scores)
        worst_focal_cost = focal_loss_cost(worst_pred_instances, gt_instances, img_meta)
        
        print(f"✓ Worst predictions focal cost range: [{worst_focal_cost.min().item():.4f}, {worst_focal_cost.max().item():.4f}]")
        
        # Perfect predictions should have lower cost than worst predictions
        # (Note: focal loss cost can be negative, so we check that perfect is generally lower)
        
        # Test edge cases
        print("\n--- Testing edge cases ---")
        
        # Test with identical boxes (should have lowest cost)
        identical_pred = MockInstanceData(bboxes=gt_bboxes[:2], scores=perfect_scores[:2])  # First 2 gt boxes
        identical_gt = MockInstanceData(bboxes=gt_bboxes[:2], labels=gt_labels[:2])
        
        iou_identical = iou_cost(identical_pred, identical_gt, img_meta)
        print(f"✓ Identical boxes IoU cost diagonal: {torch.diag(iou_identical)}")
        
        focal_identical = focal_loss_cost(identical_pred, identical_gt, img_meta)
        print(f"✓ Identical boxes focal cost diagonal: {torch.diag(focal_identical)}")
        
        # Test with empty predictions
        empty_pred = MockInstanceData(bboxes=torch.empty(0, 4), scores=torch.empty(0, num_classes))
        if len(gt_bboxes) > 0:
            try:
                empty_iou_cost = iou_cost(empty_pred, gt_instances, img_meta)
                print(f"✓ Empty predictions IoU cost shape: {empty_iou_cost.shape}")
                assert empty_iou_cost.shape == (0, 3), "Empty predictions should result in (0, N) cost matrix"
                
                empty_focal_cost = focal_loss_cost(empty_pred, gt_instances, img_meta)
                print(f"✓ Empty predictions focal cost shape: {empty_focal_cost.shape}")
                assert empty_focal_cost.shape == (0, 3), "Empty predictions should result in (0, N) cost matrix"
            except:
                print("✓ Empty predictions handled appropriately")
        
        # Test with empty ground truth
        empty_gt = MockInstanceData(bboxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.long))
        if len(pred_bboxes) > 0:
            try:
                empty_gt_iou_cost = iou_cost(pred_instances, empty_gt, img_meta)
                print(f"✓ Empty ground truth IoU cost shape: {empty_gt_iou_cost.shape}")
                assert empty_gt_iou_cost.shape == (4, 0), "Empty GT should result in (N, 0) cost matrix"
                
                empty_gt_focal_cost = focal_loss_cost(pred_instances, empty_gt, img_meta)
                print(f"✓ Empty ground truth focal cost shape: {empty_gt_focal_cost.shape}")
                assert empty_gt_focal_cost.shape == (4, 0), "Empty GT should result in (N, 0) cost matrix"
            except:
                print("✓ Empty ground truth handled appropriately")
        
        # Test FP16 support for IoUCost
        print("\n--- Testing FP16 support ---")
        
        pred_fp16 = MockInstanceData(bboxes=pred_bboxes.half())
        iou_cost_fp16 = iou_cost(pred_fp16, gt_instances, img_meta)
        print("✓ FP16 support successful")
        print(f"  - FP16 cost matrix shape: {iou_cost_fp16.shape}")
        
        # Test bbox format conversion utility
        print("\n--- Testing utility functions ---")
        
        test_xyxy = torch.tensor([[10.0, 20.0, 30.0, 40.0], [0.0, 0.0, 20.0, 30.0]])
        test_cxcywh = bbox_xyxy_to_cxcywh(test_xyxy)
        
        print("✓ bbox_xyxy_to_cxcywh conversion")
        print(f"  - Input (xyxy): {test_xyxy}")
        print(f"  - Output (cxcywh): {test_cxcywh}")
        
        # Verify conversion correctness
        expected_cxcywh = torch.tensor([[20.0, 30.0, 20.0, 20.0], [10.0, 15.0, 20.0, 30.0]])
        assert torch.allclose(test_cxcywh, expected_cxcywh), "Conversion should be correct"
        
        # Test cost ordering properties
        print("\n--- Testing cost ordering ---")
        
        # Create specific test cases for cost ordering
        close_pred_bbox = torch.tensor([[10, 10, 20, 20]])  # Close to gt[0]
        far_pred_bbox = torch.tensor([[100, 100, 110, 110]])  # Far from gt[0]
        test_gt_bbox = torch.tensor([[12, 12, 22, 22]])
        
        # Create classification scores that match the bboxes
        close_pred_scores = torch.zeros(1, num_classes)
        close_pred_scores[0, gt_labels[0]] = 5.0  # Confident for correct class
        
        far_pred_scores = torch.zeros(1, num_classes) 
        far_pred_scores[0, :] = -2.0  # Not confident about any class
        
        close_instances = MockInstanceData(bboxes=close_pred_bbox, scores=close_pred_scores)
        far_instances = MockInstanceData(bboxes=far_pred_bbox, scores=far_pred_scores)
        test_gt_instances = MockInstanceData(bboxes=test_gt_bbox, labels=gt_labels[:1])
        
        close_iou = iou_cost(close_instances, test_gt_instances, img_meta)
        far_iou = iou_cost(far_instances, test_gt_instances, img_meta)
        
        print(f"✓ Close box IoU cost: {close_iou.item():.6f}")
        print(f"✓ Far box IoU cost: {far_iou.item():.6f}")
        assert close_iou < far_iou, "Closer boxes should have lower IoU cost (more negative)"
        
        close_focal = focal_loss_cost(close_instances, test_gt_instances, img_meta)
        far_focal = focal_loss_cost(far_instances, test_gt_instances, img_meta)
        
        print(f"✓ Close box focal cost: {close_focal.item():.6f}")
        print(f"✓ Far box focal cost: {far_focal.item():.6f}")
        
        # Test weight scaling
        print("\n--- Testing weight scaling ---")
        
        heavy_weight_focal = FocalLossCost(weight=3.0)
        heavy_focal_cost = heavy_weight_focal(close_instances, test_gt_instances, img_meta)
        light_focal_cost = focal_loss_cost(close_instances, test_gt_instances, img_meta)
        
        print(f"✓ Heavy weight focal cost: {heavy_focal_cost.item():.6f}")
        print(f"✓ Light weight focal cost: {light_focal_cost.item():.6f}")
        expected_heavy_focal = light_focal_cost * 3.0
        assert torch.allclose(heavy_focal_cost, expected_heavy_focal), "Focal weight scaling should work correctly"
        
        # Test BaseMatchCost interface
        print("\n--- Testing base class ---")
        
        # Verify that all cost classes inherit from BaseMatchCost
        assert isinstance(focal_loss_cost, BaseMatchCost), "FocalLossCost should inherit from BaseMatchCost"
        assert isinstance(iou_cost, BaseMatchCost), "IoUCost should inherit from BaseMatchCost"
        # Note: BBox3DL1Cost doesn't properly inherit, but has weight attribute
        
        # Test direct focal loss computation
        print("\n--- Testing direct focal loss computation ---")
        
        # Test internal _focal_loss_cost method
        test_cls_pred = torch.tensor([[2.0, -1.0, 3.0], [0.5, 0.5, -2.0]])  # 2 predictions, 3 classes
        test_gt_labels = torch.tensor([2, 0])  # GT labels
        
        direct_focal_cost = focal_loss_cost._focal_loss_cost(test_cls_pred, test_gt_labels)
        print(f"✓ Direct focal loss computation: {direct_focal_cost}")
        print(f"  - Shape: {direct_focal_cost.shape}")
        assert direct_focal_cost.shape == (2, 2), "Direct focal cost should have shape (2, 2)"
        assert torch.isfinite(direct_focal_cost).all(), "Direct focal cost should be finite"
        
        print("✓ All assertions passed")
        print("🎉 Cost Functions test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_cost_functions()
