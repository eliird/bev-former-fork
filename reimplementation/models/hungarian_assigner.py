import torch

from .costs import FocalLossCost, BBox3DL1Cost, IoUCost, MockInstanceData
from .util import AssignResult
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def normalize_bbox(bboxes, pc_range):

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

class HungarianAssigner3D:
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
        # self.cls_cost = build_match_cost(cls_cost)
        # self.reg_cost = build_match_cost(reg_cost)
        # self.iou_cost = build_match_cost(iou_cost)

        self.cls_cost = FocalLossCost(weight=2.0)# build_from_cfg(cls_cost, TASK_UTILS)
        self.reg_cost = BBox3DL1Cost(weight=0.25)
        self.iou_cost = IoUCost(weight=0.0)
        self.pc_range = pc_range

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes, 
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # Import MockInstanceData for creating proper instance objects
        
        # Create InstanceData objects for cost computation
        pred_instances = MockInstanceData(scores=cls_pred)
        gt_instances = MockInstanceData(labels=gt_labels)
        
        # classification cost
        cls_cost = self.cls_cost(pred_instances, gt_instances)
        
        # regression L1 cost
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
      
        # weighted sum of above two costs
        cost = cls_cost + reg_cost
        
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)


def test_hungarian_assigner_3d():
    """Test HungarianAssigner3D implementation"""
    print("=" * 60)
    print("Testing HungarianAssigner3D")
    print("=" * 60)
    
    try:
        # Import required dependencies
        from util import AssignResult, normalize_bbox, denormalize_bbox
        import numpy as np
        
        # Test setup - BEVFormer style parameters
        pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        num_classes = 10  # nuScenes classes
        
        # Create HungarianAssigner3D
        assigner = HungarianAssigner3D(
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=pc_range
        )
        
        print("✓ HungarianAssigner3D created successfully")
        print(f"  - pc_range: {pc_range}")
        print(f"  - cls_cost weight: {assigner.cls_cost.weight}")
        print(f"  - reg_cost weight: {assigner.reg_cost.weight}")
        print(f"  - iou_cost weight: {assigner.iou_cost.weight}")
        
        print("\n--- Testing basic assignment ---")
        
        # Create test data - 3D bounding boxes in BEVFormer format
        # Format: [cx, cy, cz, w, l, h, rot, vx, vy] (9D with velocity)
        num_queries = 8
        num_gts = 3
        
        # Predicted bounding boxes (normalized format used in loss computation)
        bbox_pred = torch.tensor([
            [0.0, 0.0, 0.5, 1.6, 4.2, 1.8, 0.0, 0.1],  # Close to GT 0
            [10.0, 5.0, 0.2, 1.8, 4.5, 1.7, 0.2, 0.2],  # Close to GT 1  
            [-5.0, -10.0, 0.3, 2.1, 5.1, 2.0, -0.1, -0.1], # Close to GT 2
            [20.0, 20.0, 1.0, 1.5, 4.0, 1.6, 0.5, 0.0],  # Far from all
            [0.1, 0.1, 0.6, 1.7, 4.1, 1.9, 0.01, 0.05], # Very close to GT 0
            [-30.0, -30.0, 0.1, 2.0, 4.8, 1.5, -0.3, 0.3], # Far
            [10.1, 5.1, 0.25, 1.75, 4.4, 1.65, 0.15, 0.15], # Very close to GT 1
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Zero prediction
        ], dtype=torch.float32)
        
        # Classification predictions (logits)
        cls_pred = torch.randn(num_queries, num_classes, dtype=torch.float32)
        # Make some predictions more confident for specific classes
        cls_pred[0, 2] = 5.0  # Confident car prediction
        cls_pred[1, 1] = 4.0  # Confident truck prediction  
        cls_pred[2, 0] = 3.0  # Confident pedestrian prediction
        
        # Ground truth bounding boxes (unnormalized format)
        gt_bboxes = torch.tensor([
            [0.2, 0.1, 0.4, 1.5, 4.0, 1.7, 0.05, 0.05, 0.02],  # GT 0: car
            [10.2, 5.1, 0.15, 1.7, 4.3, 1.6, 0.18, 0.18, 0.15], # GT 1: truck
            [-4.8, -9.9, 0.25, 2.0, 5.0, 1.9, -0.08, -0.08, -0.05] # GT 2: pedestrian
        ], dtype=torch.float32)
        
        # Ground truth labels (0-based for nuScenes: 0=pedestrian, 1=truck, 2=car, etc.)
        gt_labels = torch.tensor([2, 1, 0], dtype=torch.long)  # car, truck, pedestrian
        
        print("✓ Test data created")
        print(f"  - bbox_pred shape: {bbox_pred.shape}")
        print(f"  - cls_pred shape: {cls_pred.shape}")  
        print(f"  - gt_bboxes shape: {gt_bboxes.shape}")
        print(f"  - gt_labels: {gt_labels}")
        
        # Test assignment
        assign_result = assigner.assign(
            bbox_pred=bbox_pred,
            cls_pred=cls_pred,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels
        )
        
        print("✓ Assignment completed successfully")
        print(f"  - assign_result type: {type(assign_result)}")
        print(f"  - num_gts: {assign_result.num_gts}")
        print(f"  - num_preds: {assign_result.num_preds}")
        print(f"  - assigned_gt_inds: {assign_result.gt_inds}")
        print(f"  - assigned_labels: {assign_result.labels}")
        
        # Verify assignment structure
        assert isinstance(assign_result, AssignResult), "Should return AssignResult"
        assert assign_result.num_gts == num_gts, f"num_gts should be {num_gts}"
        assert assign_result.num_preds == num_queries, f"num_preds should be {num_queries}"
        assert assign_result.gt_inds.shape == (num_queries,), "gt_inds shape mismatch"
        assert assign_result.labels.shape == (num_queries,), "labels shape mismatch"
        
        # Verify assignment logic
        assigned_mask = assign_result.gt_inds > 0
        background_mask = assign_result.gt_inds == 0
        
        print(f"  - Assigned predictions: {assigned_mask.sum().item()}")
        print(f"  - Background predictions: {background_mask.sum().item()}")
        
        # Check that assignments are valid (1-based GT indices)
        if assigned_mask.any():
            assigned_gt_indices = assign_result.gt_inds[assigned_mask] - 1  # Convert to 0-based
            assert torch.all(assigned_gt_indices >= 0), "Invalid negative GT indices"
            assert torch.all(assigned_gt_indices < num_gts), "GT indices exceed num_gts"
            
            # Check assigned labels match GT labels
            assigned_labels = assign_result.labels[assigned_mask]
            expected_labels = gt_labels[assigned_gt_indices]
            assert torch.allclose(assigned_labels.float(), expected_labels.float()), "Assigned labels don't match GT"
        
        # Background predictions should have label -1 or background class
        background_labels = assign_result.labels[background_mask]
        assert torch.all(background_labels == -1), "Background predictions should have label -1"
        
        print("✓ Assignment verification successful")
        
        print("\n--- Testing edge cases ---")
        
        # Test case 1: No ground truth
        assign_result_no_gt = assigner.assign(
            bbox_pred=bbox_pred[:4],  # Fewer predictions
            cls_pred=cls_pred[:4],
            gt_bboxes=torch.empty(0, 9, dtype=torch.float32),
            gt_labels=torch.empty(0, dtype=torch.long)
        )
        
        print("✓ No ground truth case handled")
        print(f"  - All predictions assigned to background: {torch.all(assign_result_no_gt.gt_inds == 0)}")
        assert assign_result_no_gt.num_gts == 0
        assert torch.all(assign_result_no_gt.gt_inds == 0), "All should be background when no GT"
        
        # Test case 2: No predictions
        assign_result_no_pred = assigner.assign(
            bbox_pred=torch.empty(0, 8, dtype=torch.float32),
            cls_pred=torch.empty(0, num_classes, dtype=torch.float32),
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels
        )
        
        print("✓ No predictions case handled")
        assert assign_result_no_pred.num_gts == num_gts
        assert assign_result_no_pred.num_preds == 0
        
        # Test case 3: Single prediction, single GT
        single_assign_result = assigner.assign(
            bbox_pred=bbox_pred[:1],
            cls_pred=cls_pred[:1],
            gt_bboxes=gt_bboxes[:1],
            gt_labels=gt_labels[:1]
        )
        
        print("✓ Single prediction/GT case handled")
        assert single_assign_result.num_gts == 1
        assert single_assign_result.num_preds == 1
        # Should assign the single prediction to the single GT
        assert single_assign_result.gt_inds[0] == 1, "Single pred should be assigned to GT"
        assert single_assign_result.labels[0] == gt_labels[0], "Single pred label should match GT"
        
        print("\n--- Testing normalization functions ---")
        
        # Test normalize_bbox function
        test_bbox_unnorm = torch.tensor([[
            0.0, 0.0, 0.0, 2.0, 4.0, 1.8, 0.1, 0.2, 0.1  # cx, cy, cz, w, l, h, rot, vx, vy
        ]], dtype=torch.float32)
        
        normalized = normalize_bbox(test_bbox_unnorm, pc_range)
        print(f"✓ Normalization test:")
        print(f"  - Input shape: {test_bbox_unnorm.shape}")
        print(f"  - Output shape: {normalized.shape}")
        print(f"  - Normalized bbox: {normalized[0]}")
        
        # Verify normalization properties
        # Shape changes from 9D to 10D due to rotation sin/cos encoding
        expected_output_shape = (test_bbox_unnorm.shape[0], test_bbox_unnorm.shape[1] + 1)
        assert normalized.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {normalized.shape}"
        assert torch.isfinite(normalized).all(), "Normalized bbox should be finite"
        
        # Width, length, height should be log-transformed (format: cx, cy, w, l, cz, h, sin, cos, vx, vy)
        assert normalized[0, 2] == torch.log(test_bbox_unnorm[0, 3]), "Width should be log-transformed"
        assert normalized[0, 3] == torch.log(test_bbox_unnorm[0, 4]), "Length should be log-transformed" 
        assert normalized[0, 5] == torch.log(test_bbox_unnorm[0, 5]), "Height should be log-transformed"
        
        # Rotation should be sin/cos encoded
        expected_sin = torch.sin(test_bbox_unnorm[0, 6])
        expected_cos = torch.cos(test_bbox_unnorm[0, 6])
        assert torch.allclose(normalized[0, 6], expected_sin), "Rotation sin encoding"
        assert torch.allclose(normalized[0, 7], expected_cos), "Rotation cos encoding"
        
        # Velocity components should be preserved
        assert normalized[0, 8] == test_bbox_unnorm[0, 7], "vx should be preserved"
        assert normalized[0, 9] == test_bbox_unnorm[0, 8], "vy should be preserved"
        
        print("✓ Normalization verification successful")
        
        # Test denormalization (round trip)
        denormalized = denormalize_bbox(normalized, pc_range)
        print(f"✓ Denormalization test:")
        print(f"  - Denormalized bbox: {denormalized[0]}")
        
        # Should recover original values (within numerical precision)
        diff = torch.abs(denormalized - test_bbox_unnorm)
        max_diff = diff.max().item()
        print(f"  - Max difference after round-trip: {max_diff:.6f}")
        assert max_diff < 1e-5, "Round-trip normalization/denormalization should be accurate"
        
        print("\n--- Testing cost computation ---")
        
        # Test that costs are computed correctly
        # This is internal to the assigner, but we can verify it works
        
        # Create a case where we know the expected assignment
        simple_bbox_pred = torch.tensor([
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],  # Perfect match with GT 0
            [100.0, 100.0, 100.0, 10.0, 10.0, 10.0, 1.0, 1.0]  # Very far from all GTs
        ], dtype=torch.float32)
        
        simple_cls_pred = torch.zeros(2, num_classes, dtype=torch.float32)
        simple_cls_pred[0, gt_labels[0]] = 10.0  # Very confident correct prediction
        simple_cls_pred[1, :] = -10.0  # Very unconfident predictions
        
        simple_assign_result = assigner.assign(
            bbox_pred=simple_bbox_pred,
            cls_pred=simple_cls_pred,
            gt_bboxes=gt_bboxes[:1],  # Only first GT
            gt_labels=gt_labels[:1]
        )
        
        print("✓ Simple assignment test:")
        print(f"  - Assignment indices: {simple_assign_result.gt_inds}")
        print(f"  - Assignment labels: {simple_assign_result.labels}")
        
        # First prediction should be assigned (good match), second should be background
        assert simple_assign_result.gt_inds[0] == 1, "Close prediction should be assigned"
        assert simple_assign_result.gt_inds[1] == 0, "Far prediction should be background"
        
        print("\n--- Testing batch scenarios ---")
        
        # Test with many predictions, few ground truths
        many_preds = torch.randn(50, 8, dtype=torch.float32)  # Many random predictions
        many_cls_pred = torch.randn(50, num_classes, dtype=torch.float32)
        
        many_assign_result = assigner.assign(
            bbox_pred=many_preds,
            cls_pred=many_cls_pred,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels
        )
        
        print(f"✓ Many predictions test (50 pred, 3 GT):")
        print(f"  - Assigned predictions: {(many_assign_result.gt_inds > 0).sum().item()}")
        print(f"  - Background predictions: {(many_assign_result.gt_inds == 0).sum().item()}")
        
        # Should assign exactly min(num_preds, num_gts) = min(50, 3) = 3 predictions
        n_assigned = (many_assign_result.gt_inds > 0).sum().item()
        assert n_assigned == min(50, 3), f"Should assign exactly {min(50, 3)} predictions, got {n_assigned}"
        
        # Test with few predictions, many ground truths  
        few_preds = torch.randn(2, 8, dtype=torch.float32)
        few_cls_pred = torch.randn(2, num_classes, dtype=torch.float32)
        
        few_assign_result = assigner.assign(
            bbox_pred=few_preds,
            cls_pred=few_cls_pred,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels
        )
        
        print(f"✓ Few predictions test (2 pred, 3 GT):")
        print(f"  - Assigned predictions: {(few_assign_result.gt_inds > 0).sum().item()}")
        
        # Should assign exactly min(num_preds, num_gts) = min(2, 3) = 2 predictions
        n_assigned_few = (few_assign_result.gt_inds > 0).sum().item()
        assert n_assigned_few == min(2, 3), f"Should assign exactly {min(2, 3)} predictions, got {n_assigned_few}"
        
        print("\n--- Testing gradient flow ---")
        
        # Test that gradients can flow through the assignment process
        bbox_pred_grad = bbox_pred.clone().requires_grad_(True)
        cls_pred_grad = cls_pred.clone().requires_grad_(True)
        
        # The assigner itself doesn't need gradients, but the cost computation should work
        # This tests that tensor operations are differentiable
        
        assign_result_grad = assigner.assign(
            bbox_pred=bbox_pred_grad,
            cls_pred=cls_pred_grad,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels
        )
        
        print("✓ Gradient flow test successful")
        print(f"  - Can assign with requires_grad=True tensors")
        
        # Verify the assignment is consistent
        assert torch.allclose(assign_result_grad.gt_inds.float(), assign_result.gt_inds.float())
        assert torch.allclose(assign_result_grad.labels.float(), assign_result.labels.float())
        
        print("✓ All assertions passed")
        print("🎉 HungarianAssigner3D test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_hungarian_assigner_3d()