import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingResult:
    """Result of pseudo sampling containing positive and negative samples.
    
    Attributes:
        pos_inds (Tensor): Indices of positive samples
        neg_inds (Tensor): Indices of negative samples
        pos_bboxes (Tensor): Predicted bboxes of positive samples
        neg_bboxes (Tensor): Predicted bboxes of negative samples
        pos_assigned_gt_inds (Tensor): GT indices that positive samples are assigned to
        pos_gt_bboxes (Tensor): GT bboxes that positive samples are assigned to
        num_gts (int): Number of ground truth boxes
    """
    pos_inds: torch.Tensor
    neg_inds: torch.Tensor
    pos_bboxes: torch.Tensor
    neg_bboxes: torch.Tensor
    pos_assigned_gt_inds: torch.Tensor
    pos_gt_bboxes: torch.Tensor
    num_gts: int


class PseudoSampler:
    """A pseudo sampler that does not do actual sampling.
    
    This sampler is used in DETR-based methods where Hungarian matching
    already determines the assignment. It simply returns all positive
    and negative samples without any sampling strategy.
    """
    
    def __init__(self):
        pass
    
    def sample(self, assign_result, pred_bboxes, gt_bboxes, **kwargs):
        """Package assignment results into sampling results.
        
        Args:
            assign_result (AssignResult): Assignment results from Hungarian matcher
                containing gt_inds, max_overlaps, and labels
            pred_bboxes (Tensor): Predicted bounding boxes [num_query, box_dim]
            gt_bboxes (Tensor): Ground truth bounding boxes [num_gts, box_dim]
            
        Returns:
            SamplingResult: Sampling results containing positive and negative indices
        """
        # Get positive and negative indices
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False
        ).squeeze(-1).unique()
        
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False
        ).squeeze(-1).unique()
        
        # Get positive and negative bboxes
        pos_bboxes = pred_bboxes[pos_inds] if pos_inds.numel() > 0 else pred_bboxes.new_empty((0, pred_bboxes.size(-1)))
        neg_bboxes = pred_bboxes[neg_inds] if neg_inds.numel() > 0 else pred_bboxes.new_empty((0, pred_bboxes.size(-1)))
        
        # Get assigned GT indices for positive samples (0-indexed)
        if pos_inds.numel() > 0:
            pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1  # Convert to 0-indexed
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds]
        else:
            pos_assigned_gt_inds = assign_result.gt_inds.new_empty((0,))
            pos_gt_bboxes = gt_bboxes.new_empty((0, gt_bboxes.size(-1)))
        
        return SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            pos_bboxes=pos_bboxes,
            neg_bboxes=neg_bboxes,
            pos_assigned_gt_inds=pos_assigned_gt_inds,
            pos_gt_bboxes=pos_gt_bboxes,
            num_gts=assign_result.num_gts
        )


def test_pseudo_sampler():
    """Test PseudoSampler functionality"""
    print("Testing PseudoSampler...")

    # Import AssignResult for testing
    from ..util import AssignResult
    
    # Create mock assignment result
    num_queries = 10
    num_gts = 3
    
    # Mock assignment: queries 0,2,5 assigned to GTs 1,2,3 respectively
    gt_inds = torch.zeros(num_queries, dtype=torch.long)
    gt_inds[0] = 1  # Query 0 -> GT 0
    gt_inds[2] = 2  # Query 2 -> GT 1  
    gt_inds[5] = 3  # Query 5 -> GT 2
    
    assign_result = AssignResult(
        num_gts=num_gts,
        gt_inds=gt_inds,
        max_overlaps=torch.rand(num_queries),
        labels=torch.randint(0, 10, (num_queries,))
    )
    
    # Create mock predictions and ground truths
    pred_bboxes = torch.randn(num_queries, 10)  # 10-dim bbox
    gt_bboxes = torch.randn(num_gts, 10)
    
    # Test sampler
    sampler = PseudoSampler()
    sampling_result = sampler.sample(assign_result, pred_bboxes, gt_bboxes)
    
    # Verify results
    assert sampling_result.pos_inds.tolist() == [0, 2, 5], "Positive indices incorrect"
    assert sampling_result.neg_inds.tolist() == [1, 3, 4, 6, 7, 8, 9], "Negative indices incorrect"
    assert sampling_result.pos_assigned_gt_inds.tolist() == [0, 1, 2], "GT assignments incorrect"
    assert sampling_result.pos_bboxes.shape == (3, 10), "Positive bboxes shape incorrect"
    assert sampling_result.neg_bboxes.shape == (7, 10), "Negative bboxes shape incorrect"
    assert sampling_result.pos_gt_bboxes.shape == (3, 10), "GT bboxes shape incorrect"
    
    print("✓ PseudoSampler test passed!")
    
    # Test edge case: no positive samples
    gt_inds_empty = torch.zeros(num_queries, dtype=torch.long)
    assign_result_empty = AssignResult(
        num_gts=num_gts,
        gt_inds=gt_inds_empty,
        max_overlaps=torch.rand(num_queries),
        labels=torch.randint(0, 10, (num_queries,))
    )
    
    sampling_result_empty = sampler.sample(assign_result_empty, pred_bboxes, gt_bboxes)
    assert sampling_result_empty.pos_inds.numel() == 0, "Should have no positive samples"
    assert sampling_result_empty.neg_inds.numel() == num_queries, "All should be negative"
    
    print("✓ Edge case test passed!")


if __name__ == "__main__":
    test_pseudo_sampler()