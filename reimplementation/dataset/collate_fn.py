"""
Custom collation function for BEVFormer temporal sequences
Handles batching of multi-temporal, multi-view data with variable metadata
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for BEVFormer temporal data batching.
    
    Handles:
    - Temporal sequences: (B, T, N, C, H, W)
    - Variable-length metadata lists
    - 3D bounding boxes with different number of objects per sample
    - Proper tensor stacking and device placement
    
    Args:
        batch: List of samples from NuScenesDataset.__getitem__()
               Each sample contains:
               - img: torch.Tensor(T, N, C, H, W)
               - img_metas: List[Dict] (length T)
               - gt_bboxes_3d: torch.Tensor(M, 9) [optional, training only]
               - gt_labels_3d: torch.Tensor(M,) [optional, training only]
    
    Returns:
        dict: Batched data
            - img: torch.Tensor(B, T, N, C, H, W)
            - img_metas: List[List[Dict]] - [batch_idx][temporal_idx] -> metadata
            - gt_bboxes_3d: List[torch.Tensor] - per-sample bbox tensors
            - gt_labels_3d: List[torch.Tensor] - per-sample label tensors
    """
    if not batch:
        raise ValueError("Empty batch provided to collate_fn")
    
    batch_size = len(batch)
    
    # Stack images: from List[(T, N, C, H, W)] to (B, T, N, C, H, W)
    imgs = []
    for sample in batch:
        if 'img' not in sample:
            raise ValueError("Sample missing 'img' key")
        imgs.append(sample['img'])
    
    try:
        # Stack all images into batch dimension
        batched_imgs = torch.stack(imgs, dim=0)  # (B, T, N, C, H, W)
    except RuntimeError as e:
        # Handle size mismatch by logging details
        shapes = [img.shape for img in imgs]
        raise RuntimeError(f"Cannot stack images with shapes: {shapes}. Original error: {e}")
    
    # Collect metadata: List[List[Dict]] - [batch][temporal] -> metadata
    batched_img_metas = []
    for sample in batch:
        if 'img_metas' not in sample:
            raise ValueError("Sample missing 'img_metas' key")
        batched_img_metas.append(sample['img_metas'])
    
    # Prepare result dictionary
    result = {
        'img': batched_imgs,
        'img_metas': batched_img_metas
    }
    
    # Handle 3D bounding boxes (training mode)
    if 'gt_bboxes_3d' in batch[0]:
        batched_gt_bboxes_3d = []
        for sample in batch:
            batched_gt_bboxes_3d.append(sample['gt_bboxes_3d'])
        result['gt_bboxes_3d'] = batched_gt_bboxes_3d
    
    # Handle 3D labels (training mode) 
    if 'gt_labels_3d' in batch[0]:
        batched_gt_labels_3d = []
        for sample in batch:
            batched_gt_labels_3d.append(sample['gt_labels_3d'])
        result['gt_labels_3d'] = batched_gt_labels_3d
    
    return result


def validate_batch(batch: Dict[str, Any], expected_batch_size: int, expected_temporal_length: int) -> bool:
    """
    Validate that a batch has the correct structure and dimensions.
    
    Args:
        batch: Collated batch dictionary
        expected_batch_size: Expected batch size
        expected_temporal_length: Expected temporal sequence length
    
    Returns:
        bool: True if batch is valid
    """
    try:
        # Check image tensor shape
        if 'img' not in batch:
            print("❌ Batch missing 'img' key")
            return False
        
        img = batch['img']
        expected_img_shape = (expected_batch_size, expected_temporal_length, 6, 3)  # (B, T, N, C, *, *)
        
        if len(img.shape) != 6:
            print(f"❌ Expected 6D image tensor, got {len(img.shape)}D: {img.shape}")
            return False
        
        if img.shape[:4] != expected_img_shape:
            print(f"❌ Expected image shape prefix {expected_img_shape}, got {img.shape[:4]}")
            return False
        
        # Check metadata structure
        if 'img_metas' not in batch:
            print("❌ Batch missing 'img_metas' key")
            return False
        
        img_metas = batch['img_metas']
        if not isinstance(img_metas, list) or len(img_metas) != expected_batch_size:
            print(f"❌ Expected img_metas list of length {expected_batch_size}, got {len(img_metas) if isinstance(img_metas, list) else type(img_metas)}")
            return False
        
        for batch_idx, sample_metas in enumerate(img_metas):
            if not isinstance(sample_metas, list) or len(sample_metas) != expected_temporal_length:
                print(f"❌ Expected temporal metadata list of length {expected_temporal_length} at batch {batch_idx}, got {len(sample_metas) if isinstance(sample_metas, list) else type(sample_metas)}")
                return False
            
            for temporal_idx, meta in enumerate(sample_metas):
                if not isinstance(meta, dict):
                    print(f"❌ Expected dict metadata at batch {batch_idx}, temporal {temporal_idx}, got {type(meta)}")
                    return False
        
        # Check GT data if present
        if 'gt_bboxes_3d' in batch:
            gt_bboxes = batch['gt_bboxes_3d']
            if not isinstance(gt_bboxes, list) or len(gt_bboxes) != expected_batch_size:
                print(f"❌ Expected gt_bboxes_3d list of length {expected_batch_size}, got {len(gt_bboxes) if isinstance(gt_bboxes, list) else type(gt_bboxes)}")
                return False
            
            for bbox in gt_bboxes:
                if not torch.is_tensor(bbox):
                    print(f"❌ Expected tensor in gt_bboxes_3d, got {type(bbox)}")
                    return False
        
        if 'gt_labels_3d' in batch:
            gt_labels = batch['gt_labels_3d']
            if not isinstance(gt_labels, list) or len(gt_labels) != expected_batch_size:
                print(f"❌ Expected gt_labels_3d list of length {expected_batch_size}, got {len(gt_labels) if isinstance(gt_labels, list) else type(gt_labels)}")
                return False
            
            for labels in gt_labels:
                if not torch.is_tensor(labels):
                    print(f"❌ Expected tensor in gt_labels_3d, got {type(labels)}")
                    return False
        
        print(f"✅ Batch validation passed:")
        print(f"   - img: {img.shape}")
        print(f"   - img_metas: {len(img_metas)} samples x {len(img_metas[0])} temporal frames")
        if 'gt_bboxes_3d' in batch:
            bbox_shapes = [bbox.shape for bbox in batch['gt_bboxes_3d']]
            print(f"   - gt_bboxes_3d: {bbox_shapes}")
        if 'gt_labels_3d' in batch:
            label_shapes = [labels.shape for labels in batch['gt_labels_3d']]
            print(f"   - gt_labels_3d: {label_shapes}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch validation failed with exception: {e}")
        return False


def test_collate_fn():
    """Test the custom collate function with dummy data."""
    print("=" * 80)
    print("TESTING CUSTOM COLLATE FUNCTION")
    print("=" * 80)
    
    # Create dummy temporal samples
    T, N, C, H, W = 4, 6, 3, 900, 1600
    batch_size = 2
    
    dummy_batch = []
    for b in range(batch_size):
        sample = {
            'img': torch.randn(T, N, C, H, W),
            'img_metas': [
                {
                    'scene_token': f'scene_{b}',
                    'can_bus': np.random.randn(18),
                    'prev_bev_exists': t > 0,
                    'frame_idx_in_sequence': t
                }
                for t in range(T)
            ],
            'gt_bboxes_3d': torch.randn(5 + b, 9),  # Variable number of objects
            'gt_labels_3d': torch.randint(0, 10, (5 + b,))
        }
        dummy_batch.append(sample)
    
    print(f"🔄 Testing collate function with {batch_size} samples...")
    
    # Test collation
    try:
        batched = custom_collate_fn(dummy_batch)
        print("✅ Collation successful!")
        
        # Validate batch
        is_valid = validate_batch(batched, batch_size, T)
        
        if is_valid:
            print("🎉 Collate function test passed!")
            return batched
        else:
            print("❌ Batch validation failed!")
            return None
            
    except Exception as e:
        print(f"❌ Collation failed: {e}")
        return None


if __name__ == "__main__":
    test_collate_fn()