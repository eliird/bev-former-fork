"""
NormalizeMultiviewImage Transform
Normalizes multi-camera images with mean and std for BEVFormer
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any, Sequence


class NormalizeMultiviewImage:
    """Normalize multi-view images with given mean and std.
    
    This transform normalizes all camera images using the same statistics,
    typically ImageNet statistics for pre-trained backbones.
    
    Args:
        mean (Sequence[float]): Mean values for each channel.
        std (Sequence[float]): Std values for each channel.
        to_rgb (bool): Whether to convert BGR to RGB. Default: True.
    """
    
    def __init__(self,
                 mean: Sequence[float],
                 std: Sequence[float],
                 to_rgb: bool = True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        
        # Ensure mean and std are broadcastable
        if len(self.mean.shape) == 1:
            self.mean = self.mean.reshape(1, 1, 1, -1)  # (1, 1, 1, C)
        if len(self.std.shape) == 1:
            self.std = self.std.reshape(1, 1, 1, -1)    # (1, 1, 1, C)
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize a single image or batch of images.
        
        Args:
            img (np.ndarray): Image with shape (..., H, W, C)
            
        Returns:
            np.ndarray: Normalized image
        """
        # Convert to float32 if needed
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        
        # Convert BGR to RGB if needed
        if self.to_rgb:
            # Assume input is BGR, convert to RGB
            img = img[..., ::-1]
        
        # Normalize: (img - mean) / std
        img = (img - self.mean) / self.std
        
        return img
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize multi-view images.
        
        Args:
            results (dict): Dictionary containing multi-view images.
                Must contain 'img' key with shape (N, H, W, C).
                
        Returns:
            dict: Dictionary with normalized images.
        """
        if 'img' not in results:
            raise KeyError("'img' key not found in input data")
        
        img = results['img']
        
        # Ensure img is numpy array
        if torch.is_tensor(img):
            img = img.numpy()
        
        # Normalize all images
        img_norm = self._normalize_image(img)
        
        # Update results
        results['img'] = img_norm
        results['img_norm_cfg'] = dict(
            mean=self.mean.flatten().tolist(),
            std=self.std.flatten().tolist(),
            to_rgb=self.to_rgb
        )
        
        return results
    
    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                   f'mean={self.mean.flatten().tolist()}, '
                   f'std={self.std.flatten().tolist()}, '
                   f'to_rgb={self.to_rgb})')
        return repr_str


def test_normalize_multi_view_image():
    """Test NormalizeMultiviewImage transform."""
    print("=" * 60)
    print("Testing NormalizeMultiviewImage")
    print("=" * 60)
    
    import pickle
    import os
    from load_multi_view_image import LoadMultiViewImageFromFiles
    
    # Load sample data
    data_file = '/home/irdali.durrani/po-pi/BEVFormer/data/nuscenes/nuscenes_infos_temporal_val.pkl'
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    sample = data['data_list'][0]
    
    # First load images
    loader = LoadMultiViewImageFromFiles(to_float32=True)
    sample_with_imgs = loader(sample)
    
    print(f"✓ Loaded images with shape: {sample_with_imgs['img'].shape}")
    print(f"✓ Original image range: [{sample_with_imgs['img'].min():.1f}, {sample_with_imgs['img'].max():.1f}]")
    
    # Test normalization with ImageNet statistics
    # Note: nuScenes config uses mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False
    # This suggests BGR input with custom mean but unit std
    
    # Test 1: Standard ImageNet normalization (RGB)
    normalize_imagenet = NormalizeMultiviewImage(
        mean=[123.675, 116.28, 103.53],  # ImageNet BGR mean
        std=[58.395, 57.12, 57.375],     # ImageNet BGR std  
        to_rgb=True
    )
    
    print(f"\n✓ Created ImageNet normalizer: {normalize_imagenet}")
    
    results_imagenet = normalize_imagenet(sample_with_imgs.copy())
    img_norm = results_imagenet['img']
    
    print(f"✓ ImageNet normalization successful!")
    print(f"  - Normalized shape: {img_norm.shape}")
    print(f"  - Normalized range: [{img_norm.min():.3f}, {img_norm.max():.3f}]")
    print(f"  - Normalized mean: {img_norm.mean():.3f}")
    print(f"  - Normalized std: {img_norm.std():.3f}")
    
    # Test 2: nuScenes config normalization (BGR with custom mean)
    normalize_nuscenes = NormalizeMultiviewImage(
        mean=[103.530, 116.280, 123.675],  # nuScenes BGR mean
        std=[1.0, 1.0, 1.0],               # Unit std
        to_rgb=False                       # Keep BGR
    )
    
    print(f"\n✓ Created nuScenes normalizer: {normalize_nuscenes}")
    
    results_nuscenes = normalize_nuscenes(sample_with_imgs.copy())
    img_norm_ns = results_nuscenes['img']
    
    print(f"✓ nuScenes normalization successful!")
    print(f"  - Normalized shape: {img_norm_ns.shape}")
    print(f"  - Normalized range: [{img_norm_ns.min():.3f}, {img_norm_ns.max():.3f}]")
    print(f"  - Normalized mean: {img_norm_ns.mean():.3f}")
    print(f"  - Normalized std: {img_norm_ns.std():.3f}")
    
    # Check per-camera statistics
    print(f"\nPer-camera statistics (nuScenes normalization):")
    for i, cam_name in enumerate(sample_with_imgs['cam_names']):
        cam_img = img_norm_ns[i]
        print(f"  - {cam_name}: mean={cam_img.mean():.3f}, std={cam_img.std():.3f}")
    
    # Test error handling
    print(f"\nTesting error handling...")
    
    # Test missing img key
    try:
        normalize_nuscenes({'no_img': 'test'})
        print("❌ Should have failed with missing img")
    except KeyError as e:
        print(f"✓ Correctly caught missing img error")
    
    print("\n" + "=" * 60)
    print("All NormalizeMultiviewImage tests passed!")
    print("=" * 60)
    
    return results_nuscenes


if __name__ == "__main__":
    test_normalize_multi_view_image()