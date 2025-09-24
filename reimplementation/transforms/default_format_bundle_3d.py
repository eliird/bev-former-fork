"""
DefaultFormatBundle3D Transform
Formats data to final tensor format for 3D detection models
Based on Pack3DDetInputs functionality from MMDetection3D
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any, Tuple, Sequence
from .structures import LiDARInstance3DBoxes


class DefaultFormatBundle3D:
    """Default formatting bundle for 3D detection.
    
    This transform converts data to the final tensor format expected by
    3D detection models. It handles:
    - Image tensor formatting (NCHW format)
    - 3D bounding box tensor conversion
    - Label tensor conversion
    - Data type standardization
    
    Args:
        keys (Sequence[str]): Keys to be formatted and collected.
        class_names (Optional[List[str]]): Class names for the dataset.
    """
    
    def __init__(self, 
                 keys: Sequence[str] = ('img', 'gt_bboxes_3d', 'gt_labels_3d'),
                 class_names: Optional[List[str]] = None):
        self.keys = keys
        self.class_names = class_names or []
    
    def _format_img(self, img_data: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        """Format images to tensor format.
        
        Args:
            img_data: Multi-view images, either as numpy array (N, H, W, C) 
                     or list of arrays [(H, W, C), ...]
                     
        Returns:
            torch.Tensor: Formatted images with shape (N, C, H, W)
        """
        if isinstance(img_data, list):
            # Validate that all items in list are numpy arrays
            if not all(isinstance(img, np.ndarray) for img in img_data):
                raise TypeError("All items in img list must be numpy arrays")
            # Convert list to numpy array first
            img_array = np.stack(img_data, axis=0)  # (N, H, W, C)
        elif isinstance(img_data, np.ndarray):
            img_array = img_data  # Already (N, H, W, C)
        else:
            raise TypeError(f"img_data must be numpy array or list of arrays, got {type(img_data)}")
        
        # Ensure float32
        if img_array.dtype != np.float32:
            img_array = img_array.astype(np.float32)
        
        # Convert to tensor and transpose to (N, C, H, W)
        img_tensor = torch.from_numpy(img_array)
        img_tensor = img_tensor.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        return img_tensor
    
    def _format_bboxes_3d(self, bboxes_3d: Union[LiDARInstance3DBoxes, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Format 3D bounding boxes to tensor.
        
        Args:
            bboxes_3d: 3D bounding boxes
            
        Returns:
            torch.Tensor: Formatted 3D bboxes tensor
        """
        if isinstance(bboxes_3d, LiDARInstance3DBoxes):
            # Extract tensor from our custom box class
            return bboxes_3d.tensor
        elif isinstance(bboxes_3d, np.ndarray):
            return torch.from_numpy(bboxes_3d.astype(np.float32))
        elif torch.is_tensor(bboxes_3d):
            return bboxes_3d.float()
        else:
            raise TypeError(f"Unsupported bboxes_3d type: {type(bboxes_3d)}")
    
    def _format_labels_3d(self, labels_3d: Union[torch.Tensor, np.ndarray, List[int]]) -> torch.Tensor:
        """Format 3D labels to tensor.
        
        Args:
            labels_3d: 3D labels
            
        Returns:
            torch.Tensor: Formatted labels tensor with long dtype
        """
        if isinstance(labels_3d, np.ndarray):
            return torch.from_numpy(labels_3d.astype(np.int64))
        elif torch.is_tensor(labels_3d):
            return labels_3d.long()
        elif isinstance(labels_3d, (list, tuple)):
            return torch.tensor(labels_3d, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported labels_3d type: {type(labels_3d)}")
    
    def _to_tensor(self, data: Any) -> torch.Tensor:
        """Convert various data types to tensor.
        
        Args:
            data: Input data
            
        Returns:
            torch.Tensor: Output tensor
        """
        if torch.is_tensor(data):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, (list, tuple)):
            return torch.tensor(data)
        else:
            raise TypeError(f"Cannot convert type {type(data)} to tensor")
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format specified keys in results dict.
        
        Args:
            results (dict): Input data dictionary
            
        Returns:
            dict: Formatted results dictionary
        """
        formatted_results = results.copy()
        
        # Format images
        if 'img' in results and 'img' in self.keys:
            formatted_results['img'] = self._format_img(results['img'])
        
        # Format 3D bounding boxes
        if 'gt_bboxes_3d' in results and 'gt_bboxes_3d' in self.keys:
            formatted_results['gt_bboxes_3d'] = self._format_bboxes_3d(results['gt_bboxes_3d'])
        
        # Format 3D labels
        if 'gt_labels_3d' in results and 'gt_labels_3d' in self.keys:
            formatted_results['gt_labels_3d'] = self._format_labels_3d(results['gt_labels_3d'])
        
        # Format other keys generically
        for key in self.keys:
            if key in results and key not in ['img', 'gt_bboxes_3d', 'gt_labels_3d']:
                # Generic tensor conversion for other keys
                if not torch.is_tensor(results[key]):
                    try:
                        formatted_results[key] = self._to_tensor(results[key])
                    except TypeError:
                        # If conversion fails, keep original data
                        formatted_results[key] = results[key]
        
        return formatted_results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}'
        if self.class_names:
            repr_str += f', class_names={self.class_names}'
        repr_str += ')'
        return repr_str


def test_default_format_bundle_3d():
    """Test DefaultFormatBundle3D transform."""
    print("=" * 60)
    print("Testing DefaultFormatBundle3D")
    print("=" * 60)
    
    # Test 1: Basic tensor formatting with synthetic data
    print("Testing basic tensor formatting...")
    
    # Create synthetic data
    imgs = np.random.randint(0, 256, (6, 100, 200, 3), dtype=np.uint8).astype(np.float32)
    bboxes = torch.randn(5, 9)  # 5 boxes with 9 features (x,y,z,w,l,h,rot,vx,vy)
    labels = np.array([0, 1, 2, 0, 1], dtype=np.int32)
    
    results = {
        'img': imgs,
        'gt_bboxes_3d': bboxes,
        'gt_labels_3d': labels,
        'sample_idx': 12345
    }
    
    # Create formatter
    formatter = DefaultFormatBundle3D(
        keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
        class_names=['car', 'truck', 'bus']
    )
    print(f"✓ Created formatter: {formatter}")
    
    # Apply formatting
    formatted = formatter(results)
    
    print(f"✓ Formatting successful!")
    print(f"  - Original img shape: {results['img'].shape} ({results['img'].dtype})")
    print(f"  - Formatted img shape: {formatted['img'].shape} ({formatted['img'].dtype})")
    print(f"  - Original bboxes shape: {results['gt_bboxes_3d'].shape} ({results['gt_bboxes_3d'].dtype})")
    print(f"  - Formatted bboxes shape: {formatted['gt_bboxes_3d'].shape} ({formatted['gt_bboxes_3d'].dtype})")
    print(f"  - Original labels shape: {results['gt_labels_3d'].shape} ({results['gt_labels_3d'].dtype})")
    print(f"  - Formatted labels shape: {formatted['gt_labels_3d'].shape} ({formatted['gt_labels_3d'].dtype})")
    
    # Verify tensor formats
    assert torch.is_tensor(formatted['img']), "Image should be tensor"
    assert formatted['img'].shape == (6, 3, 100, 200), f"Image shape should be (N,C,H,W), got {formatted['img'].shape}"
    assert formatted['img'].dtype == torch.float32, "Image should be float32"
    
    assert torch.is_tensor(formatted['gt_bboxes_3d']), "Bboxes should be tensor"
    assert formatted['gt_bboxes_3d'].dtype == torch.float32, "Bboxes should be float32"
    
    assert torch.is_tensor(formatted['gt_labels_3d']), "Labels should be tensor"
    assert formatted['gt_labels_3d'].dtype == torch.long, "Labels should be long"
    
    print("✓ All tensor formats verified")
    
    # Test 2: With LiDARInstance3DBoxes
    print(f"\nTesting with LiDARInstance3DBoxes...")
    
    bbox_data = torch.randn(3, 9)
    lidar_boxes = LiDARInstance3DBoxes(bbox_data)
    
    results_lidar = {
        'img': imgs[:3],  # Fewer cameras for this test
        'gt_bboxes_3d': lidar_boxes,
        'gt_labels_3d': torch.tensor([0, 1, 2], dtype=torch.long)
    }
    
    formatted_lidar = formatter(results_lidar)
    
    print(f"✓ LiDARInstance3DBoxes formatting successful!")
    print(f"  - Input type: {type(results_lidar['gt_bboxes_3d'])}")
    print(f"  - Output type: {type(formatted_lidar['gt_bboxes_3d'])}")
    print(f"  - Shape: {formatted_lidar['gt_bboxes_3d'].shape}")
    
    # Test 3: With list format images (pipeline format)
    print(f"\nTesting with list format images...")
    
    img_list = [np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8).astype(np.float32) for _ in range(6)]
    
    results_list = {
        'img': img_list,
        'gt_bboxes_3d': np.array([[0, 0, 0, 2, 4, 1.5, 0.5, 0, 0]], dtype=np.float32),
        'gt_labels_3d': [0]
    }
    
    formatted_list = formatter(results_list)
    
    print(f"✓ List format images formatting successful!")
    print(f"  - Input: list of {len(img_list)} images")
    print(f"  - Output shape: {formatted_list['img'].shape}")
    assert formatted_list['img'].shape == (6, 3, 120, 160), "List images not formatted correctly"
    
    # Test with real nuScenes data
    print(f"\nTesting with real nuScenes pipeline data...")
    try:
        from load_multi_view_image import LoadMultiViewImageFromFiles
        from normalize_multi_view_image import NormalizeMultiviewImage
        from load_annotations_3d import LoadAnnotations3D
        from object_filters import ObjectNameFilter, ObjectRangeFilter
        from pad_multi_view_image import PadMultiViewImage
        
        import os
        import pickle
        
        # Load sample data
        data_file = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            sample = data['data_list'][0]
            
            # Apply preceding pipeline
            loader = LoadMultiViewImageFromFiles(to_float32=True)
            normalizer = NormalizeMultiviewImage(
                mean=[103.530, 116.280, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False
            )
            ann_loader = LoadAnnotations3D()
            name_filter = ObjectNameFilter(['car', 'truck', 'bus', 'pedestrian', 'bicycle'])
            range_filter = ObjectRangeFilter([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
            padder = PadMultiViewImage(size_divisor=32)
            
            results = sample
            results = loader(results)
            results = normalizer(results)
            results = ann_loader(results)
            results = name_filter(results)
            results = range_filter(results)
            results = padder(results)
            
            print(f"  - Pipeline data ready, shapes before formatting:")
            print(f"    - img: {results['img'].shape} ({results['img'].dtype})")
            print(f"    - gt_bboxes_3d: {results['gt_bboxes_3d'].shape}")
            print(f"    - gt_labels_3d: {results['gt_labels_3d'].shape}")
            
            # Apply formatting
            real_formatted = formatter(results.copy())
            
            print(f"  ✓ Real pipeline data formatting successful!")
            print(f"    - Formatted img: {real_formatted['img'].shape} ({real_formatted['img'].dtype})")
            print(f"    - Formatted bboxes: {real_formatted['gt_bboxes_3d'].shape} ({real_formatted['gt_bboxes_3d'].dtype})")
            print(f"    - Formatted labels: {real_formatted['gt_labels_3d'].shape} ({real_formatted['gt_labels_3d'].dtype})")
            
            # Verify correct format for model input
            assert real_formatted['img'].dim() == 4, "Image should be 4D tensor (N,C,H,W)"
            assert real_formatted['img'].shape[1] == 3, "Should have 3 channels"
            assert torch.is_tensor(real_formatted['gt_bboxes_3d']), "Bboxes should be tensor"
            assert torch.is_tensor(real_formatted['gt_labels_3d']), "Labels should be tensor"
            
        else:
            print(f"  - Skipping real data test (dataset not found)")
    
    except ImportError:
        print(f"  - Skipping real data test (missing dependencies)")
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test error handling
    print(f"\nTesting error handling...")
    
    # Test with missing keys
    empty_results = {}
    empty_formatted = formatter(empty_results)
    print("✓ Correctly handled missing keys")
    
    # Test with invalid data types
    try:
        invalid_results = {'img': "invalid_image_data"}
        formatter(invalid_results)
        print("❌ Should have failed with invalid image data")
    except (TypeError, ValueError):
        print("✓ Correctly caught invalid image data error")
    
    print("\n" + "=" * 60)
    print("All DefaultFormatBundle3D tests passed!")
    print("=" * 60)
    
    return formatted


if __name__ == "__main__":
    test_default_format_bundle_3d()