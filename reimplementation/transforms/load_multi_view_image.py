"""
LoadMultiViewImageFromFiles Transform
Loads multi-camera images from file paths for BEVFormer
"""

import os
import numpy as np
from PIL import Image
import torch
from typing import Dict, List, Optional, Union, Any


class LoadMultiViewImageFromFiles:
    """Load multi-view images from file paths.
    
    This transform loads images from 6 cameras simultaneously and prepares them
    for BEVFormer processing.
    
    Args:
        to_float32 (bool): Whether to convert images to float32. Default: False.
        color_type (str): Color type of images. Default: 'color'.
        channel_order (str): Channel order. Default: 'rgb'.
        use_bgr_to_rgb (bool): Whether to convert BGR to RGB. Default: True.
    """
    
    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 channel_order: str = 'rgb',
                 use_bgr_to_rgb: bool = True):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.use_bgr_to_rgb = use_bgr_to_rgb
        
        # Standard nuScenes camera order
        self.cam_names = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT', 
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'
        ]
    
    def _load_single_image(self, filepath: str) -> np.ndarray:
        """Load a single image from filepath.
        
        Args:
            filepath (str): Path to image file
            
        Returns:
            np.ndarray: Loaded image with shape (H, W, 3)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found: {filepath}")
        
        try:
            # Load image using PIL
            img = Image.open(filepath)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Handle channel order conversion
            if self.use_bgr_to_rgb and self.channel_order == 'bgr':
                img_array = img_array[..., ::-1]  # BGR to RGB
            
            # Convert to float32 if requested
            if self.to_float32:
                img_array = img_array.astype(np.float32)
            
            return img_array
            
        except Exception as e:
            raise RuntimeError(f"Failed to load image from {filepath}: {str(e)}")
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Load multi-view images.
        
        Args:
            results (dict): Dictionary containing camera information.
                Must contain 'cams' key with camera data.
                
        Returns:
            dict: Updated dictionary with loaded images.
        """
        if 'cams' not in results:
            raise KeyError("'cams' key not found in input data")
        
        cams_data = results['cams']
        
        # Check that all required cameras are present
        missing_cams = []
        for cam_name in self.cam_names:
            if cam_name not in cams_data:
                missing_cams.append(cam_name)
        
        if missing_cams:
            raise KeyError(f"Missing camera data for: {missing_cams}")
        
        # Load images from all cameras
        imgs = []
        img_shapes = []
        cam_names = []
        
        for cam_name in self.cam_names:
            cam_info = cams_data[cam_name]
            
            if 'data_path' not in cam_info:
                raise KeyError(f"'data_path' not found for camera {cam_name}")
            
            filepath = cam_info['data_path']

            # Check if file exists, if not try alternative paths
            if not os.path.exists(filepath):
                # Try different relative paths depending on where we're running from
                possible_paths = [
                    os.path.join('..', filepath),           # One level up (from reimplementation/)
                    os.path.join('..', '..', filepath),     # Two levels up (from train/ or similar)
                    os.path.join('..', '..', '..', filepath), # Three levels up (just in case)
                    filepath.replace('./data/', '../../data/'),  # Direct replacement for common case
                ]

                for alt_path in possible_paths:
                    if os.path.exists(alt_path):
                        filepath = alt_path
                        break
                else:
                    # If none of the alternative paths work, keep the original for error reporting
                    pass

            # Load image
            img = self._load_single_image(filepath)
            imgs.append(img)
            img_shapes.append(img.shape)
            cam_names.append(cam_name)
        
        # Stack images into multi-view format
        # Shape: (num_cams, H, W, 3)
        img_array = np.stack(imgs, axis=0)
        
        # Add to results
        results['img'] = img_array
        results['img_shape'] = img_shapes  # List of shapes for each camera
        results['img_fields'] = ['img']
        results['cam_names'] = cam_names
        
        # Add filename info
        results['filename'] = [cams_data[cam]['data_path'] for cam in self.cam_names]
        
        # Store original shapes for potential resizing
        results['ori_shape'] = img_shapes
        
        return results
    
    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                   f'to_float32={self.to_float32}, '
                   f'color_type=\'{self.color_type}\', '
                   f'channel_order=\'{self.channel_order}\', '
                   f'use_bgr_to_rgb={self.use_bgr_to_rgb})')
        return repr_str


def test_load_multi_view_image():
    """Test LoadMultiViewImageFromFiles transform with real nuScenes data."""
    print("=" * 60)
    print("Testing LoadMultiViewImageFromFiles")
    print("=" * 60)
    
    import pickle
    
    # Load sample data
    data_file = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    print("Loading nuScenes dataset...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get first sample
    sample = data['data_list'][0]
    print(f"✓ Loaded dataset with {len(data['data_list'])} samples")
    
    # Create transform
    transform = LoadMultiViewImageFromFiles(to_float32=True)
    print(f"✓ Created transform: {transform}")
    
    # Test transform
    print("\nTesting image loading...")
    
    try:
        # Apply transform
        results = transform(sample)
        
        print(f"✓ Transform successful!")
        print(f"  - Loaded {len(results['cam_names'])} camera images")
        print(f"  - Image array shape: {results['img'].shape}")
        print(f"  - Image dtype: {results['img'].dtype}")
        print(f"  - Camera names: {results['cam_names']}")
        
        # Check individual image shapes
        print("\nIndividual camera image shapes:")
        for i, (cam_name, shape) in enumerate(zip(results['cam_names'], results['img_shape'])):
            print(f"  - {cam_name}: {shape}")
            
        # Check if images were loaded correctly
        img_stats = []
        for i, cam_name in enumerate(results['cam_names']):
            img = results['img'][i]
            stats = {
                'mean': np.mean(img),
                'std': np.std(img),
                'min': np.min(img),
                'max': np.max(img)
            }
            img_stats.append(stats)
            print(f"  - {cam_name}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.1f}, {stats['max']:.1f}]")
        
        # Test error handling
        print("\nTesting error handling...")
        
        # Test missing cams
        try:
            invalid_sample = {'cams': {'CAM_FRONT': sample['cams']['CAM_FRONT']}}  # Only one camera
            transform(invalid_sample)
            print("❌ Should have failed with missing cameras")
        except KeyError as e:
            print(f"✓ Correctly caught missing cameras error: {e}")
        
        # Test missing data_path
        try:
            invalid_cam = dict(sample['cams']['CAM_FRONT'])
            del invalid_cam['data_path']
            invalid_sample = dict(sample)
            invalid_sample['cams'] = dict(sample['cams'])
            invalid_sample['cams']['CAM_FRONT'] = invalid_cam
            transform(invalid_sample)
            print("❌ Should have failed with missing data_path")
        except KeyError as e:
            print(f"✓ Correctly caught missing data_path error: {e}")
        
        print("\n" + "=" * 60)
        print("All LoadMultiViewImageFromFiles tests passed!")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"❌ Transform failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_different_options():
    """Test transform with different configuration options."""
    print("\n" + "=" * 40)
    print("Testing different configuration options")
    print("=" * 40)
    
    import pickle
    
    # Load sample data
    data_file = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    sample = data['data_list'][0]
    
    # Test different configurations
    configs = [
        {'to_float32': False, 'name': 'uint8'},
        {'to_float32': True, 'name': 'float32'},
    ]
    
    for config in configs:
        name = config.pop('name')
        transform = LoadMultiViewImageFromFiles(**config)
        
        try:
            results = transform(sample)
            img = results['img']
            print(f"✓ {name}: shape={img.shape}, dtype={img.dtype}, range=[{img.min():.1f}, {img.max():.1f}]")
        except Exception as e:
            print(f"❌ {name} failed: {e}")


if __name__ == "__main__":
    # Run comprehensive tests
    results = test_load_multi_view_image()
    
    if results is not None:
        # Test additional configurations
        test_with_different_options()
        
        print(f"\nTransform is ready for use!")
        print(f"Sample output keys: {list(results.keys())}")
    else:
        print(f"\nTransform test failed!")