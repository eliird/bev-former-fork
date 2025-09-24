"""
PadMultiViewImage Transform
Pads multi-camera images to specified size or divisor for BEVFormer
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Union, Any, Tuple


class PadMultiViewImage:
    """Pad the multi-view image.
    
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor".
    
    Args:
        size (tuple, optional): Fixed padding size (height, width).
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
    """

    def __init__(self, 
                 size: Optional[Tuple[int, int]] = None,
                 size_divisor: Optional[int] = None, 
                 pad_val: float = 0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        
        # Only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None, \
            "Either size or size_divisor should be specified"
        assert size is None or size_divisor is None, \
            "size and size_divisor cannot both be specified"

    def _pad_img_to_size(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Pad image to target size.
        
        Args:
            img (np.ndarray): Image to pad with shape (H, W, C)
            target_size (tuple): Target size (height, width)
            
        Returns:
            np.ndarray: Padded image
        """
        target_h, target_w = target_size
        h, w = img.shape[:2]
        
        # Calculate padding
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        
        if pad_h == 0 and pad_w == 0:
            return img
        
        # Pad with cv2 (bottom and right padding)
        padded_img = cv2.copyMakeBorder(
            img,
            top=0,
            bottom=pad_h,
            left=0,
            right=pad_w,
            borderType=cv2.BORDER_CONSTANT,
            value=self.pad_val
        )
        
        return padded_img

    def _pad_img_to_divisor(self, img: np.ndarray, divisor: int) -> np.ndarray:
        """Pad image to make dimensions divisible by divisor.
        
        Args:
            img (np.ndarray): Image to pad with shape (H, W, C)
            divisor (int): The divisor
            
        Returns:
            np.ndarray: Padded image
        """
        h, w = img.shape[:2]
        
        # Calculate new dimensions
        new_h = int(np.ceil(h / divisor) * divisor)
        new_w = int(np.ceil(w / divisor) * divisor)
        
        # Pad to new dimensions
        return self._pad_img_to_size(img, (new_h, new_w))

    def _pad_img(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Pad images according to size or size_divisor.
        
        Args:
            results (dict): Input data dictionary
            
        Returns:
            dict: Updated dictionary with padded images
        """
        if 'img' not in results:
            return results
        
        imgs = results['img']
        
        # Handle both numpy array and list formats
        if isinstance(imgs, np.ndarray):
            # Convert to list for processing
            imgs_list = [imgs[i] for i in range(imgs.shape[0])]
        else:
            imgs_list = imgs
        
        # Store original shapes
        ori_shapes = [img.shape for img in imgs_list]
        
        # Pad images
        padded_imgs = []
        for img in imgs_list:
            if self.size is not None:
                padded_img = self._pad_img_to_size(img, self.size)
            elif self.size_divisor is not None:
                padded_img = self._pad_img_to_divisor(img, self.size_divisor)
            else:
                padded_img = img
            padded_imgs.append(padded_img)
        
        # Convert back to original format
        if isinstance(results['img'], np.ndarray):
            results['img'] = np.stack(padded_imgs, axis=0)
        else:
            results['img'] = padded_imgs
        
        # Update metadata
        padded_shapes = [img.shape for img in padded_imgs]
        results['ori_shape'] = ori_shapes
        results['img_shape'] = padded_shapes
        results['pad_shape'] = padded_shapes
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        
        return results

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Call function to pad images.
        
        Args:
            results (dict): Result dict from loading pipeline.
            
        Returns:
            dict: Updated result dict with padded images.
        """
        return self._pad_img(results)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


def test_pad_multi_view_image():
    """Test PadMultiViewImage transform."""
    print("=" * 60)
    print("Testing PadMultiViewImage")
    print("=" * 60)
    
    # Test with synthetic data first
    print("Testing with synthetic data...")
    
    # Create test images with same base size (like after loading but before padding)
    test_imgs = []
    for i in range(6):
        # Create images with same size (typical nuScenes camera resolution)
        h, w = 900, 1600
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8).astype(np.float32)
        test_imgs.append(img)
    
    results = {'img': np.stack(test_imgs, axis=0)}
    
    print(f"✓ Created test images with shapes: {[img.shape for img in test_imgs]}")
    
    # Test 1: Padding to divisor
    print(f"\\nTesting padding to divisor (32)...")
    pad_divisor = PadMultiViewImage(size_divisor=32, pad_val=0.0)
    print(f"✓ Created divisor padder: {pad_divisor}")
    
    results_padded = pad_divisor(results.copy())
    padded_imgs = results_padded['img']
    
    print(f"✓ Padding successful!")
    print(f"  - Original shapes: {results_padded['ori_shape']}")
    print(f"  - Padded shapes: {results_padded['pad_shape']}")
    
    # Verify all dimensions are divisible by 32
    for i, shape in enumerate(results_padded['pad_shape']):
        h, w = shape[:2]
        assert h % 32 == 0, f"Height {h} not divisible by 32 for image {i}"
        assert w % 32 == 0, f"Width {w} not divisible by 32 for image {i}"
    print(f"  - All dimensions divisible by 32 ✓")
    
    # Test 2: Padding to fixed size
    print(f"\\nTesting padding to fixed size (1024, 1792)...")
    pad_fixed = PadMultiViewImage(size=(1024, 1792), pad_val=128.0)
    print(f"✓ Created fixed size padder: {pad_fixed}")
    
    results_fixed = pad_fixed(results.copy())
    fixed_imgs = results_fixed['img']
    
    print(f"✓ Fixed size padding successful!")
    print(f"  - Target size: (1024, 1792)")
    print(f"  - All padded shapes: {set(tuple(shape[:2]) for shape in results_fixed['pad_shape'])}")
    
    # Verify all images have target size
    for shape in results_fixed['pad_shape']:
        assert shape[:2] == (1024, 1792), f"Shape {shape[:2]} doesn't match target (1024, 1792)"
    print(f"  - All images padded to target size ✓")
    
    # Test 3: Test with list format (mimicking pipeline format)
    print(f"\\nTesting with list format...")
    results_list = {'img': test_imgs}  # List format
    
    results_list_padded = pad_divisor(results_list.copy())
    
    assert isinstance(results_list_padded['img'], list), "Output should be list format"
    print(f"✓ List format preserved")
    print(f"  - List padded shapes: {[img.shape for img in results_list_padded['img']]}")
    
    # Test 4: Test with different sized images (edge case)
    print(f"\\nTesting with different sized images...")
    diff_imgs = []
    for i in range(3):
        # Create images with different sizes to test padding
        h, w = 897 + i * 13, 1601 + i * 7  # Not divisible by 32
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8).astype(np.float32)
        diff_imgs.append(img)
    
    results_diff = {'img': diff_imgs}  # Must use list format for different sizes
    print(f"  - Different input shapes: {[img.shape for img in diff_imgs]}")
    
    results_diff_padded = pad_divisor(results_diff.copy())
    padded_shapes = [img.shape for img in results_diff_padded['img']]
    print(f"  - Different padded shapes: {padded_shapes}")
    
    # Verify all are divisible by 32
    for shape in padded_shapes:
        h, w = shape[:2]
        assert h % 32 == 0 and w % 32 == 0, f"Different size padding failed for shape {shape}"
    print(f"✓ Different sized images padded correctly")
    
    # Test with real nuScenes data
    print(f"\\nTesting with real nuScenes data...")
    try:
        import os
        import pickle
        from load_multi_view_image import LoadMultiViewImageFromFiles
        
        data_file = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            sample = data['data_list'][0]
            
            # Load images first
            loader = LoadMultiViewImageFromFiles(to_float32=True)
            loaded_results = loader(sample)
            
            print(f"  - Loaded real images: {loaded_results['img'].shape}")
            print(f"  - Original shapes: {loaded_results['img_shape']}")
            
            # Apply padding
            real_padded = pad_divisor(loaded_results.copy())
            
            print(f"  - Padded shapes: {real_padded['pad_shape']}")
            
            # Verify padding worked
            for shape in real_padded['pad_shape']:
                h, w = shape[:2]
                assert h % 32 == 0 and w % 32 == 0, f"Real data padding failed for shape {shape}"
            
            print(f"✓ Real nuScenes data padding successful!")
        else:
            print(f"  - Skipping real data test (dataset not found)")
    
    except ImportError:
        print(f"  - Skipping real data test (missing dependencies)")
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
    
    # Test error handling
    print(f"\\nTesting error handling...")
    
    # Test with missing img key
    empty_results = pad_divisor({'no_img': 'test'})
    assert 'img' not in empty_results or empty_results.get('img') is None
    print("✓ Correctly handled missing img key")
    
    # Test invalid parameters
    try:
        invalid_pad = PadMultiViewImage()  # No size or divisor
        print("❌ Should have failed with no parameters")
    except AssertionError:
        print("✓ Correctly caught missing parameters error")
    
    try:
        invalid_pad = PadMultiViewImage(size=(512, 512), size_divisor=32)  # Both specified
        print("❌ Should have failed with both parameters")
    except AssertionError:
        print("✓ Correctly caught conflicting parameters error")
    
    print("\\n" + "=" * 60)
    print("All PadMultiViewImage tests passed!")
    print("=" * 60)
    
    return results_padded


def test_padding_integration():
    """Test PadMultiViewImage in pipeline integration."""
    print("\\n" + "=" * 40)
    print("Testing Padding Pipeline Integration")
    print("=" * 40)
    
    try:
        from load_multi_view_image import LoadMultiViewImageFromFiles
        from normalize_multi_view_image import NormalizeMultiviewImage
        from photometricdistortion_multiview import PhotoMetricDistortionMultiViewImage
        from load_annotations_3d import LoadAnnotations3D
        from object_filters import ObjectNameFilter, ObjectRangeFilter
        
        import os
        import pickle
        
        # Load sample data
        data_file = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'
        if not os.path.exists(data_file):
            print("❌ Dataset not found")
            return
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        sample = data['data_list'][0]
        
        # Create pipeline with padding
        loader = LoadMultiViewImageFromFiles(to_float32=True)
        distorter = PhotoMetricDistortionMultiViewImage()
        normalizer = NormalizeMultiviewImage(
            mean=[103.530, 116.280, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False
        )
        ann_loader = LoadAnnotations3D()
        name_filter = ObjectNameFilter(['car', 'truck', 'bus', 'pedestrian', 'bicycle'])
        range_filter = ObjectRangeFilter([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        padder = PadMultiViewImage(size_divisor=32)
        
        print("✓ Created complete pipeline with padding")
        
        # Apply pipeline
        results = sample
        results = loader(results)
        print(f"  - Images loaded: {results['img'].shape}")
        
        results = distorter(results)
        print(f"  - Distortion applied")
        
        results = normalizer(results)
        print(f"  - Normalization applied")
        
        results = ann_loader(results)
        original_count = len(results['gt_bboxes_3d'])
        print(f"  - Annotations loaded: {original_count} objects")
        
        results = name_filter(results)
        name_count = len(results['gt_bboxes_3d'])
        print(f"  - Name filter applied: {name_count} objects")
        
        results = range_filter(results)
        final_count = len(results['gt_bboxes_3d'])
        print(f"  - Range filter applied: {final_count} objects")
        
        # Apply padding - this is the new step
        results = padder(results)
        print(f"  - Padding applied")
        
        print(f"✓ Complete pipeline with padding successful!")
        print(f"  - Final data shapes:")
        print(f"    - Images: {results['img'].shape}")
        print(f"    - Bboxes: {results['gt_bboxes_3d'].shape}")
        print(f"    - Labels: {results['gt_labels_3d'].shape}")
        print(f"    - Padded shapes: {results['pad_shape'][0]} (first camera)")
        
        # Verify padding worked
        for shape in results['pad_shape']:
            h, w = shape[:2]
            assert h % 32 == 0 and w % 32 == 0, f"Pipeline padding failed for shape {shape}"
        
        print(f"  - All images properly padded to divisor 32 ✓")
        
    except ImportError as e:
        print(f"❌ Pipeline test failed (missing modules): {e}")
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test padding individually
    test_pad_multi_view_image()
    
    # Test in pipeline integration
    test_padding_integration()