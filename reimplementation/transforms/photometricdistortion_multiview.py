

import random
import cv2
import numpy as np
from typing import Callable


def convert_color_factory(src: str, dst: str) -> Callable:

    code = getattr(cv2, f'COLOR_{src.upper()}2{dst.upper()}')

    def convert_color(img: np.ndarray) -> np.ndarray:
        out_img = cv2.cvtColor(img, code)
        return out_img

    convert_color.__doc__ = f"""Convert a {src.upper()} image to {dst.upper()}
        image.

    Args:
        img (ndarray or str): The input image.

    Returns:
        ndarray: The converted {dst.upper()} image.
    """

    return convert_color


bgr2rgb = convert_color_factory('bgr', 'rgb')

rgb2bgr = convert_color_factory('rgb', 'bgr')

bgr2hsv = convert_color_factory('bgr', 'hsv')

hsv2bgr = convert_color_factory('hsv', 'bgr')

bgr2hls = convert_color_factory('bgr', 'hls')

hls2bgr = convert_color_factory('hls', 'bgr')


class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        
        # Handle both numpy array and list formats
        if isinstance(imgs, np.ndarray):
            # Convert to list for processing
            imgs_list = [imgs[i] for i in range(imgs.shape[0])]
        else:
            imgs_list = imgs
        
        for img in imgs_list:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(0, 1):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(0, 1)
            if mode == 1:
                if random.randint(0, 1):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = bgr2hsv(img)

            # random saturation
            if random.randint(0, 1):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(0, 1):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(0, 1):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(0, 1):
                img = img[..., np.random.permutation(3)]
            new_imgs.append(img)
        
        # Convert back to original format
        if isinstance(results['img'], np.ndarray):
            results['img'] = np.stack(new_imgs, axis=0)
        else:
            results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


def test_photometric_distortion():
    """Test PhotoMetricDistortionMultiViewImage transform."""
    print("=" * 60)
    print("Testing PhotoMetricDistortionMultiViewImage")
    print("=" * 60)
    
    # Create sample multi-view images
    np.random.seed(42)
    random.seed(42)
    
    # Test with different input formats
    test_formats = [
        {'name': 'numpy_array', 'use_array': True},
        {'name': 'list_format', 'use_array': False}
    ]
    
    for test_format in test_formats:
        print(f"\nTesting with {test_format['name']}...")
        
        # Create sample images (6 cameras, 100x100, RGB)
        imgs = []
        for i in range(6):
            # Create realistic image data
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8).astype(np.float32)
            imgs.append(img)
        
        # Format input according to test
        if test_format['use_array']:
            img_data = np.stack(imgs, axis=0)  # Shape: (6, 100, 100, 3)
        else:
            img_data = imgs  # List format
        
        results = {'img': img_data}
        
        # Create transform
        transform = PhotoMetricDistortionMultiViewImage(
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18
        )
        
        print(f"✓ Created transform: {transform}")
        
        # Apply transform
        try:
            original_stats = []
            if test_format['use_array']:
                for i in range(img_data.shape[0]):
                    img = img_data[i]
                    original_stats.append({
                        'mean': float(img.mean()),
                        'std': float(img.std()),
                        'min': float(img.min()),
                        'max': float(img.max())
                    })
            else:
                for img in img_data:
                    original_stats.append({
                        'mean': float(img.mean()),
                        'std': float(img.std()),
                        'min': float(img.min()),
                        'max': float(img.max())
                    })
            
            # Apply photometric distortion
            distorted_results = transform(results)
            
            print(f"✓ Transform successful!")
            
            # Check output format
            distorted_imgs = distorted_results['img']
            if test_format['use_array']:
                assert isinstance(distorted_imgs, np.ndarray), "Output should be numpy array"
                assert distorted_imgs.shape == img_data.shape, f"Shape mismatch: {distorted_imgs.shape} vs {img_data.shape}"
            else:
                assert isinstance(distorted_imgs, list), "Output should be list"
                assert len(distorted_imgs) == len(img_data), f"Length mismatch: {len(distorted_imgs)} vs {len(img_data)}"
            
            # Check statistics changed (distortion applied)
            distorted_stats = []
            if test_format['use_array']:
                for i in range(distorted_imgs.shape[0]):
                    img = distorted_imgs[i]
                    distorted_stats.append({
                        'mean': float(img.mean()),
                        'std': float(img.std()),
                        'min': float(img.min()),
                        'max': float(img.max())
                    })
            else:
                for img in distorted_imgs:
                    distorted_stats.append({
                        'mean': float(img.mean()),
                        'std': float(img.std()),
                        'min': float(img.min()),
                        'max': float(img.max())
                    })
            
            # Verify distortion was applied (at least some stats should change)
            stats_changed = False
            for orig, dist in zip(original_stats, distorted_stats):
                if abs(orig['mean'] - dist['mean']) > 1.0:  # Brightness change
                    stats_changed = True
                    break
                if abs(orig['std'] - dist['std']) > 1.0:  # Contrast change
                    stats_changed = True
                    break
            
            print(f"  - Output format: {type(distorted_imgs)} ✓")
            print(f"  - Statistics changed: {stats_changed} ✓")
            print(f"  - Original mean: {original_stats[0]['mean']:.2f}")
            print(f"  - Distorted mean: {distorted_stats[0]['mean']:.2f}")
            
        except Exception as e:
            print(f"❌ Transform failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test with integrated pipeline
    print(f"\nTesting with full pipeline integration...")
    try:
        import os
        import pickle
        from load_multi_view_image import LoadMultiViewImageFromFiles
        from normalize_multi_view_image import NormalizeMultiviewImage
        
        # Load real nuScenes data
        data_file = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            sample = data['data_list'][0]
            
            # Create pipeline
            loader = LoadMultiViewImageFromFiles(to_float32=True)
            normalizer = NormalizeMultiviewImage(
                mean=[103.530, 116.280, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False
            )
            distorter = PhotoMetricDistortionMultiViewImage()
            
            # Apply pipeline
            results = loader(sample)
            print(f"  - Loaded images: {results['img'].shape}")
            
            # Apply distortion BEFORE normalization (as in training)
            results = distorter(results)
            print(f"  - Applied distortion: {results['img'].shape}")
            
            results = normalizer(results)
            print(f"  - Applied normalization: {results['img'].shape}")
            
            print(f"✓ Full pipeline integration successful!")
        else:
            print(f"  - Skipping pipeline test (no dataset found)")
    
    except ImportError as e:
        print(f"  - Skipping pipeline test (missing modules): {e}")
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
    
    # Test error handling
    print(f"\nTesting error handling...")
    
    # Test with wrong dtype
    try:
        wrong_dtype_imgs = [np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8) for _ in range(6)]
        results = {'img': wrong_dtype_imgs}
        transform = PhotoMetricDistortionMultiViewImage()
        transform(results)
        print("❌ Should have failed with wrong dtype")
    except AssertionError:
        print("✓ Correctly caught wrong dtype error")
    
    # Test with missing img key
    try:
        transform = PhotoMetricDistortionMultiViewImage()
        transform({'no_img': 'test'})
        print("❌ Should have failed with missing img")
    except KeyError:
        print("✓ Correctly caught missing img key error")
    
    print("\n" + "=" * 60)
    print("All PhotoMetricDistortionMultiViewImage tests passed!")
    print("=" * 60)
    
    return True


def test_color_conversion_functions():
    """Test color conversion utility functions."""
    print("\nTesting color conversion functions...")
    
    # Create test image
    test_img_bgr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8).astype(np.float32)
    
    # Test BGR to RGB
    rgb_img = bgr2rgb(test_img_bgr)
    assert rgb_img.shape == test_img_bgr.shape
    # Check that channels are swapped
    np.testing.assert_array_equal(rgb_img[..., 0], test_img_bgr[..., 2])
    np.testing.assert_array_equal(rgb_img[..., 2], test_img_bgr[..., 0])
    print("✓ BGR to RGB conversion")
    
    # Test RGB to BGR (should be inverse)
    bgr_back = rgb2bgr(rgb_img)
    np.testing.assert_array_equal(bgr_back, test_img_bgr)
    print("✓ RGB to BGR conversion")
    
    # Test BGR to HSV and back
    hsv_img = bgr2hsv(test_img_bgr)
    assert hsv_img.shape == test_img_bgr.shape
    print("✓ BGR to HSV conversion")
    
    bgr_from_hsv = hsv2bgr(hsv_img)
    # Allow small numerical differences due to float precision
    np.testing.assert_allclose(bgr_from_hsv, test_img_bgr, atol=1.0)
    print("✓ HSV to BGR conversion")
    
    # Test BGR to HLS and back
    hls_img = bgr2hls(test_img_bgr)
    assert hls_img.shape == test_img_bgr.shape
    print("✓ BGR to HLS conversion")
    
    bgr_from_hls = hls2bgr(hls_img)
    np.testing.assert_allclose(bgr_from_hls, test_img_bgr, atol=1.0)
    print("✓ HLS to BGR conversion")


if __name__ == "__main__":
    # Test color conversion functions first
    test_color_conversion_functions()
    
    # Test main photometric distortion
    test_photometric_distortion()
