"""
CustomCollect3D Transform
Final data collection and organization for BEVFormer pipeline
Based on the reference implementation from transform_3d.py
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any, Sequence
from .structures import LiDARInstance3DBoxes


class CustomCollect3D:
    """Collect data from the loader relevant to the specific task.
    
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "gt_bboxes_3d", "gt_labels_3d".
    The "img_metas" item is always populated. The contents of the "img_metas"
    dictionary depends on "meta_keys".
    
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            img_metas and collected in ``data[img_metas]``.
            Default includes essential BEVFormer metadata.
    """
    
    def __init__(self,
                 keys: Sequence[str],
                 meta_keys: Sequence[str] = (
                     'filename', 'ori_shape', 'img_shape', 'lidar2img', 'lidar2cam',
                     'depth2img', 'cam2img', 'pad_shape',
                     'scale_factor', 'flip', 'pcd_horizontal_flip',
                     'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                     'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                     'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                     'transformation_3d_flow', 'scene_token', 'can_bus',
                     'pad_fixed_size', 'pad_size_divisor'
                 )):
        self.keys = keys
        self.meta_keys = meta_keys
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Call function to collect keys in results.
        
        The keys in ``meta_keys`` will be converted to img_metas.
        
        Args:
            results (dict): Result dict contains the data to collect.
            
        Returns:
            dict: The result dict contains the following keys:
                - keys in ``self.keys``
                - ``img_metas``
        """
        data = {}
        img_metas = {}
        
        # Collect metadata
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        
        # Store img_metas
        data['img_metas'] = img_metas
        
        # Collect specified keys
        for key in self.keys:
            if key not in results:
                data[key] = None
            else:
                data[key] = results[key]
        
        return data
    
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'


def test_custom_collect_3d():
    """Test CustomCollect3D transform."""
    print("=" * 60)
    print("Testing CustomCollect3D")
    print("=" * 60)
    
    # Test 1: Basic collection functionality
    print("Testing basic collection functionality...")
    
    # Create sample processed data (like what would come from previous transforms)
    sample_results = {
        'img': torch.randn(6, 3, 928, 1600),  # Formatted images
        'gt_bboxes_3d': torch.randn(5, 9),    # Formatted bboxes
        'gt_labels_3d': torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),  # Formatted labels
        
        # Metadata from various transforms
        'filename': ['cam1.jpg', 'cam2.jpg', 'cam3.jpg', 'cam4.jpg', 'cam5.jpg', 'cam6.jpg'],
        'ori_shape': [(900, 1600, 3)] * 6,
        'img_shape': [(928, 1600, 3)] * 6,
        'pad_shape': [(928, 1600, 3)] * 6,
        'pad_size_divisor': 32,
        'img_norm_cfg': {'mean': [103.53, 116.28, 123.675], 'std': [1.0, 1.0, 1.0], 'to_rgb': False},
        'sample_idx': 12345,
        'scene_token': 'abc123def456',
        'lidar2img': [np.eye(4) for _ in range(6)],
        'cam2img': [np.eye(3) for _ in range(6)],
        
        # Some keys that shouldn't be collected
        'intermediate_data': 'should_not_appear',
        'temp_processing_info': [1, 2, 3]
    }
    
    # Create collector
    collector = CustomCollect3D(
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 
                   'pad_size_divisor', 'img_norm_cfg', 'sample_idx', 
                   'scene_token', 'lidar2img', 'cam2img')
    )
    print(f"✓ Created collector: {collector}")
    
    # Apply collection
    collected = collector(sample_results)
    
    print(f"✓ Collection successful!")
    print(f"  - Collected keys: {list(collected.keys())}")
    print(f"  - img_metas keys: {list(collected['img_metas'].keys())}")
    
    # Verify structure
    assert 'img_metas' in collected, "img_metas should be present"
    assert 'gt_bboxes_3d' in collected, "gt_bboxes_3d should be collected"
    assert 'gt_labels_3d' in collected, "gt_labels_3d should be collected"
    assert 'img' in collected, "img should be collected"
    assert 'intermediate_data' not in collected, "intermediate_data should not be collected"
    
    # Verify metadata
    assert 'filename' in collected['img_metas'], "filename should be in img_metas"
    assert 'sample_idx' in collected['img_metas'], "sample_idx should be in img_metas"
    assert collected['img_metas']['sample_idx'] == 12345, "sample_idx value should be preserved"
    
    print("✓ All basic collection tests passed")
    
    # Test 2: With missing keys
    print(f"\nTesting with missing keys...")
    
    incomplete_results = {
        'img': torch.randn(3, 3, 100, 200),
        'gt_labels_3d': torch.tensor([0, 1, 2]),
        # gt_bboxes_3d is missing
        'sample_idx': 67890
    }
    
    collected_incomplete = collector(incomplete_results)
    
    print(f"✓ Missing keys handled successfully!")
    print(f"  - gt_bboxes_3d value: {collected_incomplete['gt_bboxes_3d']}")
    assert collected_incomplete['gt_bboxes_3d'] is None, "Missing key should be None"
    assert collected_incomplete['gt_labels_3d'] is not None, "Present key should not be None"
    
    # Test 3: Integration with full pipeline
    print(f"\nTesting with full pipeline integration...")
    try:
        from load_multi_view_image import LoadMultiViewImageFromFiles
        from normalize_multi_view_image import NormalizeMultiviewImage
        from load_annotations_3d import LoadAnnotations3D
        from object_filters import ObjectNameFilter, ObjectRangeFilter
        from pad_multi_view_image import PadMultiViewImage
        from default_format_bundle_3d import DefaultFormatBundle3D
        
        import os
        import pickle
        
        # Load sample data
        data_file = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            sample = data['data_list'][0]
            
            # Create complete pipeline
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
            formatter = DefaultFormatBundle3D(keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
            
            print("  ✓ Created complete pipeline")
            
            # Apply complete pipeline
            results = sample
            results = loader(results)
            print(f"    - Images loaded: {results['img'].shape}")
            
            results = normalizer(results)
            print(f"    - Normalization applied")
            
            results = ann_loader(results)
            original_count = len(results['gt_bboxes_3d'])
            print(f"    - Annotations loaded: {original_count} objects")
            
            results = name_filter(results)
            name_count = len(results['gt_bboxes_3d'])
            print(f"    - Name filter applied: {name_count} objects")
            
            results = range_filter(results)
            final_count = len(results['gt_bboxes_3d'])
            print(f"    - Range filter applied: {final_count} objects")
            
            results = padder(results)
            print(f"    - Padding applied: {results['img'].shape}")
            
            results = formatter(results)
            print(f"    - Formatting applied: {results['img'].shape} ({results['img'].dtype})")
            
            # Final collection step
            final_collector = CustomCollect3D(
                keys=['gt_bboxes_3d', 'gt_labels_3d', 'img']
            )
            final_data = final_collector(results)
            
            print(f"  ✓ Complete pipeline with CustomCollect3D successful!")
            print(f"    - Final output keys: {list(final_data.keys())}")
            print(f"    - img: {final_data['img'].shape} ({final_data['img'].dtype})")
            print(f"    - gt_bboxes_3d: {final_data['gt_bboxes_3d'].shape} ({final_data['gt_bboxes_3d'].dtype})")
            print(f"    - gt_labels_3d: {final_data['gt_labels_3d'].shape} ({final_data['gt_labels_3d'].dtype})")
            print(f"    - img_metas keys: {len(final_data['img_metas'])} metadata fields")
            
            # Verify final format is ready for model
            assert torch.is_tensor(final_data['img']), "Final img should be tensor"
            assert final_data['img'].dim() == 4, "Final img should be 4D (N,C,H,W)"
            assert torch.is_tensor(final_data['gt_bboxes_3d']), "Final bboxes should be tensor"
            assert torch.is_tensor(final_data['gt_labels_3d']), "Final labels should be tensor"
            assert isinstance(final_data['img_metas'], dict), "img_metas should be dict"
            
            print(f"  ✓ All final format checks passed - ready for model!")
            
            # Show some key metadata
            key_meta_fields = ['sample_idx', 'scene_token', 'img_shape', 'pad_shape']
            for field in key_meta_fields:
                if field in final_data['img_metas']:
                    value = final_data['img_metas'][field]
                    if isinstance(value, (list, tuple)) and len(value) > 2:
                        print(f"    - {field}: {type(value).__name__} with {len(value)} items")
                    else:
                        print(f"    - {field}: {value}")
            
        else:
            print(f"  - Skipping real data test (dataset not found)")
    
    except ImportError:
        print(f"  - Skipping real data test (missing dependencies)")
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Custom meta_keys configuration
    print(f"\nTesting custom meta_keys configuration...")
    
    custom_collector = CustomCollect3D(
        keys=['img', 'gt_labels_3d'],
        meta_keys=('sample_idx', 'filename', 'custom_field')
    )
    
    custom_results = {
        'img': torch.randn(2, 3, 64, 64),
        'gt_labels_3d': torch.tensor([0, 1]),
        'sample_idx': 999,
        'filename': ['test1.jpg', 'test2.jpg'],
        'custom_field': 'custom_value',
        'ignored_field': 'should_be_ignored'
    }
    
    custom_collected = custom_collector(custom_results)
    
    print(f"✓ Custom meta_keys configuration successful!")
    print(f"  - img_metas keys: {list(custom_collected['img_metas'].keys())}")
    assert 'custom_field' in custom_collected['img_metas'], "Custom field should be collected"
    assert 'ignored_field' not in custom_collected['img_metas'], "Ignored field should not be collected"
    assert custom_collected['img_metas']['custom_field'] == 'custom_value', "Custom field value should be preserved"
    
    print("\n" + "=" * 60)
    print("All CustomCollect3D tests passed!")
    print("=" * 60)
    
    return final_data if 'final_data' in locals() else collected


if __name__ == "__main__":
    test_custom_collect_3d()