"""
Object filtering transforms for 3D detection
Filters objects by class names and spatial ranges
"""

import numpy as np
import torch
from typing import Dict, List, Any
from structures import LiDARInstance3DBoxes


class ObjectNameFilter:
    """Filter GT objects by their names.

    This transform filters 3D bounding boxes and labels to keep only
    objects belonging to specified classes.

    Required Keys:
    - gt_labels_3d

    Modified Keys:
    - gt_labels_3d
    - gt_bboxes_3d

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes: List[str]) -> None:
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict: dict) -> dict:
        """Filter objects by their class names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
            keys are updated in the result dict.
        """
        if 'gt_labels_3d' not in input_dict:
            return input_dict
            
        gt_labels_3d = input_dict['gt_labels_3d']
        
        # Convert to numpy if tensor
        if torch.is_tensor(gt_labels_3d):
            labels_np = gt_labels_3d.cpu().numpy()
        else:
            labels_np = gt_labels_3d
        
        # Create mask for valid labels
        gt_bboxes_mask = np.array([n in self.labels for n in labels_np], dtype=bool)
        
        # Apply mask to bboxes and labels
        if 'gt_bboxes_3d' in input_dict:
            input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        
        # Apply mask to labels
        if torch.is_tensor(gt_labels_3d):
            input_dict['gt_labels_3d'] = gt_labels_3d[torch.from_numpy(gt_bboxes_mask)]
        else:
            input_dict['gt_labels_3d'] = gt_labels_3d[gt_bboxes_mask]

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str
    

class ObjectRangeFilter:
    """Filter objects by the range.

    This transform filters 3D bounding boxes to keep only objects
    within the specified spatial range (typically BEV range).

    Required Keys:
    - gt_bboxes_3d

    Modified Keys:
    - gt_bboxes_3d
    - gt_labels_3d

    Args:
        point_cloud_range (list[float]): Point cloud range in format
            [x_min, y_min, z_min, x_max, y_max, z_max].
    """

    def __init__(self, point_cloud_range: List[float]) -> None:
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict: dict) -> dict:
        """Filter objects by spatial range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
            keys are updated in the result dict.
        """
        if 'gt_bboxes_3d' not in input_dict:
            return input_dict
            
        # For LiDAR coordinate system (nuScenes): x, y, z, x, y, z -> x_min, y_min, x_max, y_max
        bev_range = self.pcd_range[[0, 1, 3, 4]]  # x_min, y_min, x_max, y_max

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        
        # Get object centers and check if they're within BEV range
        if hasattr(gt_bboxes_3d, 'gravity_center'):
            centers = gt_bboxes_3d.gravity_center  # Shape: (N, 3)
        else:
            # Fallback if using tensor format
            if torch.is_tensor(gt_bboxes_3d):
                centers = gt_bboxes_3d[:, :3]  # x, y, z
            else:
                centers = gt_bboxes_3d[:, :3]
        
        # Create BEV range mask
        x_min, y_min, x_max, y_max = bev_range
        mask = (
            (centers[:, 0] >= x_min) & (centers[:, 0] <= x_max) &
            (centers[:, 1] >= y_min) & (centers[:, 1] <= y_max)
        )
        
        # Apply mask to bboxes
        gt_bboxes_3d = gt_bboxes_3d[mask]
        
        # Apply mask to labels if present
        if 'gt_labels_3d' in input_dict:
            gt_labels_3d = input_dict['gt_labels_3d']
            
            # Handle label indexing safely
            if torch.is_tensor(gt_labels_3d):
                input_dict['gt_labels_3d'] = gt_labels_3d[mask]
            else:
                # Convert mask to numpy if needed
                if torch.is_tensor(mask):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = mask
                input_dict['gt_labels_3d'] = gt_labels_3d[mask_np]
        
        # Limit yaw to [-pi, pi] if boxes have yaw
        if hasattr(gt_bboxes_3d, 'yaw') and len(gt_bboxes_3d) > 0:
            yaw = gt_bboxes_3d.yaw
            # Normalize yaw to [-pi, pi]
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            if hasattr(gt_bboxes_3d, 'tensor'):
                gt_bboxes_3d.tensor[:, 6] = yaw
        
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


def test_object_filters():
    """Test ObjectNameFilter and ObjectRangeFilter transforms."""
    print("=" * 60)
    print("Testing Object Filters")
    print("=" * 60)
    
    import os
    import pickle
    from load_annotations_3d import LoadAnnotations3D
    
    # Load sample data
    data_file = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    sample = data['data_list'][0]
    print(f"✓ Loaded sample with {len(sample['gt_names'])} original objects")
    
    # Load annotations first
    loader = LoadAnnotations3D()
    results = loader(sample)
    
    original_count = len(results['gt_bboxes_3d'])
    print(f"✓ Loaded {original_count} 3D annotations")
    
    # Test ObjectNameFilter
    print(f"\nTesting ObjectNameFilter...")
    
    # Define nuScenes classes (subset for testing)
    test_classes = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
                   'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
    
    name_filter = ObjectNameFilter(classes=test_classes)
    print(f"✓ Created name filter: {name_filter}")
    
    # Apply name filter
    results_name_filtered = name_filter(results.copy())
    name_filtered_count = len(results_name_filtered['gt_bboxes_3d'])
    
    print(f"✓ Name filtering successful!")
    print(f"  - Original objects: {original_count}")
    print(f"  - After name filter: {name_filtered_count}")
    print(f"  - Filtered out: {original_count - name_filtered_count}")
    
    # Check label distribution after name filtering
    labels = results_name_filtered['gt_labels_3d']
    if torch.is_tensor(labels):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = labels
    
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    print(f"  - Remaining label distribution:")
    for label, count in zip(unique_labels, counts):
        if label >= 0 and label < len(test_classes):
            print(f"    - {test_classes[label]} (id={label}): {count} objects")
    
    # Test ObjectRangeFilter
    print(f"\nTesting ObjectRangeFilter...")
    
    # nuScenes typical range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    range_filter = ObjectRangeFilter(point_cloud_range=point_cloud_range)
    print(f"✓ Created range filter: {range_filter}")
    
    # Apply range filter to name-filtered results
    results_range_filtered = range_filter(results_name_filtered.copy())
    range_filtered_count = len(results_range_filtered['gt_bboxes_3d'])
    
    print(f"✓ Range filtering successful!")
    print(f"  - After name filter: {name_filtered_count}")
    print(f"  - After range filter: {range_filtered_count}")
    print(f"  - Filtered out by range: {name_filtered_count - range_filtered_count}")
    
    # Check spatial distribution
    if range_filtered_count > 0:
        bboxes = results_range_filtered['gt_bboxes_3d']
        centers = bboxes.gravity_center if hasattr(bboxes, 'gravity_center') else bboxes[:, :3]
        
        if torch.is_tensor(centers):
            centers_np = centers.cpu().numpy()
        else:
            centers_np = centers
        
        print(f"  - Spatial statistics:")
        print(f"    - X range: [{centers_np[:, 0].min():.1f}, {centers_np[:, 0].max():.1f}]")
        print(f"    - Y range: [{centers_np[:, 1].min():.1f}, {centers_np[:, 1].max():.1f}]")
        print(f"    - Z range: [{centers_np[:, 2].min():.1f}, {centers_np[:, 2].max():.1f}]")
    
    # Test with tighter range
    print(f"\nTesting with tighter range...")
    tight_range = [-25.0, -25.0, -2.0, 25.0, 25.0, 2.0]  # Smaller range
    tight_range_filter = ObjectRangeFilter(point_cloud_range=tight_range)
    
    results_tight = tight_range_filter(results_name_filtered.copy())
    tight_count = len(results_tight['gt_bboxes_3d'])
    
    print(f"✓ Tight range filter: {tight_count} objects remaining")
    print(f"  - Original: {original_count} → Name: {name_filtered_count} → Tight Range: {tight_count}")
    
    # Test error handling
    print(f"\nTesting error handling...")
    
    # Test with missing keys
    empty_results = {}
    name_filtered_empty = name_filter(empty_results.copy())
    range_filtered_empty = range_filter(empty_results.copy())
    print("✓ Correctly handled missing keys")
    
    # Test with empty bboxes
    empty_bbox_results = {
        'gt_bboxes_3d': LiDARInstance3DBoxes(torch.empty(0, 9)),
        'gt_labels_3d': torch.empty(0, dtype=torch.long)
    }
    name_filtered_empty_bbox = name_filter(empty_bbox_results.copy())
    range_filtered_empty_bbox = range_filter(empty_bbox_results.copy())
    print("✓ Correctly handled empty bboxes")
    
    print("\n" + "=" * 60)
    print("All Object Filter tests passed!")
    print("=" * 60)
    
    return results_range_filtered


def test_filter_pipeline():
    """Test complete filtering pipeline integration."""
    print("\n" + "=" * 40)
    print("Testing Filter Pipeline Integration")
    print("=" * 40)
    
    try:
        from load_multi_view_image import LoadMultiViewImageFromFiles
        from normalize_multi_view_image import NormalizeMultiviewImage
        from photometricdistortion_multiview import PhotoMetricDistortionMultiViewImage
        from load_annotations_3d import LoadAnnotations3D
        
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
        
        # Create complete pipeline
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
        
        print("✓ Created complete pipeline")
        
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
        
        print(f"✓ Complete pipeline successful!")
        print(f"  - Final data shapes:")
        print(f"    - Images: {results['img'].shape}")
        print(f"    - Bboxes: {results['gt_bboxes_3d'].shape}")
        print(f"    - Labels: {results['gt_labels_3d'].shape}")
        
    except ImportError as e:
        print(f"❌ Pipeline test failed (missing modules): {e}")
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test filters individually 
    test_object_filters()
    
    # Test complete pipeline
    test_filter_pipeline()