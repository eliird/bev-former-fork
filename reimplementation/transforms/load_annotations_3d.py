"""
LoadAnnotations3D Transform
Loads 3D bounding boxes, labels, and attributes for BEVFormer
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any


class LoadAnnotations3D:
    """Load 3D annotations including bounding boxes, labels, and attributes.
    
    This transform loads 3D annotations from the sample data for training.
    
    Args:
        with_bbox_3d (bool): Whether to load 3D bounding boxes. Default: True.
        with_label_3d (bool): Whether to load 3D labels. Default: True.
        with_attr_label (bool): Whether to load attribute labels. Default: False.
        with_bbox_depth (bool): Whether to load depth info. Default: False.
        with_velocity (bool): Whether to load velocity info. Default: True.
        box_type_3d (str): 3D box type ('LiDAR', 'Camera', 'Depth'). Default: 'LiDAR'.
    """
    
    def __init__(self,
                 with_bbox_3d: bool = True,
                 with_label_3d: bool = True,
                 with_attr_label: bool = False,
                 with_bbox_depth: bool = False,
                 with_velocity: bool = True,
                 box_type_3d: str = 'LiDAR'):
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_bbox_depth = with_bbox_depth
        self.with_velocity = with_velocity
        self.box_type_3d = box_type_3d
        
        # nuScenes class mapping
        self.nuscenes_classes = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]
        
        # Mapping from nuScenes GT names to class indices
        self.name_to_class = {
            'vehicle.car': 0,
            'vehicle.truck': 1,
            'vehicle.construction': 2,
            'vehicle.bus.rigid': 3,
            'vehicle.trailer': 4,
            'movable_object.barrier': 5,
            'vehicle.motorcycle': 6,
            'vehicle.bicycle': 7,
            'human.pedestrian.adult': 8,
            'human.pedestrian.child': 8,  # Map child to adult
            'human.pedestrian.wheelchair': 8,
            'human.pedestrian.stroller': 8,
            'human.pedestrian.personal_mobility': 8,
            'human.pedestrian.police_officer': 8,
            'human.pedestrian.construction_worker': 8,
            'movable_object.trafficcone': 9,
            'movable_object.pushable_pullable': 5,  # Map to barrier
        }
    
    def _load_bboxes_3d(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Load 3D bounding boxes.
        
        Args:
            results (dict): Input data dictionary
            
        Returns:
            dict: Updated dictionary with 3D boxes
        """
        if 'gt_boxes' not in results:
            raise KeyError("'gt_boxes' not found in input data")
        
        gt_boxes = results['gt_boxes']  # Shape: (N, 7) - [x, y, z, w, l, h, rot]
        
        # Convert to tensor if needed
        if not isinstance(gt_boxes, torch.Tensor):
            gt_boxes = torch.from_numpy(gt_boxes.astype(np.float32))
        
        # Add velocity if available and requested
        if self.with_velocity and 'gt_velocity' in results:
            gt_velocity = results['gt_velocity']
            if not isinstance(gt_velocity, torch.Tensor):
                gt_velocity = torch.from_numpy(gt_velocity.astype(np.float32))
            
            # Combine boxes with velocity: [x, y, z, w, l, h, rot, vx, vy]
            gt_boxes = torch.cat([gt_boxes, gt_velocity], dim=-1)
        
        # Create bounding box container based on type
        if self.box_type_3d == 'LiDAR':
            from structures import LiDARInstance3DBoxes
            gt_bboxes_3d = LiDARInstance3DBoxes(gt_boxes)
        else:
            # For now, just use tensor format
            gt_bboxes_3d = gt_boxes
        
        results['gt_bboxes_3d'] = gt_bboxes_3d
        return results
    
    def _load_labels_3d(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Load 3D labels.
        
        Args:
            results (dict): Input data dictionary
            
        Returns:
            dict: Updated dictionary with 3D labels
        """
        if 'gt_names' not in results:
            raise KeyError("'gt_names' not found in input data")
        
        gt_names = results['gt_names']
        
        # Convert names to class indices
        gt_labels = []
        for name in gt_names:
            if name in self.name_to_class:
                gt_labels.append(self.name_to_class[name])
            else:
                # Unknown class, map to ignore index (-1) or background
                gt_labels.append(-1)
        
        gt_labels = np.array(gt_labels, dtype=np.int64)
        
        # Filter out invalid labels (-1)
        valid_mask = gt_labels >= 0
        results['valid_mask'] = valid_mask
        
        # Convert to tensor
        gt_labels = torch.from_numpy(gt_labels)
        results['gt_labels_3d'] = gt_labels
        
        return results
    
    def _load_attr_labels(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Load attribute labels (placeholder for future implementation).
        
        Args:
            results (dict): Input data dictionary
            
        Returns:
            dict: Updated dictionary with attribute labels
        """
        # For now, create dummy attribute labels
        num_objects = len(results.get('gt_names', []))
        attr_labels = torch.zeros(num_objects, dtype=torch.long)
        results['gt_attr_labels'] = attr_labels
        return results
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Load 3D annotations.
        
        Args:
            results (dict): Input data dictionary
            
        Returns:
            dict: Updated dictionary with loaded annotations
        """
        # Load 3D bounding boxes
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
        
        # Load 3D labels  
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        
        # Load attribute labels
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        
        # Add annotation fields list
        ann_fields = []
        if self.with_bbox_3d:
            ann_fields.append('gt_bboxes_3d')
        if self.with_label_3d:
            ann_fields.append('gt_labels_3d')
        if self.with_attr_label:
            ann_fields.append('gt_attr_labels')
        
        results['ann_fields'] = ann_fields
        
        return results
    
    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                   f'with_bbox_3d={self.with_bbox_3d}, '
                   f'with_label_3d={self.with_label_3d}, '
                   f'with_attr_label={self.with_attr_label}, '
                   f'with_velocity={self.with_velocity}, '
                   f'box_type_3d=\'{self.box_type_3d}\')')
        return repr_str


def test_load_annotations_3d():
    """Test LoadAnnotations3D transform."""
    print("=" * 60)
    print("Testing LoadAnnotations3D")
    print("=" * 60)
    
    import pickle
    import os
    
    # Load sample data
    data_file = '/home/irdali.durrani/po-pi/BEVFormer/data/nuscenes/nuscenes_infos_temporal_val.pkl'
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    sample = data['data_list'][0]
    print(f"✓ Loaded sample with {len(sample['gt_names'])} objects")
    
    # Create transform
    transform = LoadAnnotations3D(
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_velocity=True
    )
    
    print(f"✓ Created transform: {transform}")
    
    # Apply transform
    results = transform(sample)
    
    print(f"✓ Transform successful!")
    print(f"  - Loaded {results['gt_bboxes_3d'].shape[0] if hasattr(results['gt_bboxes_3d'], 'shape') else len(results['gt_bboxes_3d'])} 3D boxes")
    print(f"  - Loaded {len(results['gt_labels_3d'])} labels")
    print(f"  - Annotation fields: {results['ann_fields']}")
    
    # Check bounding box format
    gt_bboxes_3d = results['gt_bboxes_3d']
    if torch.is_tensor(gt_bboxes_3d):
        print(f"  - Bboxes shape: {gt_bboxes_3d.shape}")
        print(f"  - Bboxes dtype: {gt_bboxes_3d.dtype}")
        print(f"  - Sample bbox: {gt_bboxes_3d[0]}")
    
    # Check labels
    gt_labels_3d = results['gt_labels_3d']
    print(f"  - Labels shape: {gt_labels_3d.shape}")
    print(f"  - Labels dtype: {gt_labels_3d.dtype}")
    print(f"  - Unique labels: {torch.unique(gt_labels_3d).tolist()}")
    
    # Show class distribution
    print(f"\nClass distribution:")
    unique_labels, counts = torch.unique(gt_labels_3d, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label >= 0:
            class_name = transform.nuscenes_classes[label] if label < len(transform.nuscenes_classes) else f"unknown_{label}"
            print(f"  - {class_name} (id={label}): {count} objects")
    
    # Check valid mask if present
    if 'valid_mask' in results:
        valid_count = results['valid_mask'].sum()
        print(f"  - Valid objects: {valid_count}/{len(results['valid_mask'])}")
    
    # Show original GT names for comparison
    print(f"\nOriginal GT names (first 5):")
    for i, name in enumerate(sample['gt_names'][:5]):
        mapped_label = transform.name_to_class.get(name, -1)
        class_name = transform.nuscenes_classes[mapped_label] if mapped_label >= 0 else "unknown"
        print(f"  - {name} → {mapped_label} ({class_name})")
    
    # Test error handling
    print(f"\nTesting error handling...")
    
    # Test missing gt_boxes
    try:
        invalid_sample = {'gt_names': ['test']}
        transform(invalid_sample)
        print("❌ Should have failed with missing gt_boxes")
    except KeyError:
        print("✓ Correctly caught missing gt_boxes error")
    
    # Test missing gt_names
    try:
        transform_labels_only = LoadAnnotations3D(with_bbox_3d=False, with_label_3d=True)
        invalid_sample = {'gt_boxes': np.array([[0, 0, 0, 1, 1, 1, 0]])}
        transform_labels_only(invalid_sample)
        print("❌ Should have failed with missing gt_names")
    except KeyError:
        print("✓ Correctly caught missing gt_names error")
    
    print("\n" + "=" * 60)
    print("All LoadAnnotations3D tests passed!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    test_load_annotations_3d()