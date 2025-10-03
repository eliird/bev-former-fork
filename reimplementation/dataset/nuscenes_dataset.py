"""
Pure PyTorch implementation of nuScenes dataset for BEVFormer
Handles temporal sequences, multi-view images, and CAN bus data without MMDetection dependencies
"""

import os
import pickle
import copy
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any, Tuple

# Import our reimplemented transforms
from transforms.load_multi_view_image import LoadMultiViewImageFromFiles
from transforms.normalize_multi_view_image import NormalizeMultiviewImage
from transforms.photometricdistortion_multiview import PhotoMetricDistortionMultiViewImage
from transforms.load_annotations_3d import LoadAnnotations3D
from transforms.object_filters import ObjectNameFilter, ObjectRangeFilter
from transforms.pad_multi_view_image import PadMultiViewImage
from transforms.default_format_bundle_3d import DefaultFormatBundle3D
from transforms.custom_collect_3d import CustomCollect3D


class NuScenesDataset(Dataset):
    """
    PyTorch Dataset for nuScenes temporal multi-view 3D object detection
    
    This dataset handles:
    - Temporal sequences with configurable queue_length (default 4)
    - Multi-view camera images (6 cameras per frame)
    - 3D bounding box annotations with velocity
    - CAN bus data for ego motion
    - Data augmentation pipeline
    
    Args:
        data_file (str): Path to nuScenes pickle file (e.g., nuscenes_infos_temporal_val.pkl)
        transforms (list): List of transform objects to apply
        queue_length (int): Number of temporal frames per sequence (default 4)
        training (bool): Whether in training mode (affects augmentation)
        point_cloud_range (list): 3D detection range [x_min, y_min, z_min, x_max, y_max, z_max]
        class_names (list): List of class names for detection
    """
    
    def __init__(self,
                 data_file: str,
                 transforms: Optional[List] = None,
                 queue_length: int = 4,
                 training: bool = True,
                 point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 class_names: List[str] = [
                     'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                     'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
                 ],
                 use_cpu_augmentation: bool = True):
        
        self.data_file = data_file
        self.queue_length = queue_length
        self.training = training
        self.point_cloud_range = point_cloud_range
        self.class_names = class_names
        self.use_cpu_augmentation = use_cpu_augmentation
        
        # Load the dataset
        print(f"Loading nuScenes dataset from {data_file}...")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Dataset file not found: {data_file}")
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        self.data_list = data['data_list']
        self.metainfo = data.get('metainfo', {})
        
        print(f"✅ Loaded {len(self.data_list)} samples")
        
        # Build scene-based index for temporal sequences
        self._build_scene_index()
        
        # Build temporal sequences
        self._build_temporal_sequences()
        
        # Setup transforms
        if transforms is None:
            transforms = self._create_default_transforms()
        self.transforms = transforms
        
        print(f"✅ Built {len(self.temporal_sequences)} temporal sequences")
    
    def _build_scene_index(self):
        """Build index of samples grouped by scene for temporal consistency."""
        self.scene_to_samples = {}
        
        for idx, sample in enumerate(self.data_list):
            scene_token = sample.get('scene_token', 'unknown')
            if scene_token not in self.scene_to_samples:
                self.scene_to_samples[scene_token] = []
            self.scene_to_samples[scene_token].append(idx)
        
        # Sort samples within each scene by frame_idx for temporal order
        for scene_token in self.scene_to_samples:
            samples = self.scene_to_samples[scene_token]
            # Sort by frame_idx if available, otherwise by token
            samples.sort(key=lambda idx: self.data_list[idx].get('frame_idx', 0))
            self.scene_to_samples[scene_token] = samples
    
    def _build_temporal_sequences(self):
        """Build valid temporal sequences for training/testing."""
        self.temporal_sequences = []
        
        for scene_token, sample_indices in self.scene_to_samples.items():
            # For each scene, create overlapping temporal sequences
            scene_length = len(sample_indices)
            
            if scene_length < self.queue_length:
                # Scene too short, pad with duplicates of the last frame
                if scene_length > 0:
                    padded_sequence = sample_indices + [sample_indices[-1]] * (self.queue_length - scene_length)
                    self.temporal_sequences.append({
                        'indices': padded_sequence,
                        'scene_token': scene_token,
                        'is_padded': True
                    })
            else:
                # Create sliding window sequences
                for start_idx in range(scene_length - self.queue_length + 1):
                    sequence_indices = sample_indices[start_idx:start_idx + self.queue_length]
                    self.temporal_sequences.append({
                        'indices': sequence_indices,
                        'scene_token': scene_token,
                        'is_padded': False
                    })
    
    def _create_default_transforms(self):
        """Create default transform pipeline matching BEVFormer configuration."""
        img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
        
        transforms = []
        transforms.append(LoadMultiViewImageFromFiles(to_float32=True))

        # Only add CPU photometric distortion if not using GPU augmentation
        if self.training and self.use_cpu_augmentation:
            transforms.append(PhotoMetricDistortionMultiViewImage())

        transforms.append(LoadAnnotations3D(with_bbox_3d=True, with_label_3d=True, with_attr_label=False))
        transforms.append(ObjectRangeFilter(point_cloud_range=self.point_cloud_range))
        transforms.append(ObjectNameFilter(classes=self.class_names))
        transforms.append(NormalizeMultiviewImage(**img_norm_cfg))
        transforms.append(PadMultiViewImage(size_divisor=32))
        transforms.append(DefaultFormatBundle3D(keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'], class_names=self.class_names))
        transforms.append(CustomCollect3D(keys=['gt_bboxes_3d', 'gt_labels_3d', 'img']))
        
        return transforms
    
    def __len__(self) -> int:
        """Return number of temporal sequences."""
        return len(self.temporal_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a temporal sequence of frames for training/inference.
        
        Args:
            idx: Index of the temporal sequence
            
        Returns:
            dict: Processed temporal sequence with:
                - img: torch.Tensor(T, N, C, H, W) - temporal, multi-view images
                - img_metas: List[Dict] - metadata for each frame
                - gt_bboxes_3d: torch.Tensor - 3D bounding boxes (training only)
                - gt_labels_3d: torch.Tensor - class labels (training only)
        """
        sequence_info = self.temporal_sequences[idx]
        sample_indices = sequence_info['indices']
        scene_token = sequence_info['scene_token']
        
        # Process each frame in the temporal sequence
        temporal_frames = []
        prev_ego_pose = None
        
        for frame_idx, sample_idx in enumerate(sample_indices):
            sample = copy.deepcopy(self.data_list[sample_idx])
            
            # Add temporal and scene information
            sample['scene_token'] = scene_token
            sample['frame_idx_in_sequence'] = frame_idx
            sample['is_first_frame'] = (frame_idx == 0)
            sample['is_last_frame'] = (frame_idx == len(sample_indices) - 1)
            
            # Compute relative ego motion for temporal alignment
            if prev_ego_pose is not None and 'can_bus' in sample:
                current_pose = sample['can_bus'][:3]  # [x, y, z]
                current_angle = sample['can_bus'][-1]  # yaw angle
                
                # Compute relative motion
                relative_translation = current_pose - prev_ego_pose[:3]
                relative_angle = current_angle - prev_ego_pose[-1]
                
                # Store relative motion in can_bus
                sample['can_bus'][:3] = relative_translation
                sample['can_bus'][-1] = relative_angle
                sample['prev_bev_exists'] = True
            else:
                # First frame or missing CAN bus data
                if 'can_bus' in sample:
                    sample['can_bus'][:3] = 0.0  # No relative motion
                    sample['can_bus'][-1] = 0.0
                sample['prev_bev_exists'] = False
            
            # Store current pose for next frame
            if 'can_bus' in sample:
                prev_ego_pose = self.data_list[sample_idx]['can_bus'].copy()
            
            # Apply transform pipeline
            processed_sample = self._apply_transforms(sample)
            if processed_sample is None:
                # Handle failed transforms by using a fallback
                print(f"⚠️ Failed to process sample {sample_idx}, using fallback")
                processed_sample = self._create_fallback_sample(sample)
            
            temporal_frames.append(processed_sample)
        
        # Stack temporal frames into final format
        return self._stack_temporal_frames(temporal_frames)
    
    def _apply_transforms(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply the transform pipeline to a single sample."""
        result = sample.copy()
        
        # Store temporal information that should be preserved
        temporal_info = {
            'scene_token': sample.get('scene_token', 'unknown'),
            'frame_idx_in_sequence': sample.get('frame_idx_in_sequence', -1),
            'is_first_frame': sample.get('is_first_frame', False),
            'is_last_frame': sample.get('is_last_frame', False),
            'prev_bev_exists': sample.get('prev_bev_exists', False)
        }
        
        for transform in self.transforms:
            try:
                result = transform(result)
                if result is None:
                    return None
            except Exception as e:
                print(f"⚠️ Transform {transform.__class__.__name__} failed: {e}")
                return None
        
        # Ensure temporal information is preserved in img_metas
        if 'img_metas' in result:
            result['img_metas'].update(temporal_info)
        
        return result
    
    def _create_fallback_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback sample when transform pipeline fails."""
        # Create minimal sample with dummy data
        H, W = 900, 1600  # Standard nuScenes image size
        N_cams = 6
        
        fallback = {
            'img': torch.zeros(N_cams, 3, H, W),
            'gt_bboxes_3d': torch.zeros(0, 9),  # Empty bboxes
            'gt_labels_3d': torch.zeros(0, dtype=torch.long),
            'img_metas': {
                'scene_token': sample.get('scene_token', 'unknown'),
                'can_bus': np.zeros(18),
                'prev_bev_exists': sample.get('prev_bev_exists', False),
                'frame_idx_in_sequence': sample.get('frame_idx_in_sequence', 0)
            }
        }
        
        return fallback
    
    def _stack_temporal_frames(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stack temporal frames into final batch format."""
        T = len(frames)
        
        # Stack images: (T, N, C, H, W)
        imgs = torch.stack([frame['img'] for frame in frames], dim=0)
        
        # Collect metadata for each frame
        img_metas = [frame['img_metas'] for frame in frames]
        
        # For training, use GT from the last frame (current frame)
        result = {
            'img': imgs,
            'img_metas': img_metas
        }
        
        # Always include GT data if available (needed for both training and validation)
        if 'gt_bboxes_3d' in frames[-1]:
            result['gt_bboxes_3d'] = frames[-1]['gt_bboxes_3d']
            result['gt_labels_3d'] = frames[-1]['gt_labels_3d']
        
        return result
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a temporal sequence."""
        sequence_info = self.temporal_sequences[idx]
        sample_indices = sequence_info['indices']
        
        info = {
            'sequence_idx': idx,
            'scene_token': sequence_info['scene_token'],
            'num_frames': len(sample_indices),
            'is_padded': sequence_info['is_padded'],
            'sample_tokens': [self.data_list[i]['token'] for i in sample_indices]
        }
        
        return info


def test_nuscenes_dataset():
    """Test the NuScenesDataset implementation."""
    print("=" * 80)
    print("TESTING NUSCENES DATASET")
    print("=" * 80)
    
    # Test dataset loading
    data_file = '/home/irdali.durrani/po-pi/BEVFormer/data/nuscenes/nuscenes_infos_temporal_val.pkl'
    if not os.path.exists(data_file):
        print("❌ Dataset file not found!")
        return
    
    print("🔄 Creating dataset...")
    dataset = NuScenesDataset(
        data_file=data_file,
        queue_length=4,
        training=True
    )
    
    print(f"✅ Dataset created with {len(dataset)} temporal sequences")
    
    # Test __getitem__
    print("\n🔄 Testing __getitem__...")
    sample = dataset[0]
    
    print(f"✅ Sample loaded successfully!")
    print(f"   - img shape: {sample['img'].shape}")  # Should be (T, N, C, H, W)
    print(f"   - img_metas length: {len(sample['img_metas'])}")
    if 'gt_bboxes_3d' in sample:
        print(f"   - gt_bboxes_3d shape: {sample['gt_bboxes_3d'].shape}")
        print(f"   - gt_labels_3d shape: {sample['gt_labels_3d'].shape}")
    
    # Test sequence info
    print("\n🔄 Testing sequence info...")
    info = dataset.get_sample_info(0)
    print(f"✅ Sequence info: {info}")
    
    # Test multiple samples
    print("\n🔄 Testing multiple samples...")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        info = dataset.get_sample_info(i)
        print(f"   Sample {i}: {sample['img'].shape}, scene={info['scene_token'][:8]}...")
    
    print(f"\n🎉 NuScenesDataset test successful!")
    return dataset


if __name__ == "__main__":
    test_nuscenes_dataset()