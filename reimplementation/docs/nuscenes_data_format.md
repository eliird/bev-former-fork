# nuScenes Dataset Format Documentation

## Overview

The nuScenes dataset used by BEVFormer is a temporal multi-camera 3D object detection dataset. This document describes the data structure and format for implementing data loading and transforms.

## Dataset Statistics

- **Version**: v1.0-mini (validation split)
- **Total Samples**: 404 temporal samples
- **Cameras per Sample**: 6 cameras (360° coverage)
- **Temporal Structure**: Linked sequences with prev/next relationships
- **3D Classes**: 9 nuScenes object classes

## Data Structure

### Top-Level Structure
```python
{
    'data_list': List[Dict],    # 404 samples
    'metainfo': {
        'version': 'v1.0-mini'
    }
}
```

### Individual Sample Structure

Each sample in `data_list` contains:

#### 1. Scene & Temporal Information
```python
{
    'token': str,           # Unique sample identifier (e.g., "ca9a282c9e77460f8360f564131a8af5")
    'scene_token': str,     # Scene identifier (multiple samples per scene)
    'prev': str,           # Previous frame token ("" for first frame)
    'next': str,           # Next frame token ("" for last frame)  
    'frame_idx': int,      # Frame index within sequence (0, 1, 2, ...)
    'timestamp': int,      # Unix timestamp (e.g., 1532402927647951)
}
```

#### 2. Vehicle State (CAN Bus Data)
```python
{
    'can_bus': np.ndarray(18,),  # Vehicle state vector
    # Structure: [x, y, z, quat(4), velocity(3), accel(3), rotation_rate(3), other(2)]
    # Example: [411.25, 1180.74, 0.0, 0.572, 0.0, 0.0, -0.820, -0.575, -0.215, 10.016, ...]
}
```

#### 3. Multi-Camera Data
```python
{
    'cams': {
        'CAM_FRONT': {
            'data_path': str,                      # Image file path
            'type': str,                           # Camera type
            'sample_data_token': str,              # Unique token for this camera sample
            'timestamp': int,                      # Camera timestamp
            
            # Camera Extrinsics
            'sensor2ego_translation': List[float],  # [x, y, z] translation
            'sensor2ego_rotation': List[float],     # [w, x, y, z] quaternion
            'ego2global_translation': List[float],  # Global position
            'ego2global_rotation': List[float],     # Global orientation
            
            # Camera Intrinsics & Transforms
            'cam_intrinsic': np.ndarray(3,3),      # Camera intrinsic matrix
            'sensor2lidar_rotation': np.ndarray(3,3),    # Rotation to LiDAR frame
            'sensor2lidar_translation': np.ndarray(3,),  # Translation to LiDAR frame
        },
        'CAM_FRONT_RIGHT': { ... },
        'CAM_FRONT_LEFT': { ... },
        'CAM_BACK': { ... },
        'CAM_BACK_LEFT': { ... },
        'CAM_BACK_RIGHT': { ... }
    }
}
```

#### 4. 3D Ground Truth Annotations
```python
{
    # 3D Bounding Boxes (N objects)
    'gt_boxes': np.ndarray(N, 7),        # [x, y, z, w, l, h, rotation] in LiDAR coordinates
    'gt_names': np.ndarray(N,),          # Class names (strings)
    'gt_velocity': np.ndarray(N, 2),     # [vx, vy] velocity in m/s
    
    # Metadata
    'valid_flag': np.ndarray(N,),        # Boolean mask for valid annotations
    'num_lidar_pts': np.ndarray(N,),     # Number of LiDAR points per object
    'num_radar_pts': np.ndarray(N,),     # Number of radar points per object
    
    # Additional fields
    'lidar_path': str,                    # LiDAR file path (not used in camera-only)
    'lidar2ego_translation': List[float], # LiDAR to ego transformation
    'lidar2ego_rotation': List[float],    # LiDAR to ego rotation
    'sweeps': List,                       # Additional LiDAR sweeps (empty for camera-only)
}
```

## Class Mapping

The dataset contains 9 nuScenes object classes:

| Index | GT Name | Config Class Name |
|-------|---------|------------------|
| 0 | `vehicle.car` | `car` |
| 1 | `vehicle.truck` | `truck` |
| 2 | `vehicle.construction` | `construction_vehicle` |
| 3 | `vehicle.bus.rigid` | `bus` |
| 4 | (trailer) | `trailer` |
| 5 | `movable_object.barrier` | `barrier` |
| 6 | (motorcycle) | `motorcycle` |
| 7 | `vehicle.bicycle` | `bicycle` |
| 8 | `human.pedestrian.adult` | `pedestrian` |
| 9 | `movable_object.trafficcone` | `traffic_cone` |

Note: Some classes may not appear in every sample.

## Coordinate Systems

### 1. LiDAR Coordinate System
- **Origin**: LiDAR sensor position
- **X-axis**: Forward (vehicle direction)
- **Y-axis**: Left
- **Z-axis**: Up
- **Units**: Meters

### 2. Camera Coordinate System  
- **Origin**: Camera optical center
- **X-axis**: Right (image width direction)
- **Y-axis**: Down (image height direction)  
- **Z-axis**: Forward (depth)

### 3. Global Coordinate System
- **Origin**: Map origin
- **Units**: Meters
- **Rotation**: Quaternions [w, x, y, z]

## Temporal Relationships

Samples are organized in temporal sequences:
- `prev` and `next` tokens link consecutive frames
- `scene_token` groups frames from the same driving sequence
- First frame in sequence has `prev = ""`
- Last frame in sequence has `next = ""`

Example temporal chain:
```
Sample 0: prev="" → token="ca9a282c..." → next="39586f9d..."
Sample 1: prev="ca9a282c..." → token="39586f9d..." → next="356d81f3..."
Sample 2: prev="39586f9d..." → token="356d81f3..." → next="e0845f53..."
```

## Camera Configuration

### Camera Layout
```
    [CAM_FRONT_LEFT]  [CAM_FRONT]  [CAM_FRONT_RIGHT]
           |              |              |
    [CAM_BACK_LEFT]   [VEHICLE]    [CAM_BACK_RIGHT]
           |              |              |
                    [CAM_BACK]
```

### Typical Image Resolution
- Standard nuScenes images: ~1600x900 pixels
- May vary between cameras and samples

### Camera Intrinsics Format
```python
intrinsic = [
    [fx,  0, cx],
    [ 0, fy, cy], 
    [ 0,  0,  1]
]
```

## Usage in BEVFormer

### Data Loading Pipeline
1. **LoadMultiViewImageFromFiles**: Load 6 camera images
2. **Normalization**: Apply ImageNet normalization
3. **Padding**: Pad to be divisible by 32 (for FPN)
4. **3D Annotation Loading**: Load gt_boxes, gt_names, etc.
5. **Filtering**: Apply range and class filters
6. **Format Bundle**: Convert to tensors for model

### Temporal Modeling
- BEVFormer uses temporal sequences (queue_length=4)
- Previous BEV features are maintained across frames
- CAN bus data provides ego motion for temporal alignment

### Model Input Format
The model expects:
```python
{
    'img': torch.Tensor(B, T, N, C, H, W),    # T=temporal, N=6 cameras
    'img_metas': List[List[Dict]],            # Metadata per frame
    'gt_bboxes_3d': List[Tensor],            # 3D boxes per sample
    'gt_labels_3d': List[Tensor],            # Labels per sample
}
```

## File System Structure
```
data/nuscenes/
├── samples/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   └── LIDAR_TOP/
└── nuscenes_infos_temporal_val.pkl
```

## Notes

- All 3D coordinates are in LiDAR frame unless otherwise specified
- Timestamps are in microseconds
- Quaternions follow [w, x, y, z] convention
- GT boxes format: [center_x, center_y, center_z, width, length, height, rotation]
- Rotation is around Z-axis (yaw) in radians