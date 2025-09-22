"""
NuScenes data converter for BEVFormer reimplementation
Generates temporal pickle files without mmcv/mmdet3d dependencies
"""

import os
import pickle
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Union, Dict, Any
from tqdm import tqdm
from pyquaternion import Quaternion

# NuScenes imports
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils import splits


# NuScenes categories for BEVFormer
nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

# NuScenes attributes
nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

# Class name mapping from NuScenes to BEVFormer format
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Mapping from NuScenes class names to BEVFormer class names
NUSCENES_TO_BEVFORMER = {
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'static_object.bicycle_rack': 'ignore',
    'movable_object.debris': 'ignore',
    'movable_object.pushable_pullable': 'ignore',
    'animal': 'ignore',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
}


def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        out_path (str): Path to save the info files.
        can_bus_root_path (str): Path to CAN bus data.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data. Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps. Default: 10
    """
    print(f"\n📂 Loading NuScenes dataset...")
    print(f"   Version: {version}")
    print(f"   Root path: {root_path}")

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    # Load CAN bus data if available
    try:
        nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
        print(f"✅ CAN bus data loaded from {can_bus_root_path}")
    except Exception as e:
        print(f"⚠️  CAN bus data not available: {e}")
        nusc_can_bus = None

    # Get train/val scene splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers, f"Version {version} not in {available_vers}"

    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError(f'Unknown version: {version}')

    # Filter existing scenes
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]

    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))

    # Convert scene names to tokens
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print(f'📊 Test scenes: {len(train_scenes)}')
    else:
        print(f'📊 Train scenes: {len(train_scenes)}, Val scenes: {len(val_scenes)}')

    # Generate info for train and val sets
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    # Save info files in the format expected by NuScenesDataset
    metadata = dict(version=version)

    if test:
        print(f'💾 Saving test samples: {len(train_nusc_infos)}')
        data = dict(data_list=train_nusc_infos, metainfo=metadata)
        info_path = os.path.join(out_path, f'{info_prefix}_infos_temporal_test.pkl')
        with open(info_path, 'wb') as f:
            pickle.dump(data, f)
        print(f'✅ Saved to {info_path}')
    else:
        print(f'💾 Saving train samples: {len(train_nusc_infos)}, val samples: {len(val_nusc_infos)}')

        # Save train info
        data = dict(data_list=train_nusc_infos, metainfo=metadata)
        info_path = os.path.join(out_path, f'{info_prefix}_infos_temporal_train.pkl')
        with open(info_path, 'wb') as f:
            pickle.dump(data, f)
        print(f'✅ Saved to {info_path}')

        # Save val info
        data = dict(data_list=val_nusc_infos, metainfo=metadata)
        info_val_path = os.path.join(out_path, f'{info_prefix}_infos_temporal_val.pkl')
        with open(info_val_path, 'wb') as f:
            pickle.dump(data, f)
        print(f'✅ Saved to {info_val_path}')


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of available scenes.
    """
    available_scenes = []
    print(f'Total scenes: {len(nusc.scene)}')

    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])

        has_more_frames = True
        scene_not_exist = False

        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)

            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path

            if not os.path.exists(lidar_path):
                # Check if at least one file exists for this scene
                if sd_rec['next'] == '':
                    has_more_frames = False
                else:
                    sd_rec = nusc.get('sample_data', sd_rec['next'])
            else:
                scene_not_exist = True
                break

        if scene_not_exist:
            available_scenes.append(scene)

    print(f'Available scenes: {len(available_scenes)}')
    return available_scenes


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    """Get CAN bus information for a sample.

    Args:
        nusc: NuScenes instance
        nusc_can_bus: NuScenes CAN bus API instance
        sample: Sample dictionary

    Returns:
        np.ndarray: CAN bus data (18 elements)
    """
    if nusc_can_bus is None:
        return np.zeros(18)

    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']

    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # Some scenes don't have CAN bus information

    can_bus = []
    # During each scene, the first timestamp of can_bus may be larger than the first sample's timestamp
    last_pose = pose_list[0]

    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose

    _ = last_pose.pop('utime')  # Remove timestamp
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')

    can_bus.extend(pos)
    can_bus.extend(rotation)

    for key in last_pose.keys():
        can_bus.extend(last_pose[key])  # Velocity, acceleration, etc.

    # Pad to 18 elements if needed
    while len(can_bus) < 18:
        can_bus.append(0.0)

    return np.array(can_bus[:18])


def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc: NuScenes instance
        nusc_can_bus: NuScenes CAN bus API instance
        train_scenes: Set of training scene tokens
        val_scenes: Set of validation scene tokens
        test: Whether in test mode
        max_sweeps: Max number of sweeps

    Returns:
        tuple: (train_infos, val_infos)
    """
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0

    print("\n🔄 Processing samples...")
    for sample in tqdm(nusc.sample, desc="Converting samples"):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        # Get CAN bus info
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)

        # Build info dictionary
        info = {
            'lidar_path': str(lidar_path),
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': frame_idx,
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        # Update frame index
        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        # Get transformation matrices
        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # Process 6 cameras
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]

        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                        e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # Get sweeps for temporal information
        sweeps = []
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                        l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break

        info['sweeps'] = sweeps

        # Get annotations if not in test mode
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]

            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                           for b in boxes]).reshape(-1, 1)

            # Get velocity
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])

            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)

            # Convert velocity from global to lidar coordinates
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]

            # Convert class names
            names = [b.name for b in boxes]
            converted_names = []
            for name in names:
                if name in NUSCENES_TO_BEVFORMER:
                    converted_name = NUSCENES_TO_BEVFORMER[name]
                    if converted_name != 'ignore':
                        converted_names.append(converted_name)
                    else:
                        converted_names.append(name)  # Keep original if ignore
                else:
                    converted_names.append(name)  # Keep original if not mapped

            names = np.array(converted_names)

            # Format gt_boxes: [x, y, z, w, l, h, rotation]
            # Note: rotation needs to be adjusted for coordinate system
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)

            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

        # Add to appropriate list
        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matrix from general sensor to Top LiDAR.

    Args:
        nusc: NuScenes instance
        sensor_token: Sample data token for the sensor
        l2e_t: Translation from lidar to ego
        l2e_r_mat: Rotation matrix from lidar to ego
        e2g_t: Translation from ego to global
        e2g_r_mat: Rotation matrix from ego to global
        sensor_type: Type of sensor ('lidar' or camera name)

    Returns:
        dict: Sensor information with transformations
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])

    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # Convert absolute to relative path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]

    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    # Calculate sensor to lidar transformation
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # Get rotation matrices
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

    # Compute sensor->ego->global->ego'->lidar transformation
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)

    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                 ) + l2e_t @ np.linalg.inv(l2e_r_mat).T

    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T

    return sweep