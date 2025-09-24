# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
# Reimplemented without mmcv dependencies

import numpy as np
import os
import pickle
import logging
import time
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
from tqdm import tqdm
import cv2  # Replacing mmcv.imread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# from mmdet3d.core.bbox.box_np_ops import points_cam2img
# Replace with local implementation
def points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project 3D points to 2D image plane."""
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    points_4 = np.concatenate([points_3d, np.ones(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        depth = point_2d[..., 2]
        return np.concatenate([point_2d_res, depth[..., None]], axis=-1)
    return point_2d_res

from nuscenes.utils import splits

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')


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
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    start_time = time.time()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus

    logger.info("=" * 80)
    logger.info("NuScenes Data Conversion Starting")
    logger.info("=" * 80)
    logger.info(f"Version: {version}")
    logger.info(f"Root path: {root_path}")
    logger.info(f"Output path: {out_path}")
    logger.info(f"CAN bus path: {can_bus_root_path}")
    logger.info(f"Max sweeps: {max_sweeps}")

    # Load NuScenes dataset
    logger.info("\n📂 Loading NuScenes dataset...")
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    # Load CAN bus data
    logger.info("\n📂 Loading CAN bus data...")
    try:
        nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
        logger.info("✅ CAN bus data loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️  CAN bus data not available: {e}")
        logger.warning("   Proceeding without CAN bus data...")
        nusc_can_bus = None

    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers, f"Version {version} not supported"

    # Get scene splits
    logger.info("\n📊 Getting scene splits...")
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
        logger.info(f"   Train scenes (predefined): {len(train_scenes)}")
        logger.info(f"   Val scenes (predefined): {len(val_scenes)}")
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
        logger.info(f"   Test scenes: {len(train_scenes)}")
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
        logger.info(f"   Mini train scenes: {len(train_scenes)}")
        logger.info(f"   Mini val scenes: {len(val_scenes)}")
    else:
        raise ValueError(f'Unknown version: {version}')

    # Filter existing scenes
    logger.info("\n🔍 Checking scene availability...")
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]

    # Filter train/val scenes based on availability
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))

    logger.info(f"   Available train scenes: {len(train_scenes)}")
    logger.info(f"   Available val scenes: {len(val_scenes)}")

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
        logger.info(f'\n📊 Final test scenes: {len(train_scenes)}')
    else:
        logger.info(f'\n📊 Final train scenes: {len(train_scenes)}, val scenes: {len(val_scenes)}')

    # Generate infos
    logger.info("\n🔄 Generating sample infos...")
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    # Use the format expected by NuScenesDataset class
    metadata = dict(version=version)

    # Save pickle files in the format expected by NuScenesDataset
    logger.info("\n💾 Saving pickle files...")
    if test:
        logger.info(f'   Test samples: {len(train_nusc_infos)}')
        # Use data_list key as expected by NuScenesDataset
        data = dict(data_list=train_nusc_infos, metainfo=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_test.pkl'.format(info_prefix))
        logger.info(f'   Saving to: {info_path}')
        with open(info_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f'   ✅ Test file saved')
    else:
        logger.info(f'   Train samples: {len(train_nusc_infos)}')
        logger.info(f'   Val samples: {len(val_nusc_infos)}')

        # Save train info - use data_list key as expected by NuScenesDataset
        data = dict(data_list=train_nusc_infos, metainfo=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_train.pkl'.format(info_prefix))
        logger.info(f'   Saving train to: {info_path}')
        with open(info_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f'   ✅ Train file saved')

        # Save val info - use data_list key
        data = dict(data_list=val_nusc_infos, metainfo=metadata)
        info_val_path = osp.join(out_path,
                                 '{}_infos_temporal_val.pkl'.format(info_prefix))
        logger.info(f'   Saving val to: {info_val_path}')
        with open(info_val_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f'   ✅ Val file saved')

    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ Data conversion completed in {elapsed_time:.2f} seconds")
    logger.info("=" * 80)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    total_scenes = len(nusc.scene)
    logger.info(f'   Total scenes in dataset: {total_scenes}')
    logger.info(f'   🎥 Using CAMERA-ONLY mode (no LIDAR validation)')

    scenes_with_data = 0
    scenes_without_data = 0

    # Progress bar for scene validation
    with tqdm(total=total_scenes, desc="   Validating scenes", unit="scene") as pbar:
        for scene in nusc.scene:
            scene_token = scene['token']
            scene_rec = nusc.get('scene', scene_token)
            sample_rec = nusc.get('sample', scene_rec['first_sample_token'])

            # Check for camera data instead of LIDAR
            scene_exists = False

            # Check if at least one camera image exists
            camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                          'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

            for cam in camera_types:
                if cam in sample_rec['data']:
                    cam_token = sample_rec['data'][cam]
                    sd_rec = nusc.get('sample_data', cam_token)
                    cam_path = nusc.get_sample_data_path(cam_token)
                    cam_path = str(cam_path)

                    if os.getcwd() in cam_path:
                        cam_path = cam_path.split(f'{os.getcwd()}/')[-1]

                    if osp.exists(cam_path):
                        scene_exists = True
                        break

            # Include scene if camera data exists
            if not scene_exists:
                scenes_without_data += 1
                pbar.set_postfix({'valid': scenes_with_data, 'invalid': scenes_without_data})
                pbar.update(1)
                continue

            available_scenes.append(scene)
            scenes_with_data += 1
            pbar.set_postfix({'valid': scenes_with_data, 'invalid': scenes_without_data})
            pbar.update(1)

    logger.info(f'   ✅ Scenes with valid camera data: {scenes_with_data}')
    if scenes_without_data > 0:
        logger.warning(f'   ⚠️  Scenes without camera data: {scenes_without_data}')

    return available_scenes


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)


def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0

    total_samples = len(nusc.sample)
    logger.info(f"   Total samples to process: {total_samples}")

    # Statistics counters
    train_count = 0
    val_count = 0
    skipped_count = 0
    errors_count = 0

    # Use tqdm for progress tracking with detailed stats
    with tqdm(total=total_samples, desc="   Processing samples", unit="sample") as pbar:
        for sample in nusc.sample:
            try:
                lidar_token = sample['data']['LIDAR_TOP']
                sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                cs_record = nusc.get('calibrated_sensor',
                                     sd_rec['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

                # For camera-only mode, we don't require LIDAR files
                # Just log if missing but continue processing
                if not osp.exists(lidar_path):
                    logger.debug(f"LIDAR file not found for sample {sample['token']}, continuing with camera-only")

                can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)

                # Build info dictionary
                info = {
                    'lidar_path': lidar_path,
                    'token': sample['token'],
                    'prev': sample['prev'],
                    'next': sample['next'],
                    'can_bus': can_bus,
                    'frame_idx': frame_idx,  # temporal related info
                    'sweeps': [],
                    'cams': dict(),
                    'scene_token': sample['scene_token'],  # temporal related info
                    'lidar2ego_translation': cs_record['translation'],
                    'lidar2ego_rotation': cs_record['rotation'],
                    'ego2global_translation': pose_record['translation'],
                    'ego2global_rotation': pose_record['rotation'],
                    'timestamp': sample['timestamp'],
                }

                if sample['next'] == '':
                    frame_idx = 0
                else:
                    frame_idx += 1

                l2e_r = info['lidar2ego_rotation']
                l2e_t = info['lidar2ego_translation']
                e2g_r = info['ego2global_rotation']
                e2g_t = info['ego2global_translation']
                l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                # obtain 6 image's information per frame
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

                # obtain sweeps for a single key-frame
                sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                sweeps = []
                while len(sweeps) < max_sweeps:
                    if not sd_rec['prev'] == '':
                        sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                                  l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                        sweeps.append(sweep)
                        sd_rec = nusc.get('sample_data', sd_rec['prev'])
                    else:
                        break
                info['sweeps'] = sweeps

                # obtain annotation
                if not test:
                    annotations = [
                        nusc.get('sample_annotation', token)
                        for token in sample['anns']
                    ]
                    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
                    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
                    rots = np.array([b.orientation.yaw_pitch_roll[0]
                                     for b in boxes]).reshape(-1, 1)
                    velocity = np.array(
                        [nusc.box_velocity(token)[:2] for token in sample['anns']])
                    valid_flag = np.array(
                        [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                         for anno in annotations],
                        dtype=bool).reshape(-1)

                    # convert velo from global to lidar
                    for i in range(len(boxes)):
                        velo = np.array([*velocity[i], 0.0])
                        velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                            l2e_r_mat).T
                        velocity[i] = velo[:2]

                    names = [b.name for b in boxes]
                    # Map class names using the same mapping as MMDet3D
                    name_mapping = {
                        'human.pedestrian.adult': 'pedestrian',
                        'human.pedestrian.child': 'pedestrian',
                        'human.pedestrian.wheelchair': 'ignore',
                        'human.pedestrian.stroller': 'ignore',
                        'human.pedestrian.personal_mobility': 'ignore',
                        'human.pedestrian.police_officer': 'pedestrian',
                        'human.pedestrian.construction_worker': 'pedestrian',
                        'animal': 'ignore',
                        'vehicle.car': 'car',
                        'vehicle.motorcycle': 'motorcycle',
                        'vehicle.bicycle': 'bicycle',
                        'vehicle.bus.bendy': 'bus',
                        'vehicle.bus.rigid': 'bus',
                        'vehicle.truck': 'truck',
                        'vehicle.construction': 'construction_vehicle',
                        'vehicle.emergency.ambulance': 'ignore',
                        'vehicle.emergency.police': 'ignore',
                        'vehicle.trailer': 'trailer',
                        'movable_object.barrier': 'barrier',
                        'movable_object.trafficcone': 'traffic_cone',
                        'movable_object.pushable_pullable': 'ignore',
                        'movable_object.debris': 'ignore',
                        'static_object.bicycle_rack': 'ignore',
                    }

                    for i in range(len(names)):
                        if names[i] in name_mapping:
                            names[i] = name_mapping[names[i]]
                    names = np.array(names)

                    # we need to convert rot to SECOND format.
                    gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                    assert len(gt_boxes) == len(
                        annotations), f'{len(gt_boxes)}, {len(annotations)}'
                    info['gt_boxes'] = gt_boxes
                    info['gt_names'] = names
                    info['gt_velocity'] = velocity.reshape(-1, 2)
                    info['num_lidar_pts'] = np.array(
                        [a['num_lidar_pts'] for a in annotations])
                    info['num_radar_pts'] = np.array(
                        [a['num_radar_pts'] for a in annotations])
                    info['valid_flag'] = valid_flag

                # Add to appropriate list
                if sample['scene_token'] in train_scenes:
                    train_nusc_infos.append(info)
                    train_count += 1
                else:
                    val_nusc_infos.append(info)
                    val_count += 1

                # Update progress bar
                pbar.set_postfix({'train': train_count, 'val': val_count,
                                 'skip': skipped_count, 'err': errors_count})
                pbar.update(1)

            except Exception as e:
                logger.debug(f"Error processing sample {sample.get('token', 'unknown')}: {str(e)}")
                errors_count += 1
                pbar.set_postfix({'train': train_count, 'val': val_count,
                                 'skip': skipped_count, 'err': errors_count})
                pbar.update(1)
                continue

    # Log final statistics
    logger.info(f"\n   📊 Processing complete:")
    logger.info(f"      Train samples: {train_count}")
    logger.info(f"      Val samples: {val_count}")
    if skipped_count > 0:
        logger.warning(f"      Skipped samples: {skipped_count}")
    if errors_count > 0:
        logger.warning(f"      Error samples: {errors_count}")

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
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

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    logger.info("\n🔄 Exporting 2D annotations...")
    logger.info(f"   Info path: {info_path}")
    logger.info(f"   Mono3D: {mono3d}")

    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]

    with open(info_path, 'rb') as f:
        nusc_infos = pickle.load(f)['infos']

    logger.info(f"   Samples to process: {len(nusc_infos)}")

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)

    with tqdm(total=len(nusc_infos), desc="   Exporting 2D annotations", unit="sample") as pbar:
        for info in nusc_infos:
            for cam in camera_types:
                cam_info = info['cams'][cam]
                coco_infos = get_2d_boxes(
                    nusc,
                    cam_info['sample_data_token'],
                    visibilities=['', '1', '2', '3', '4'],
                    mono3d=mono3d)
                # Use cv2 instead of mmcv.imread
                img = cv2.imread(cam_info['data_path'])
                if img is None:
                    logger.warning(f"Could not read image: {cam_info['data_path']}")
                    continue

                (height, width, _) = img.shape
                coco_2d_dict['images'].append(
                    dict(
                        file_name=cam_info['data_path'].split('data/nuscenes/')
                        [-1],
                        id=cam_info['sample_data_token'],
                        token=info['token'],
                        cam2ego_rotation=cam_info['sensor2ego_rotation'],
                        cam2ego_translation=cam_info['sensor2ego_translation'],
                        ego2global_rotation=info['ego2global_rotation'],
                        ego2global_translation=info['ego2global_translation'],
                        cam_intrinsic=cam_info['cam_intrinsic'],
                        width=width,
                        height=height))
                for coco_info in coco_infos:
                    if coco_info is None:
                        continue
                    # add an empty key for coco format
                    coco_info['segmentation'] = []
                    coco_info['id'] = coco_ann_id
                    coco_2d_dict['annotations'].append(coco_info)
                    coco_ann_id += 1
            pbar.update(1)

    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'

    import json
    output_file = f'{json_prefix}.coco.json'
    logger.info(f"   Saving to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(coco_2d_dict, f)

    logger.info(f"   ✅ 2D annotations exported: {coco_ann_id} annotations")


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    # Use the class name mapping
    name_mapping = {
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.wheelchair': 'pedestrian',
        'human.pedestrian.stroller': 'pedestrian',
        'human.pedestrian.personal_mobility': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'animal': 'animal',
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.truck': 'truck',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.emergency.ambulance': 'car',
        'vehicle.emergency.police': 'car',
        'vehicle.trailer': 'trailer',
        'movable_object.barrier': 'barrier',
        'movable_object.trafficcone': 'traffic_cone',
        'movable_object.pushable_pullable': 'barrier',
        'movable_object.debris': 'debris',
        'static_object.bicycle_rack': 'bicycle_rack',
    }

    if repro_rec['category_name'] in name_mapping:
        cat_name = name_mapping[repro_rec['category_name']]
    else:
        cat_name = repro_rec['category_name']

    # Only include if it's in the valid categories
    if cat_name not in nus_categories:
        return None

    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec