#!/usr/bin/env python3
"""
Data preparation script for BEVFormer reimplementation
Generates temporal pickle files without mmcv/mmdet3d dependencies

Usage:
    python reimplementation/tools/create_data.py nuscenes \
        --root-path ./data/nuscenes \
        --out-dir ./data/nuscenes \
        --extra-tag nuscenes \
        --canbus ./data
"""

import argparse
import sys
from os import path as osp

# Add parent directory to path to import nuscenes_converter
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import nuscenes_converter


def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Args:
        root_path (str): Path of dataset root.
        can_bus_root_path (str): Path to CAN bus data.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the info files.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix,
        version=version, max_sweeps=max_sweeps)


def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument('dataset', metavar='dataset', help='name of the dataset')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data/nuscenes',
        help='specify the root path of dataset')
    parser.add_argument(
        '--canbus',
        type=str,
        default='./data',
        help='specify the root path of nuScenes canbus')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0',
        required=False,
        help='specify the dataset version, no need for kitti')
    parser.add_argument(
        '--max-sweeps',
        type=int,
        default=10,
        required=False,
        help='specify sweeps of lidar per example')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/nuscenes',
        required=False,
        help='name of info pkl')
    parser.add_argument('--extra-tag', type=str, default='nuscenes')
    parser.add_argument(
        '--workers', type=int, default=4, help='number of threads to be used')
    parser.add_argument(
        '--include-test',
        action='store_true',
        help='Include test set processing (requires v1.0-test data)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset != 'nuscenes':
        print(f"Error: Only 'nuscenes' dataset is supported in this reimplementation")
        print(f"Got: {args.dataset}")
        sys.exit(1)

    print("=" * 80)
    print("BEVFormer Data Preparation (Reimplementation)")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Root path: {args.root_path}")
    print(f"CAN bus path: {args.canbus}")
    print(f"Version: {args.version}")
    print(f"Output dir: {args.out_dir}")
    print("=" * 80)

    if args.version == 'v1.0-mini':
        # For mini dataset, no train/val split in version name
        train_version = args.version
        print(f"\nProcessing {train_version}...")
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    else:
        # For full dataset, process train-val
        train_version = f'{args.version}-trainval'
        print(f"\nProcessing {train_version}...")
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)

        # Only process test if explicitly requested
        if args.include_test:
            test_version = f'{args.version}-test'
            print(f"\nProcessing {test_version}...")
            try:
                nuscenes_data_prep(
                    root_path=args.root_path,
                    can_bus_root_path=args.canbus,
                    info_prefix=args.extra_tag,
                    version=test_version,
                    dataset_name='NuScenesDataset',
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps)
            except AssertionError as e:
                print(f"⚠️  Warning: Test dataset not found. Skipping test set processing.")
                print(f"   If you need test set, download v1.0-test data and use --include-test flag")
        else:
            print("\n📝 Note: Skipping test set processing (use --include-test flag if needed)")

    print("\n✅ Data preparation completed!")


if __name__ == '__main__':
    main()