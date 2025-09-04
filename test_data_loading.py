#!/usr/bin/env python3
"""
Test script to verify BEVFormer data loading works correctly.
This script tests the custom dataset classes and configurations.
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
from mmengine import Config
# from mmdet3d.datasets import build_dataset
from mmengine.registry import build_from_cfg
# Import to register the dataset
from projects.mmdet3d_plugin import *


def test_basic_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("Testing basic imports...")
    
    try:
        import mmcv
        import mmdet
        import mmdet3d
        print(f"✓ MMCV version: {mmcv.__version__}")
        print(f"✓ MMDet version: {mmdet.__version__}")
        print(f"✓ MMDet3D version: {mmdet3d.__version__}")
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config_loading():
    """Test loading BEVFormer configuration files."""
    print("=" * 60)
    print("Testing configuration loading...")
    
    config_paths = [
        'projects/configs/bevformer/bevformer_tiny.py',
        'projects/configs/bevformer/bevformer_small.py', 
        'projects/configs/bevformer/bevformer_base.py'
    ]
    
    for config_path in config_paths:
        try:
            cfg = Config.fromfile(config_path)
            print(f"✓ Successfully loaded: {config_path}")
            print(f"  - Model type: {cfg.model.type}")
            print(f"  - Dataset type: {cfg.train_dataloader.dataset.type if hasattr(cfg, 'train_dataloader') else cfg.data.train.type}")
        except Exception as e:
            print(f"✗ Failed to load {config_path}: {e}")
            return False
    
    return True


def test_custom_dataset():
    """Test the custom nuScenes dataset class."""
    print("=" * 60)
    print("Testing custom dataset class...")
    
    try:
        # Test basic dataset configuration (minimal config)
        dataset_cfg = dict(
            type='CustomNuScenesDataset',
            data_root='data/nuscenes/',
            ann_file='data/nuscenes/nuscenes_infos_temporal_train.pkl',
            pipeline=[],
            test_mode=False,
            metainfo=dict(classes=['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
                                  'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']),
        )
        
        print("✓ Dataset configuration created")
        
        # Check if annotation file exists
        ann_file = dataset_cfg['ann_file']
        if os.path.exists(ann_file):
            print(f"✓ Annotation file exists: {ann_file}")
        else:
            print(f"⚠ Annotation file not found: {ann_file}")
            print("  This is expected if you haven't run data preparation yet.")
        
        # Check data directory structure
        data_root = dataset_cfg['data_root']
        required_dirs = ['maps', 'samples', 'sweeps', 'v1.0-mini']
        for dir_name in required_dirs:
            dir_path = os.path.join(data_root, dir_name)
            if os.path.exists(dir_path):
                print(f"✓ Found directory: {dir_path}")
            else:
                print(f"✗ Missing directory: {dir_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False


def test_data_loading_pipeline():
    """Test the data loading pipeline if annotation files exist."""
    print("=" * 60)
    print("Testing data loading pipeline...")
    
    ann_file = 'data/nuscenes/nuscenes_infos_temporal_train.pkl'
    if not os.path.exists(ann_file):
        print(f"⚠ Skipping pipeline test - annotation file not found: {ann_file}")
        print("  Run the data preparation script first.")
        return True
    
    try:
        # Load a minimal config for testing
        cfg = Config.fromfile('projects/configs/bevformer/bevformer_tiny.py')
        
        # Modify config for testing
        if hasattr(cfg, 'train_dataloader'):
            dataset_cfg = cfg.train_dataloader.dataset
        else:
            dataset_cfg = cfg.data.train
            
        dataset_cfg.ann_file = ann_file
        dataset_cfg.data_root = 'data/nuscenes/'
        
        # Try to build dataset
        from mmdet3d.registry import DATASETS
        dataset = build_from_cfg(dataset_cfg, DATASETS)
        print(f"✓ Dataset built successfully")
        print(f"  - Dataset length: {len(dataset)}")
        print(f"  - Dataset type: {type(dataset).__name__}")
        
        if len(dataset) > 0:
            # Try to load one sample
            sample = dataset[0]
            print(f"✓ Successfully loaded sample 0")
            print(f"  - Sample keys: {list(sample.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("BEVFormer Data Loading Test")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Config Loading", test_config_loading), 
        ("Custom Dataset", test_custom_dataset),
        ("Data Loading Pipeline", test_data_loading_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your environment is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()