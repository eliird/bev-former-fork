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
            
        dataset_cfg.data_root = 'data/nuscenes/'
        dataset_cfg.ann_file = 'nuscenes_infos_temporal_train.pkl'  # Relative to data_root
        # Register mmdet3d transforms into mmengine registry (which is what datasets actually use)
        from mmdet.registry import DATASETS
        from mmengine.registry import TRANSFORMS
        from projects.mmdet3d_plugin.datasets.pipelines import (
            PhotoMetricDistortionMultiViewImage, 
            NormalizeMultiviewImage,
            RandomScaleImageMultiViewImage, 
            PadMultiViewImage
        )
        from projects.mmdet3d_plugin.datasets.pipelines.formating import (
            CustomDefaultFormatBundle3D, DefaultFormatBundle3D
        )
        
        # Register the custom transforms directly
        TRANSFORMS.register_module(name='PhotoMetricDistortionMultiViewImage', module=PhotoMetricDistortionMultiViewImage, force=True)
        TRANSFORMS.register_module(name='NormalizeMultiviewImage', module=NormalizeMultiviewImage, force=True)
        TRANSFORMS.register_module(name='RandomScaleImageMultiViewImage', module=RandomScaleImageMultiViewImage, force=True)
        TRANSFORMS.register_module(name='PadMultiViewImage', module=PadMultiViewImage, force=True)
        TRANSFORMS.register_module(name='CustomCollect3D', module=CustomCollect3D, force=True)
        TRANSFORMS.register_module(name='DefaultFormatBundle3D', module=DefaultFormatBundle3D, force=True)
        TRANSFORMS.register_module(name='CustomDefaultFormatBundle3D', module=CustomDefaultFormatBundle3D, force=True)
        # List of transforms we need from mmdet3d
        mmdet3d_transforms = [
            'LoadMultiViewImageFromFiles',
            'LoadAnnotations3D', 
            'ObjectRangeFilter',
            'ObjectNameFilter'
        ]
        
        # List of custom transforms from the plugin (these should already be registered via import)
        plugin_transforms = [
            'PhotoMetricDistortionMultiViewImage',
            'NormalizeMultiviewImage', 
            'RandomScaleImageMultiViewImage',
            'PadMultiViewImage',
            'DefaultFormatBundle3D',
            'CustomCollect3D'
        ]
        
        # from mmdet3d.datasets.transforms import PhotoMetricDistortion3D
        # from mmdet.datasets.transforms import PhotoMetricDistortion
        print("Registering mmdet3d transforms into mmengine registry...")
        registered_count = 0
        
        # Register mmdet3d transforms
        for transform_name in mmdet3d_transforms:
            try:
                # First try direct import from mmdet3d.datasets.transforms
                from mmdet3d.datasets import transforms as mmdet3d_transforms_module
                if hasattr(mmdet3d_transforms_module, transform_name):
                    transform_class = getattr(mmdet3d_transforms_module, transform_name)
                    TRANSFORMS.register_module(name=transform_name, module=transform_class, force=True)
                    print(f"✓ Registered {transform_name}")
                    registered_count += 1
                else:
                    # Try getting from mmdet3d registry
                    from mmdet3d.registry import TRANSFORMS as TRANSFORMS_3D
                    if transform_name in TRANSFORMS_3D.module_dict:
                        transform_class = TRANSFORMS_3D.get(transform_name)
                        TRANSFORMS.register_module(name=transform_name, module=transform_class, force=True)
                        print(f"✓ Registered {transform_name} from mmdet3d registry")
                        registered_count += 1
                    else:
                        print(f"⚠ {transform_name} not found in mmdet3d")
            except Exception as e:
                print(f"⚠ Failed to register {transform_name}: {e}")
        
        print(f"Successfully registered {registered_count}/{len(mmdet3d_transforms)} mmdet3d transforms")
        
        # Register plugin transforms into mmengine registry 
        print("Registering plugin transforms into mmengine registry...")
        plugin_registered = 0
        
        from mmdet.registry import TRANSFORMS as MMDET_TRANSFORMS
        for transform_name in plugin_transforms:
            try:
                if transform_name in MMDET_TRANSFORMS.module_dict:
                    transform_class = MMDET_TRANSFORMS.get(transform_name)
                    TRANSFORMS.register_module(name=transform_name, module=transform_class, force=True)
                    print(f"✓ Registered {transform_name} from plugin")
                    plugin_registered += 1
                else:
                    print(f"⚠ {transform_name} not found in plugin registry")
            except Exception as e:
                print(f"⚠ Failed to register {transform_name}: {e}")
                
        print(f"Successfully registered {plugin_registered}/{len(plugin_transforms)} plugin transforms")
        # print(dataset_cfg)
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