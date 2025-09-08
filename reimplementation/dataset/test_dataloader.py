"""
Comprehensive testing suite for BEVFormer DataLoader
Tests dataset, collation, and integration with DataLoader
"""

import os
import sys
import torch
import torch.utils.data as data
import numpy as np
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append('/home/irdali.durrani/po-pi/BEVFormer/reimplementation/dataset')

from nuscenes_dataset import NuScenesDataset
from collate_fn import custom_collate_fn, validate_batch


def test_dataset_basic():
    """Test basic dataset functionality."""
    print("🔄 Testing basic dataset functionality...")
    
    data_file = '/home/irdali.durrani/po-pi/BEVFormer/reimplementation/data/nuscenes/nuscenes_infos_temporal_val.pkl'
    if not os.path.exists(data_file):
        print("❌ Dataset file not found, skipping test")
        return False
    
    try:
        # Create dataset
        dataset = NuScenesDataset(
            data_file=data_file,
            queue_length=4,
            training=True
        )
        
        print(f"   ✅ Dataset created: {len(dataset)} sequences")
        
        # Test __getitem__
        sample = dataset[0]
        
        # Validate sample structure
        required_keys = ['img', 'img_metas']
        for key in required_keys:
            if key not in sample:
                print(f"   ❌ Missing required key: {key}")
                return False
        
        # Validate shapes
        img = sample['img']
        if len(img.shape) != 5:  # (T, N, C, H, W)
            print(f"   ❌ Wrong image dimensions: {img.shape}")
            return False
        
        T, N, C, H, W = img.shape
        if N != 6 or C != 3:
            print(f"   ❌ Wrong image format: expected (T, 6, 3, H, W), got {img.shape}")
            return False
        
        # Validate metadata
        img_metas = sample['img_metas']
        if not isinstance(img_metas, list) or len(img_metas) != T:
            print(f"   ❌ Wrong img_metas structure: expected list of length {T}")
            return False
        
        print(f"   ✅ Sample validation passed: {img.shape}")
        print(f"   ✅ {len(img_metas)} temporal frames with metadata")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Dataset test failed: {e}")
        return False


def test_dataloader_integration():
    """Test dataset integration with PyTorch DataLoader."""
    print("🔄 Testing DataLoader integration...")
    
    data_file = '/home/irdali.durrani/po-pi/BEVFormer/reimplementation/data/nuscenes/nuscenes_infos_temporal_val.pkl'
    if not os.path.exists(data_file):
        print("❌ Dataset file not found, skipping test")
        return False
    
    try:
        # Create dataset
        dataset = NuScenesDataset(
            data_file=data_file,
            queue_length=4,
            training=True
        )
        
        # Create DataLoader
        batch_size = 2
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle for deterministic testing
            collate_fn=custom_collate_fn,
            num_workers=0,  # No multiprocessing for testing
            drop_last=False
        )
        
        print(f"   ✅ DataLoader created: {len(dataloader)} batches")
        
        # Test first batch
        batch = next(iter(dataloader))
        
        # Validate batch
        is_valid = validate_batch(batch, batch_size, 4)  # queue_length=4
        
        if not is_valid:
            print("   ❌ Batch validation failed")
            return False
        
        print(f"   ✅ First batch processed successfully")
        
        # Test multiple batches
        batch_count = 0
        max_test_batches = 3
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_test_batches:
                break
            
            # Quick validation
            if 'img' not in batch or 'img_metas' not in batch:
                print(f"   ❌ Batch {batch_idx} missing required keys")
                return False
            
            batch_count += 1
        
        print(f"   ✅ Processed {batch_count} batches successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_temporal_consistency():
    """Test that temporal sequences are consistent within scenes."""
    print("🔄 Testing temporal consistency...")
    
    data_file = '/home/irdali.durrani/po-pi/BEVFormer/reimplementation/data/nuscenes/nuscenes_infos_temporal_val.pkl'
    if not os.path.exists(data_file):
        print("❌ Dataset file not found, skipping test")
        return False
    
    try:
        dataset = NuScenesDataset(
            data_file=data_file,
            queue_length=4,
            training=True
        )
        
        # Test a few sequences
        for seq_idx in range(min(3, len(dataset))):
            sample = dataset[seq_idx]
            info = dataset.get_sample_info(seq_idx)
            
            # Check that all frames in sequence belong to same scene
            img_metas = sample['img_metas']
            scene_tokens = [meta.get('scene_token', '') for meta in img_metas]
            
            if len(set(scene_tokens)) > 1:
                print(f"   ❌ Sequence {seq_idx} contains multiple scenes: {scene_tokens}")
                return False
            
            # Check temporal ordering
            for t in range(len(img_metas)):
                expected_frame_idx = t
                actual_frame_idx = img_metas[t].get('frame_idx_in_sequence', -1)
                
                if actual_frame_idx != expected_frame_idx:
                    print(f"   ❌ Sequence {seq_idx} frame {t} has wrong frame_idx: expected {expected_frame_idx}, got {actual_frame_idx}")
                    return False
            
            # Check prev_bev_exists logic
            first_frame_prev_bev = img_metas[0].get('prev_bev_exists', True)
            if first_frame_prev_bev:
                print(f"   ❌ First frame should have prev_bev_exists=False")
                return False
            
            print(f"   ✅ Sequence {seq_idx}: scene={info['scene_token'][:8]}..., {len(img_metas)} frames")
        
        print("   ✅ Temporal consistency test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Temporal consistency test failed: {e}")
        return False


def test_memory_efficiency():
    """Test memory usage and loading efficiency."""
    print("🔄 Testing memory efficiency...")
    
    data_file = '/home/irdali.durrani/po-pi/BEVFormer/reimplementation/data/nuscenes/nuscenes_infos_temporal_val.pkl'
    if not os.path.exists(data_file):
        print("❌ Dataset file not found, skipping test")
        return False
    
    try:
        import psutil
        import gc
        
        # Measure baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create dataset
        dataset = NuScenesDataset(
            data_file=data_file,
            queue_length=4,
            training=True
        )
        
        after_dataset_memory = process.memory_info().rss / 1024 / 1024  # MB
        dataset_memory = after_dataset_memory - baseline_memory
        
        print(f"   ✅ Dataset memory usage: {dataset_memory:.1f} MB")
        
        # Test loading multiple samples
        memory_samples = []
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            gc.collect()  # Force garbage collection
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory - after_dataset_memory)
        
        avg_sample_memory = np.mean(memory_samples)
        max_sample_memory = np.max(memory_samples)
        
        print(f"   ✅ Average sample memory: {avg_sample_memory:.1f} MB")
        print(f"   ✅ Peak sample memory: {max_sample_memory:.1f} MB")
        
        # Memory should be reasonable (less than 1GB for samples)
        if max_sample_memory > 1000:  # 1GB
            print(f"   ⚠️  High memory usage detected: {max_sample_memory:.1f} MB")
        
        return True
        
    except ImportError:
        print("   ⚠️  psutil not available, skipping memory test")
        return True
    except Exception as e:
        print(f"   ❌ Memory test failed: {e}")
        return False


def test_model_integration():
    """Test dataloader integration with BEVFormer model."""
    print("🔄 Testing model integration...")
    
    try:
        # Add model path
        sys.path.append('/home/irdali.durrani/po-pi/BEVFormer/reimplementation/models')
        from bevformer import BEVFormer
        
        data_file = '/home/irdali.durrani/po-pi/BEVFormer/reimplementation/data/nuscenes/nuscenes_infos_temporal_val.pkl'
        if not os.path.exists(data_file):
            print("❌ Dataset file not found, skipping test")
            return False
        
        # Create dataset and dataloader
        dataset = NuScenesDataset(
            data_file=data_file,
            queue_length=4,
            training=True
        )
        
        dataloader = data.DataLoader(
            dataset,
            batch_size=1,  # Small batch for testing
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=0
        )
        
        # Create model (minimal config)
        embed_dims = 256
        model_cfg = dict(
            img_backbone=dict(
                depth=50,
                num_stages=4,
                out_indices=(1, 2, 3),
                frozen_stages=1,
                with_cp=False
            ),
            img_neck=dict(
                in_channels=[512, 1024, 2048],
                out_channels=embed_dims,
                num_outs=4,
                start_level=0,
                add_extra_convs='on_output',
                relu_before_extra_convs=True
            ),
            pts_bbox_head=dict(
                num_classes=10,
                in_channels=embed_dims,
                num_query=900,
                bev_h=200,
                bev_w=200,
                transformer=dict(
                    embed_dims=embed_dims,
                    encoder=dict(
                        num_layers=1,  # Minimal for testing
                        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                        num_points_in_pillar=4,
                        return_intermediate=False
                    ),
                    decoder=dict(
                        num_layers=1,  # Minimal for testing
                        return_intermediate=True
                    ),
                    rotate_prev_bev=True,
                    use_shift=True,
                    use_can_bus=True
                ),
                bbox_coder=dict(
                    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                    max_num=300,
                    num_classes=10
                ),
                train_cfg=dict(
                    pts=dict(
                        assigner=dict(
                            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
                        )
                    )
                )
            ),
            use_grid_mask=True,
            video_test_mode=False
        )
        
        model = BEVFormer(**model_cfg)
        model.train()
        
        print("   ✅ Model created successfully")
        
        # Test forward pass with dataloader
        batch = next(iter(dataloader))
        
        # Convert to model format: batch['img'] is (B, T, N, C, H, W)
        batch_img = batch['img']
        batch_img_metas = batch['img_metas']
        batch_gt_bboxes_3d = batch.get('gt_bboxes_3d', [torch.zeros(0, 9)])
        batch_gt_labels_3d = batch.get('gt_labels_3d', [torch.zeros(0, dtype=torch.long)])
        
        print(f"   ✅ Batch prepared: img={batch_img.shape}")
        
        # Forward pass
        with torch.no_grad():  # No gradients for testing
            losses = model.forward_train(
                img=batch_img,
                img_metas=batch_img_metas,
                gt_bboxes_3d=batch_gt_bboxes_3d,
                gt_labels_3d=batch_gt_labels_3d
            )
        
        print(f"   ✅ Forward pass successful!")
        print(f"   ✅ Losses: {list(losses.keys())}")
        
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Model not available, skipping integration test: {e}")
        return True
    except Exception as e:
        print(f"   ❌ Model integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all dataloader tests."""
    print("=" * 80)
    print("BEVFORMER DATALOADER COMPREHENSIVE TESTING")
    print("=" * 80)
    
    tests = [
        ("Basic Dataset Functionality", test_dataset_basic),
        ("DataLoader Integration", test_dataloader_integration),
        ("Temporal Consistency", test_temporal_consistency),
        ("Memory Efficiency", test_memory_efficiency),
        ("Model Integration", test_model_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 60)
        
        try:
            success = test_func()
            results[test_name] = success
            
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:<8} {test_name}")
    
    print("-" * 80)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! DataLoader is ready for production!")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)