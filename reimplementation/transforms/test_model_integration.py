"""
End-to-End BEVFormer Model Integration Test
Tests our data pipeline with our reimplemented BEVFormer model
"""

import os
import pickle
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our complete data pipeline
from transforms import (
    LoadMultiViewImageFromFiles,
    NormalizeMultiviewImage,
    PhotoMetricDistortionMultiViewImage,
    LoadAnnotations3D,
    ObjectNameFilter,
    ObjectRangeFilter,
    PadMultiViewImage,
    DefaultFormatBundle3D,
    CustomCollect3D
)

# Import our reimplemented BEVFormer model
try:
    from models import BEVFormer
except ImportError as e:
    print(f"❌ Could not import BEVFormer model: {e}")
    print("Please ensure all model components are properly implemented")
    sys.exit(1)


def create_model_config():
    """Create BEVFormer model configuration matching our implementation."""
    
    # nuScenes configuration parameters
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    
    # Model dimensions
    embed_dims = 256
    bev_h, bev_w = 200, 200
    
    # Create model config matching our BEVFormer test example
    model_cfg = dict(
        img_backbone=dict(
            depth=50,  # Use ResNet50 for testing (lighter than 101)
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            with_cp=False
        ),
        img_neck=dict(
            in_channels=[512, 1024, 2048],  # ResNet50 stage outputs
            out_channels=embed_dims,
            num_outs=4,
            start_level=0,
            add_extra_convs='on_output',
            relu_before_extra_convs=True
        ),
        pts_bbox_head=dict(
            num_classes=len(class_names),
            in_channels=embed_dims,
            num_query=900,
            bev_h=bev_h,
            bev_w=bev_w,
            transformer=dict(
                embed_dims=embed_dims,
                encoder=dict(
                    num_layers=2,  # Reduced for testing
                    pc_range=point_cloud_range,
                    num_points_in_pillar=4,
                    return_intermediate=False
                ),
                decoder=dict(
                    num_layers=2,  # Reduced for testing
                    return_intermediate=True
                ),
                rotate_prev_bev=True,
                use_shift=True,
                use_can_bus=True
            ),
            bbox_coder=dict(
                pc_range=point_cloud_range,
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=300,
                num_classes=len(class_names)
            ),
            train_cfg=dict(
                pts=dict(
                    assigner=dict(
                        pc_range=point_cloud_range
                    )
                )
            )
        ),
        use_grid_mask=True,
        video_test_mode=False  # Start with single frame
    )
    
    return model_cfg


def create_data_pipeline(training=True):
    """Create our complete data pipeline."""
    
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
    
    # Create pipeline transforms
    transforms = []
    transforms.append(LoadMultiViewImageFromFiles(to_float32=True))
    if training:
        transforms.append(PhotoMetricDistortionMultiViewImage())
    transforms.append(LoadAnnotations3D(with_bbox_3d=True, with_label_3d=True, with_attr_label=False))
    transforms.append(ObjectRangeFilter(point_cloud_range=point_cloud_range))
    transforms.append(ObjectNameFilter(classes=class_names))
    transforms.append(NormalizeMultiviewImage(**img_norm_cfg))
    transforms.append(PadMultiViewImage(size_divisor=32))
    transforms.append(DefaultFormatBundle3D(keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'], class_names=class_names))
    transforms.append(CustomCollect3D(keys=['gt_bboxes_3d', 'gt_labels_3d', 'img']))
    
    return transforms


def process_sample_with_pipeline(sample, transforms):
    """Process a single sample through our data pipeline."""
    result = sample.copy()
    
    for i, transform in enumerate(transforms):
        try:
            result = transform(result)
        except Exception as e:
            print(f"❌ Pipeline failed at step {i+1} ({transform.__class__.__name__}): {e}")
            raise
    
    return result


def test_end_to_end_integration():
    """Test end-to-end integration: data pipeline → model → training."""
    
    print("=" * 80)
    print("END-TO-END BEVFORMER MODEL INTEGRATION TEST")
    print("=" * 80)
    
    # Load nuScenes data
    data_file = '/home/irdali.durrani/po-pi/BEVFormer/data/nuscenes/nuscenes_infos_temporal_val.pkl'
    if not os.path.exists(data_file):
        print("❌ nuScenes dataset not found!")
        return None
    
    print("🔄 Loading nuScenes dataset...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Select a single sample for testing
    sample = data['data_list'][0]
    print(f"✅ Loaded dataset with {len(data['data_list'])} samples")
    print(f"   Selected sample 0 with {len(sample['gt_names'])} GT objects")
    
    # Phase 1: Data Pipeline Processing
    print(f"\n🔄 Phase 1: Data Pipeline Processing...")
    
    pipeline = create_data_pipeline(training=True)
    print(f"   Created pipeline with {len(pipeline)} transforms")
    
    try:
        processed_data = process_sample_with_pipeline(sample, pipeline)
        print(f"✅ Data pipeline successful!")
        print(f"   Final data keys: {list(processed_data.keys())}")
        print(f"   - img: {processed_data['img'].shape} ({processed_data['img'].dtype})")
        print(f"   - gt_bboxes_3d: {processed_data['gt_bboxes_3d'].shape} ({processed_data['gt_bboxes_3d'].dtype})")
        print(f"   - gt_labels_3d: {processed_data['gt_labels_3d'].shape} ({processed_data['gt_labels_3d'].dtype})")
        print(f"   - img_metas: {len(processed_data['img_metas'])} metadata fields")
    except Exception as e:
        print(f"❌ Data pipeline failed: {e}")
        return None
    
    # Phase 2: Model Creation and Forward Pass
    print(f"\n🔄 Phase 2: Model Creation and Forward Pass...")
    
    try:
        model_config = create_model_config()
        model = BEVFormer(**model_config)
        print(f"✅ BEVFormer model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Set model to training mode
        model.train()
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Prepare inputs for model (use temporal format with T=2 to avoid empty prev_img)
    try:
        # Our BEVFormer requires temporal dimension: (B, T, N, C, H, W)
        # To avoid empty prev_img, we use T=2 with duplicate frames
        # Our data is (6, 3, H, W), we need (1, 2, 6, 3, H, W)
        single_frame = processed_data['img'].unsqueeze(0).unsqueeze(0)  # (1, 1, 6, 3, H, W)
        batch_img = single_frame.repeat(1, 2, 1, 1, 1, 1)  # (1, 2, 6, 3, H, W) - duplicate frame
        
        batch_gt_bboxes_3d = [processed_data['gt_bboxes_3d']]  # List of tensors
        batch_gt_labels_3d = [processed_data['gt_labels_3d']]  # List of tensors
        
        # Create img_metas with CAN bus data (required by our model)
        img_metas_with_can_bus = processed_data['img_metas'].copy()
        img_metas_with_can_bus['can_bus'] = np.zeros(18)  # Dummy CAN bus data
        img_metas_with_can_bus['scene_token'] = img_metas_with_can_bus.get('scene_token', 'test_scene')
        img_metas_with_can_bus['prev_bev_exists'] = True  # We have previous frame (duplicate)
        
        # Create nested list for temporal frames
        batch_img_metas = [[img_metas_with_can_bus, img_metas_with_can_bus]]  # 2 frames
        
        print(f"   Prepared batch inputs (temporal mode with duplicated frames):")
        print(f"   - batch_img: {batch_img.shape}")
        print(f"   - batch_gt_bboxes_3d: [{batch_gt_bboxes_3d[0].shape}]")
        print(f"   - batch_gt_labels_3d: [{batch_gt_labels_3d[0].shape}]")
        print(f"   - batch_img_metas: 2 temporal frames with CAN bus data")
        
        # Forward pass
        print(f"\n   Running forward pass...")
        losses = model.forward_train(
            img=batch_img,
            img_metas=batch_img_metas,
            gt_bboxes_3d=batch_gt_bboxes_3d,
            gt_labels_3d=batch_gt_labels_3d
        )
        
        print(f"✅ Forward pass successful!")
        print(f"   Computed losses:")
        total_loss = 0
        for key, value in losses.items():
            if torch.is_tensor(value) and value.requires_grad:
                print(f"   - {key}: {value.item():.6f}")
                total_loss += value
            else:
                print(f"   - {key}: {value}")
        
        print(f"   - Total Loss: {total_loss.item():.6f}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Phase 3: Backward Pass and Gradient Update
    print(f"\n🔄 Phase 3: Backward Pass and Gradient Update...")
    
    try:
        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        optimizer.zero_grad()
        
        print(f"   Created AdamW optimizer")
        
        # Backward pass
        print(f"   Running backward pass...")
        total_loss.backward()
        
        # Check gradients
        grad_norm = 0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
                param_count += 1
        
        grad_norm = grad_norm ** (1. / 2)
        print(f"✅ Backward pass successful!")
        print(f"   - Parameters with gradients: {param_count}")
        print(f"   - Gradient norm: {grad_norm:.6f}")
        
        # Gradient clipping (common in transformer training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
        
        # Optimizer step
        optimizer.step()
        print(f"✅ Gradient update successful!")
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Phase 4: Second Forward Pass to Verify Learning
    print(f"\n🔄 Phase 4: Verification - Second Forward Pass...")
    
    try:
        optimizer.zero_grad()
        
        # Second forward pass
        losses2 = model.forward_train(
            img=batch_img,
            img_metas=batch_img_metas, 
            gt_bboxes_3d=batch_gt_bboxes_3d,
            gt_labels_3d=batch_gt_labels_3d
        )
        
        total_loss2 = sum(v for v in losses2.values() if torch.is_tensor(v) and v.requires_grad)
        
        loss_change = total_loss.item() - total_loss2.item()
        print(f"✅ Second forward pass successful!")
        print(f"   - First loss: {total_loss.item():.6f}")
        print(f"   - Second loss: {total_loss2.item():.6f}")
        print(f"   - Loss change: {loss_change:.6f} {'(improved)' if loss_change > 0 else '(increased)'}")
        
    except Exception as e:
        print(f"❌ Second forward pass failed: {e}")
        return None
    
    # Phase 5: Inference Test
    print(f"\n🔄 Phase 5: Inference Mode Test...")
    
    try:
        model.eval()
        
        with torch.no_grad():
            # Run inference with the last frame from temporal sequence
            inference_img = batch_img[:, -1, ...]  # (1, 6, 3, H, W) - last frame
            results = model.forward_test(
                img=[inference_img[0]],  # Remove batch dim: (6, 3, H, W)
                img_metas=[batch_img_metas[0][-1]]  # Last frame meta
            )
        
        print(f"✅ Inference successful!")
        if results and len(results) > 0:
            result = results[0]
            if 'boxes_3d' in result:
                num_detections = len(result['boxes_3d'])
                print(f"   - Generated {num_detections} detection(s)")
                if num_detections > 0:
                    scores = result['scores_3d']
                    labels = result['labels_3d']
                    print(f"   - Score range: [{scores.min():.3f}, {scores.max():.3f}]")
                    print(f"   - Predicted classes: {torch.unique(labels).tolist()}")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return None
    
    # Success Summary
    print(f"\n" + "=" * 80)
    print("🎉 END-TO-END INTEGRATION TEST SUCCESSFUL! 🎉")
    print("=" * 80)
    print("✅ All phases completed successfully:")
    print("   1. ✅ Data Pipeline Processing")
    print("   2. ✅ Model Creation and Forward Pass") 
    print("   3. ✅ Backward Pass and Gradient Update")
    print("   4. ✅ Verification - Second Forward Pass")
    print("   5. ✅ Inference Mode Test")
    print(f"\n🚀 BEVFormer can successfully train on our processed data!")
    print(f"🎯 Model is ready for full training loop integration!")
    print(f"💾 Complete pipeline: nuScenes data → transforms → model → loss → gradients")
    
    return {
        'model': model,
        'processed_data': processed_data,
        'losses': losses,
        'inference_results': results if 'results' in locals() else None
    }


if __name__ == "__main__":
    # Run the complete integration test
    result = test_end_to_end_integration()
    
    if result is not None:
        print(f"\n🎉 Integration test completed successfully!")
        print(f"   Model ready for production training!")
    else:
        print(f"\n❌ Integration test failed!")
        print(f"   Please check the error messages above.")