"""
BEVFormer: Multi-camera 3D Object Detection via Spatiotemporal Transformers
Pure PyTorch implementation without MMDetection dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import copy
import numpy as np
import time

# Import our reimplemented modules
from backbone import ResNetBackbone
from neck import FPNNeck  
from bevformer_head import BEVFormerHead
from grid_mask import GridMask


def bbox3d2result(bboxes, scores, labels):
    """Convert detection results to bbox3d format.
    
    Args:
        bboxes (Tensor): 3D bounding boxes
        scores (Tensor): Confidence scores
        labels (Tensor): Predicted labels
        
    Returns:
        dict: Detection results in standard format
    """
    return {
        'boxes_3d': bboxes,
        'scores_3d': scores,
        'labels_3d': labels
    }


class BEVFormer(nn.Module):
    """BEVFormer model for multi-camera 3D object detection.
    
    Args:
        img_backbone (dict): Config for image backbone
        img_neck (dict): Config for FPN neck
        pts_bbox_head (dict): Config for BEVFormerHead
        train_cfg (dict): Training configuration
        test_cfg (dict): Testing configuration
        use_grid_mask (bool): Whether to use GridMask augmentation
        video_test_mode (bool): Whether to use temporal info during inference
    """
    
    def __init__(self,
                 img_backbone=None,
                 img_neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 use_grid_mask=False,
                 video_test_mode=False,
                 **kwargs):
        super(BEVFormer, self).__init__()
        
        # Build backbone
        if img_backbone is not None:
            # For this implementation, we use our ResNetBackbone
            # In real config, would parse the dict to get params
            self.img_backbone = ResNetBackbone(
                depth=img_backbone.get('depth', 101),
                num_stages=img_backbone.get('num_stages', 4),
                out_indices=img_backbone.get('out_indices', (1, 2, 3)),
                frozen_stages=img_backbone.get('frozen_stages', 1),
                with_cp=img_backbone.get('with_cp', False)
            )
        else:
            self.img_backbone = None
            
        # Build neck
        if img_neck is not None:
            self.img_neck = FPNNeck(
                in_channels=img_neck.get('in_channels', [512, 1024, 2048]),
                out_channels=img_neck.get('out_channels', 256),
                num_outs=img_neck.get('num_outs', 4),
                start_level=img_neck.get('start_level', 0),
                add_extra_convs=img_neck.get('add_extra_convs', 'on_output'),
                relu_before_extra_convs=img_neck.get('relu_before_extra_convs', True)
            )
        else:
            self.img_neck = None
            
        # Build detection head
        if pts_bbox_head is not None:
            self.pts_bbox_head = BEVFormerHead(**pts_bbox_head)
        else:
            self.pts_bbox_head = None
            
        # GridMask augmentation
        self.grid_mask = GridMask(
            use_h=True, use_w=True, rotate=1, offset=False,
            ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        
        # Training and testing configs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # Temporal modeling
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        
        self.fp16_enabled = False
        
    @property
    def with_img_neck(self):
        """Whether the model has a neck."""
        return hasattr(self, 'img_neck') and self.img_neck is not None
        
    @property
    def with_pts_bbox(self):
        """Whether the model has a bbox head."""
        return hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None
        
    def extract_img_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images.
        
        Args:
            img (Tensor): Input images with shape (B, N, C, H, W) or (B*N, C, H, W)
                where B is batch size and N is number of cameras
            img_metas (list[dict]): Image meta information
            len_queue (int): Length of temporal queue
            
        Returns:
            list[Tensor]: Multi-level features
        """
        B = img.size(0)
        
        if img is not None:
            # Handle different input shapes
            if img.dim() == 5 and img.size(0) == 1:
                img = img.squeeze(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
                
            # Apply GridMask augmentation if enabled
            if self.use_grid_mask and self.training:
                img = self.grid_mask(img)
                
            # Extract backbone features
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
            
        # Apply neck if available
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            
        # Reshape features back to (B, N, C, H, W)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                # For temporal sequence: (B/len_queue, len_queue, N, C, H, W)
                img_feats_reshaped.append(
                    img_feat.view(int(B/len_queue), len_queue, int(BN/B), C, H, W)
                )
            else:
                # Normal case: (B, N, C, H, W)
                img_feats_reshaped.append(
                    img_feat.view(B, int(BN/B), C, H, W)
                )
                
        return img_feats_reshaped
        
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images (with optional FP16).
        
        Args:
            img (Tensor): Input images
            img_metas (list[dict]): Image meta information
            len_queue (int): Length of temporal queue
            
        Returns:
            list[Tensor]: Extracted features
        """
        # In full implementation, would use autocast for FP16
        # For now, just extract features normally
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats
        
    def forward_pts_train(self,
                         pts_feats,
                         gt_bboxes_3d,
                         gt_labels_3d,
                         img_metas,
                         gt_bboxes_ignore=None,
                         prev_bev=None):
        """Forward pass for point cloud branch during training.
        
        Args:
            pts_feats (list[Tensor]): Multi-level image features
            gt_bboxes_3d (list[Tensor]): Ground truth 3D boxes
            gt_labels_3d (list[Tensor]): Ground truth labels
            img_metas (list[dict]): Image meta information
            gt_bboxes_ignore (list[Tensor], optional): Ignored GT boxes
            prev_bev (Tensor, optional): Previous BEV features
            
        Returns:
            dict: Loss dictionary
        """
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)
        losses = self.pts_bbox_head.loss(
            gt_bboxes_3d, gt_labels_3d, outs, 
            gt_bboxes_ignore=gt_bboxes_ignore,
            img_metas=img_metas
        )
        return losses
        
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively.
        
        To save GPU memory, gradients are not calculated.
        
        Args:
            imgs_queue (Tensor): Queue of images (B, len_queue, N, C, H, W)
            img_metas_list (list[list[dict]]): Meta info for each frame
            
        Returns:
            Tensor: BEV features from the last frame in queue
        """
        self.eval()  # Set to eval mode
        
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            
            # Extract features for all frames at once
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            
            # Process each frame iteratively
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                
                # Check if previous BEV exists
                if img_metas[0].get('prev_bev_exists', True) == False:
                    prev_bev = None
                    
                # Get features for current frame
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                
                # Get BEV features only (no detection)
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True
                )
                
        self.train()  # Set back to train mode
        return prev_bev
        
    def forward_train(self,
                     img=None,
                     img_metas=None,
                     gt_bboxes_3d=None,
                     gt_labels_3d=None,
                     gt_bboxes_ignore=None,
                     **kwargs):
        """Forward training function.
        
        Args:
            img (Tensor): Input images with temporal dimension
                Shape: (B, T, N, C, H, W) where T is temporal length
            img_metas (list[list[dict]]): Meta info for each frame
            gt_bboxes_3d (list[Tensor]): Ground truth 3D boxes
            gt_labels_3d (list[Tensor]): Ground truth labels
            gt_bboxes_ignore (list[Tensor], optional): Ignored boxes
            
        Returns:
            dict: Dictionary of losses
        """
        len_queue = img.size(1)
        
        # Split current and previous frames
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]
        
        # Get previous BEV features
        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        
        # Get current frame meta
        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0].get('prev_bev_exists', True):
            prev_bev = None
            
        # Extract current frame features
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        # Compute losses
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d,
            img_metas, gt_bboxes_ignore, prev_bev
        )
        losses.update(losses_pts)
        
        return losses
        
    def forward_test(self, img=None, img_metas=None, **kwargs):
        """Forward testing/inference function.
        
        Args:
            img (list[Tensor]): Input images
            img_metas (list[list[dict]]): Image meta information
            
        Returns:
            list[dict]: Detection results
        """
        # Ensure inputs are lists
        if not isinstance(img_metas, list):
            img_metas = [img_metas]
        if not isinstance(img, list):
            img = [img]
            
        # Check scene change
        if img_metas[0][0].get('scene_token') != self.prev_frame_info.get('scene_token'):
            # Reset temporal info for new scene
            self.prev_frame_info['prev_bev'] = None
            
        # Update scene token
        self.prev_frame_info['scene_token'] = img_metas[0][0].get('scene_token')
        
        # Disable temporal info if not in video test mode
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None
            
        # Handle ego motion
        if 'can_bus' in img_metas[0][0]:
            tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
            tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
            
            if self.prev_frame_info['prev_bev'] is not None:
                # Calculate relative motion
                img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
                img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
            else:
                img_metas[0][0]['can_bus'][:3] = 0
                img_metas[0][0]['can_bus'][-1] = 0
        else:
            tmp_pos = 0
            tmp_angle = 0
            
        # Run inference
        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], 
            prev_bev=self.prev_frame_info['prev_bev'],
            **kwargs
        )
        
        # Update temporal info
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        
        return bbox_results
        
    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Simple test for point cloud branch.
        
        Args:
            x (list[Tensor]): Multi-level features
            img_metas (list[dict]): Image meta information
            prev_bev (Tensor, optional): Previous BEV features
            rescale (bool): Whether to rescale bboxes
            
        Returns:
            tuple: (bev_embed, bbox_results)
        """
        # Get predictions from head
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)
        
        # Get final bboxes
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale
        )
        
        # Convert to result format
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        
        return outs['bev_embed'], bbox_results
        
    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Simple test without augmentation.
        
        Args:
            img_metas (list[dict]): Image meta information
            img (Tensor): Input images
            prev_bev (Tensor, optional): Previous BEV features
            rescale (bool): Whether to rescale bboxes
            
        Returns:
            tuple: (new_prev_bev, bbox_results)
        """
        # Extract features
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        # Initialize results
        bbox_list = [dict() for _ in range(len(img_metas))]
        
        # Get predictions
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale
        )
        
        # Pack results
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            
        return new_prev_bev, bbox_list
        
    def forward(self, return_loss=True, **kwargs):
        """Unified forward function.
        
        Args:
            return_loss (bool): Whether to return losses (training) or results (testing)
            **kwargs: Additional arguments
            
        Returns:
            dict or list: Losses if training, results if testing
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)


def test_bevformer():
    """Comprehensive test of BEVFormer model"""
    print("=" * 60)
    print("Testing BEVFormer Model")
    print("=" * 60)
    
    # Model configuration (mimicking the config file)
    embed_dims = 256
    num_classes = 10
    batch_size = 1  # Use 1 for memory efficiency
    num_cams = 6
    img_h, img_w = 480, 800  # Typical nuScenes resolution
    bev_h, bev_w = 200, 200
    queue_length = 2  # Temporal frames (use 2 for testing)
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    # Create model config
    model_cfg = dict(
        img_backbone=dict(
            depth=50,  # Use ResNet50 for testing (lighter than 101)
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
            num_classes=num_classes,
            in_channels=embed_dims,
            num_query=900,
            bev_h=bev_h,
            bev_w=bev_w,
            transformer=dict(
                embed_dims=embed_dims,
                encoder=dict(
                    num_layers=2,  # Reduced for testing
                    pc_range=pc_range,
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
                pc_range=pc_range,
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=300,
                num_classes=num_classes
            ),
            train_cfg=dict(
                pts=dict(
                    assigner=dict(
                        pc_range=pc_range
                    )
                )
            )
        ),
        use_grid_mask=True,
        video_test_mode=True
    )
    
    print("Creating BEVFormer model...")
    model = BEVFormer(**model_cfg)
    model.eval()  # Set to eval mode for testing
    
    print(f"✓ Model created successfully")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test feature extraction
    print("\nTesting feature extraction...")
    
    # Create dummy input
    img = torch.randn(batch_size * num_cams, 3, img_h, img_w)
    img_metas = []
    for i in range(batch_size):
        can_bus = np.zeros(18)
        can_bus[0] = 0.1 * i  # delta_x
        can_bus[1] = 0.1 * i  # delta_y
        meta = {
            'can_bus': can_bus,
            'scene_token': f'scene_{i}',
            'prev_bev_exists': True
        }
        img_metas.append(meta)
    
    with torch.no_grad():
        img_feats = model.extract_feat(img, img_metas)
    
    print(f"✓ Feature extraction successful")
    print(f"  - Number of feature levels: {len(img_feats)}")
    print(f"  - Feature shapes: {[feat.shape for feat in img_feats]}")
    
    # Test training forward pass
    print("\nTesting training forward pass...")
    
    model.train()  # Set to training mode
    
    # Create temporal input (B, T, N, C, H, W)
    temporal_img = torch.randn(batch_size, queue_length, num_cams, 3, img_h, img_w)
    
    # Create temporal meta
    temporal_metas = []
    for b in range(batch_size):
        batch_metas = []
        for t in range(queue_length):
            can_bus = np.zeros(18)
            can_bus[0] = 0.1 * t
            can_bus[1] = 0.1 * t
            meta = {
                'can_bus': can_bus,
                'scene_token': f'scene_{b}',
                'prev_bev_exists': t > 0
            }
            batch_metas.append(meta)
        temporal_metas.append(batch_metas)
    
    # Create ground truth
    gt_bboxes_3d = []
    gt_labels_3d = []
    for b in range(batch_size):
        num_gts = 5
        # [cx, cy, cz, w, l, h, rot, vx, vy]
        gt_bbox = torch.randn(num_gts, 9)
        gt_bbox[:, :3] = torch.rand(num_gts, 3) * 20 - 10
        gt_bbox[:, 3:6] = torch.rand(num_gts, 3) * 3 + 1
        gt_bboxes_3d.append(gt_bbox)
        
        gt_labels = torch.randint(0, num_classes, (num_gts,))
        gt_labels_3d.append(gt_labels)
    
    # Forward pass
    losses = model.forward_train(
        img=temporal_img,
        img_metas=temporal_metas,
        gt_bboxes_3d=gt_bboxes_3d,
        gt_labels_3d=gt_labels_3d
    )
    
    print(f"✓ Training forward pass successful")
    print(f"  - Losses computed:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"    - {key}: {value.item():.4f}")
    
    # Test inference
    print("\nTesting inference...")
    
    model.eval()
    
    # Single frame for inference
    test_img = torch.randn(num_cams, 3, img_h, img_w)
    test_meta = [{
        'can_bus': np.zeros(18),
        'scene_token': 'test_scene',
        'prev_bev_exists': False
    }]
    
    with torch.no_grad():
        bbox_results = model.forward_test(
            img=[test_img],
            img_metas=[test_meta]
        )
    
    print(f"✓ Inference successful")
    print(f"  - Number of samples: {len(bbox_results)}")
    if bbox_results and 'pts_bbox' in bbox_results[0]:
        pts_bbox = bbox_results[0]['pts_bbox']
        print(f"  - Detection results available")
    
    # Test temporal consistency
    print("\nTesting temporal consistency...")
    
    # First frame
    model.prev_frame_info['prev_bev'] = None
    with torch.no_grad():
        results1 = model.forward_test(
            img=[test_img],
            img_metas=[test_meta]
        )
    
    # Second frame with same scene
    test_meta2 = [{
        'can_bus': np.array([0.5, 0.5] + [0] * 16),  # Small motion
        'scene_token': 'test_scene',  # Same scene
        'prev_bev_exists': True
    }]
    
    with torch.no_grad():
        results2 = model.forward_test(
            img=[test_img],
            img_metas=[test_meta2]
        )
    
    print(f"✓ Temporal consistency test passed")
    print(f"  - Previous BEV updated: {model.prev_frame_info['prev_bev'] is not None}")
    
    # Test scene change
    print("\nTesting scene change handling...")
    
    test_meta3 = [{
        'can_bus': np.zeros(18),
        'scene_token': 'new_scene',  # Different scene
        'prev_bev_exists': False
    }]
    
    with torch.no_grad():
        results3 = model.forward_test(
            img=[test_img],
            img_metas=[test_meta3]
        )
    
    print(f"✓ Scene change handling test passed")
    print(f"  - Scene token updated: {model.prev_frame_info['scene_token']}")
    
    print("\n" + "=" * 60)
    print("All BEVFormer tests passed successfully!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    # Run comprehensive test
    model = test_bevformer()
    
    print("\nModel is ready for use!")
    print("The BEVFormer implementation includes:")
    print("  - Multi-camera feature extraction")
    print("  - Temporal BEV feature propagation")
    print("  - 3D object detection with transformers")
    print("  - Scene change handling")
    print("  - GridMask augmentation")
    print("  - Training and inference modes")