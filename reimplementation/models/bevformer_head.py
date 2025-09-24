import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Any
import copy
import numpy as np

# Import our reimplemented modules
from .perception_transformer import PerceptionTransformer
from .learned_positional_encoding import LearnedPositionalEncoding
from .nms_coder import NMSFreeCoder
from .hungarian_assigner import HungarianAssigner3D
from .focal_loss import FocalLoss
from .l1_loss import L1Loss
from .GIoULoss import GIoULoss
from .pseudo_sampler import PseudoSampler
from .utils_bevhead import (
    multi_apply, reduce_mean, bias_init_with_prob, 
    inverse_sigmoid, normalize_bbox
)
from .costs import FocalLossCost, BBox3DL1Cost, IoUCost


class BEVFormerHead(nn.Module):
    """BEVFormer detection head for 3D object detection.
    
    Args:
        num_classes (int): Number of object classes
        in_channels (int): Number of input channels (embed_dims)
        num_query (int): Number of object queries
        num_reg_fcs (int): Number of FFN layers in regression branch
        transformer (dict): Config for PerceptionTransformer
        bbox_coder (dict): Config for NMSFreeCoder
        positional_encoding (dict): Config for positional encoding
        loss_cls (dict): Config for classification loss
        loss_bbox (dict): Config for bbox regression loss
        loss_iou (dict): Config for IoU loss
        train_cfg (dict): Training config with assigner settings
        test_cfg (dict): Testing config
        bev_h (int): Height of BEV queries
        bev_w (int): Width of BEV queries
        with_box_refine (bool): Whether to refine reference points
        as_two_stage (bool): Whether to use two-stage paradigm
    """
    
    def __init__(
        self,
        num_classes=10,
        in_channels=256,
        num_query=900,
        num_reg_fcs=2,
        transformer=None,
        bbox_coder=None,
        positional_encoding=None,
        loss_cls=None,
        loss_bbox=None, 
        loss_iou=None,
        train_cfg=None,
        test_cfg=None,
        bev_h=200,
        bev_w=200,
        with_box_refine=True,
        as_two_stage=False,
        code_size=10,
        code_weights=None,
        sync_cls_avg_factor=True,
        bg_cls_weight=0.0,
        **kwargs
    ):
        super(BEVFormerHead, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = in_channels
        self.num_query = num_query
        self.num_reg_fcs = num_reg_fcs
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
            
        self.code_size = code_size
        if code_weights is None:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        else:
            self.code_weights = code_weights
            
        self.bg_cls_weight = bg_cls_weight
        self.sync_cls_avg_factor = sync_cls_avg_factor
        
        # Build transformer
        if transformer is not None:
            self.transformer = PerceptionTransformer(**transformer)
        else:
            self.transformer = None
            
        # Build bbox coder
        if bbox_coder is not None:
            self.bbox_coder = NMSFreeCoder(**bbox_coder)
        else:
            # Default bbox coder
            self.bbox_coder = NMSFreeCoder(
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=300,
                num_classes=num_classes
            )
            
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        
        # Build positional encoding
        if positional_encoding is not None:
            self.positional_encoding = LearnedPositionalEncoding(**positional_encoding)
        else:
            self.positional_encoding = LearnedPositionalEncoding(
                num_feats=128,
                row_num_embed=bev_h,
                col_num_embed=bev_w
            )
            
        # Build losses
        if loss_cls is not None:
            self.loss_cls = FocalLoss(**loss_cls)
        else:
            self.loss_cls = FocalLoss(
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0
            )
            
        if loss_bbox is not None:
            self.loss_bbox = L1Loss(**loss_bbox)
        else:
            self.loss_bbox = L1Loss(loss_weight=0.25)
            
        if loss_iou is not None:
            self.loss_iou = GIoULoss(**loss_iou)
        else:
            self.loss_iou = GIoULoss(loss_weight=0.0)
            
        # Determine output channels for classification
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
            
        # Build assigner and sampler for training
        if train_cfg is not None:
            pts_train_cfg = train_cfg.get('pts', train_cfg)
            assigner_cfg = pts_train_cfg.get('assigner', None)
            if assigner_cfg is not None:
                self.assigner = HungarianAssigner3D(**assigner_cfg)
            else:
                # Default assigner
                self.assigner = HungarianAssigner3D(
                    cls_cost=FocalLossCost(weight=2.0),
                    reg_cost=BBox3DL1Cost(weight=0.25),
                    iou_cost=IoUCost(weight=0.0),
                    pc_range=self.pc_range
                )
        else:
            self.assigner = None
            
        # Always use PseudoSampler for DETR-style detection
        self.sampler = PseudoSampler()
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # Convert code_weights to parameter
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, dtype=torch.float32),
            requires_grad=False
        )
        
        # Initialize layers
        self._init_layers()
        
    def _init_layers(self):
        """Initialize classification and regression branches."""
        
        # Classification branch
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)
        
        # Regression branch
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)
        
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        
        # Number of decoder predictions
        if self.transformer is not None and self.transformer.decoder is not None:
            num_pred = self.transformer.decoder.num_layers
        else:
            num_pred = 6  # Default
            
        if self.as_two_stage:
            num_pred = num_pred + 1
            
        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
            
        # Initialize embeddings
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights of the head."""
        # Initialize transformer weights
        if self.transformer is not None:
            if hasattr(self.transformer, 'init_weights'):
                self.transformer.init_weights()
                
        # Initialize classification bias
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                if isinstance(m, nn.Sequential):
                    nn.init.constant_(m[-1].bias, bias_init)
                    
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        """Forward function.
        
        Args:
            mlvl_feats (list[Tensor]): Multi-level features from backbone + neck
                Each tensor has shape (B, N, C, H, W) where N is number of cameras
            img_metas (list[dict]): Meta information of each sample
            prev_bev (Tensor, optional): Previous BEV features for temporal modeling
            only_bev (bool): If True, only compute BEV features without detection
            
        Returns:
            dict: Outputs containing:
                - bev_embed: BEV features
                - all_cls_scores: Classification scores from all decoder layers
                - all_bbox_preds: Bounding box predictions from all decoder layers
                - enc_cls_scores: Encoder classification scores (None for now)
                - enc_bbox_preds: Encoder bbox predictions (None for now)
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device
        
        # Get embeddings
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)
        
        # Create BEV mask and positional encoding
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        
        if only_bev:
            # Only compute BEV features without detection
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            # Full forward pass with detection
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )
            
            bev_embed, hs, init_reference, inter_references = outputs
            
            # Process decoder outputs
            hs = hs.permute(0, 2, 1, 3)  # (num_layers, bs, num_query, embed_dims)
            outputs_classes = []
            outputs_coords = []
            
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                    
                reference = inverse_sigmoid(reference)
                outputs_class = self.cls_branches[lvl](hs[lvl])
                tmp = self.reg_branches[lvl](hs[lvl])
                
                # Update reference points
                assert reference.shape[-1] == 3
                tmp[..., 0:2] += reference[..., 0:2]
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 4:5] += reference[..., 2:3]
                tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
                
                # Denormalize to actual coordinates
                tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
                tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
                
                outputs_coord = tmp
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
                
            outputs_classes = torch.stack(outputs_classes)
            outputs_coords = torch.stack(outputs_coords)
            
            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
            }
            
            return outs
            
    def _get_target_single(self,
                          cls_score,
                          bbox_pred,
                          gt_labels,
                          gt_bboxes,
                          gt_bboxes_ignore=None):
        """Compute regression and classification targets for one image.
        
        Args:
            cls_score (Tensor): Box score logits from single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Predicted bboxes from single decoder layer
                for one image. Shape [num_query, code_size].
            gt_labels (Tensor): Ground truth class labels for one image
                with shape (num_gts, ).
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9) in [cx, cy, cz, w, l, h, rot, vx, vy] format.
            gt_bboxes_ignore (Tensor, optional): Ignored bboxes. Default None.
            
        Returns:
            tuple[Tensor]: Tuple containing:
                - labels (Tensor): Labels of each image
                - label_weights (Tensor): Label weights of each image  
                - bbox_targets (Tensor): BBox targets of each image
                - bbox_weights (Tensor): BBox weights of each image
                - pos_inds (Tensor): Sampled positive indices
                - neg_inds (Tensor): Sampled negative indices
        """
        num_bboxes = bbox_pred.size(0)
        
        # Assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        
        assign_result = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore
        )
        
        sampling_result = self.sampler.sample(
            assign_result, bbox_pred, gt_bboxes
        )
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        # Label targets
        # For sigmoid focal loss, background should be all zeros (multi-label format)
        # not a separate background class
        labels = gt_bboxes.new_zeros(
            (num_bboxes, self.num_classes), dtype=torch.float32
        )
        # Set positive samples to 1 for their corresponding class
        if len(pos_inds) > 0:
            pos_labels = gt_labels[sampling_result.pos_assigned_gt_inds]
            labels[pos_inds, pos_labels] = 1.0
        label_weights = gt_bboxes.new_ones(num_bboxes)
        
        # Bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR style: directly use GT boxes as targets
        if len(pos_inds) > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)
                
    def get_targets(self,
                   cls_scores_list,
                   bbox_preds_list,
                   gt_bboxes_list,
                   gt_labels_list,
                   gt_bboxes_ignore_list=None):
        """Compute regression and classification targets for batch images.
        
        Args:
            cls_scores_list (list[Tensor]): Box scores for each image
            bbox_preds_list (list[Tensor]): Bbox predictions for each image
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
            gt_labels_list (list[Tensor]): Ground truth labels for each image
            gt_bboxes_ignore_list (list[Tensor], optional): Ignore bboxes
            
        Returns:
            tuple: Tuple containing targets components
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports gt_bboxes_ignore setting to None.'
            
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list
        )
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)
                
    def loss_single(self,
                   cls_scores,
                   bbox_preds,
                   gt_bboxes_list,
                   gt_labels_list,
                   gt_bboxes_ignore_list=None):
        """Loss function for single decoder layer.
        
        Args:
            cls_scores (Tensor): Box score logits from single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Bbox predictions from single decoder layer
                for all images. Shape [bs, num_query, code_size].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
            gt_labels_list (list[Tensor]): Ground truth labels for each image
            gt_bboxes_ignore_list (list[Tensor], optional): Ignore bboxes
            
        Returns:
            tuple[Tensor]: Loss components (loss_cls, loss_bbox)
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        
        cls_reg_targets = self.get_targets(
            cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list,
            gt_bboxes_ignore_list
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
         
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        
        # Classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        
        # Construct weighted avg_factor to match with official DETR
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
            
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor])
            )
            
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )
        
        # Compute average number of gt boxes for normalization
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        
        # Regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )
        
        # IoU loss - compute GIoU for better box regression
        if hasattr(self, 'loss_iou') and self.loss_iou.loss_weight > 0:
            # Convert predictions and targets to corners format for IoU computation
            # For now, use L1 loss as a placeholder since GIoU requires corner conversion
            loss_iou = self.loss_iou(
                bbox_preds[isnotnan, :10],
                normalized_bbox_targets[isnotnan, :10],
                bbox_weights[isnotnan, :10],
                avg_factor=num_total_pos
            )
        else:
            # IoU loss is disabled (weight=0) but still return for compatibility
            loss_iou = loss_bbox.new_zeros(1)
        
        # Handle NaN/Inf
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        loss_iou = torch.nan_to_num(loss_iou)
        
        return loss_cls, loss_bbox, loss_iou
        
    def loss(self,
            gt_bboxes_list,
            gt_labels_list,
            preds_dicts,
            gt_bboxes_ignore=None,
            img_metas=None):
        """Compute losses.
        
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
            gt_labels_list (list[Tensor]): Ground truth labels for each image
            preds_dicts (dict): Predictions from forward containing:
                - all_cls_scores: Classification scores of all decoder layers
                - all_bbox_preds: Bbox predictions of all decoder layers
            gt_bboxes_ignore (list[Tensor], optional): Ignored bboxes
            img_metas (list[dict], optional): Image meta information
            
        Returns:
            dict[str, Tensor]: Dictionary of loss components
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'gt_bboxes_ignore setting to None.'
            
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        
        # Format gt_bboxes
        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(device)
            if hasattr(gt_bboxes, 'gravity_center') else gt_bboxes.to(device)
            for gt_bboxes in gt_bboxes_list
        ]
        
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list
        )
        
        loss_dict = dict()
        
        # Loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        
        # Loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
            
        return loss_dict
        
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from predictions.
        
        Args:
            preds_dicts (dict): Prediction results from forward
            img_metas (list[dict]): Image meta information
            rescale (bool): Whether to rescale bboxes
            
        Returns:
            list[tuple]: Decoded bbox, scores and labels for each sample
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        
        num_samples = len(preds_dicts)
        ret_list = []
        
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            
            # Adjust z center
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            
            scores = preds['scores']
            labels = preds['labels']
            
            ret_list.append([bboxes, scores, labels])
            
        return ret_list


def test_bevformer_head():
    """Test BEVFormerHead implementation"""
    print("=" * 60)
    print("Testing BEVFormerHead")
    print("=" * 60)
    
    # Configuration
    embed_dims = 256
    num_classes = 10
    num_query = 900
    bev_h, bev_w = 200, 200
    batch_size = 2
    num_cams = 6
    num_levels = 4
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    # Create config dictionaries
    transformer_cfg = dict(
        embed_dims=embed_dims,
        encoder=dict(
            num_layers=3,
            pc_range=pc_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayers=dict(
                feedforward_channels=512,
                ffn_dropout=0.1
            )
        ),
        decoder=dict(
            num_layers=3,
            return_intermediate=True,
            transformerlayers=dict(
                feedforward_channels=512,
                ffn_dropout=0.1
            )
        ),
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True
    )
    
    bbox_coder_cfg = dict(
        pc_range=pc_range,
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_num=300,
        num_classes=num_classes
    )
    
    train_cfg = dict(
        pts=dict(
            assigner=dict(
                cls_cost=dict(weight=2.0),
                reg_cost=dict(weight=0.25),
                iou_cost=dict(weight=0.0),
                pc_range=pc_range
            )
        )
    )
    
    # Create BEVFormerHead
    print("Creating BEVFormerHead...")
    bevformer_head = BEVFormerHead(
        num_classes=num_classes,
        in_channels=embed_dims,
        num_query=num_query,
        transformer=transformer_cfg,
        bbox_coder=bbox_coder_cfg,
        train_cfg=train_cfg,
        bev_h=bev_h,
        bev_w=bev_w,
        with_box_refine=True,
        as_two_stage=False
    )
    
    print(f"✓ BEVFormerHead created successfully")
    print(f"  - Number of parameters: {sum(p.numel() for p in bevformer_head.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    
    # Create dummy multi-level features
    feat_shapes = [(116, 200), (58, 100), (29, 50), (15, 25)]
    mlvl_feats = []
    for h, w in feat_shapes:
        feat = torch.randn(batch_size, num_cams, embed_dims, h, w)
        mlvl_feats.append(feat)
    
    # Create dummy img_metas
    img_metas = []
    for i in range(batch_size):
        can_bus = torch.zeros(18)  # CAN bus signals
        can_bus[0] = 0.1 * i  # delta_x
        can_bus[1] = 0.1 * i  # delta_y
        can_bus[-2] = 0.1  # ego_angle
        can_bus[-1] = 0.05  # rotation_angle
        
        meta = {
            'can_bus': can_bus.numpy(),  # CAN bus signals as numpy array
            'scene_token': f'scene_{i}',
            'lidar2img': torch.randn(num_cams, 4, 4)
        }
        img_metas.append(meta)
    
    # Forward pass
    with torch.no_grad():
        outputs = bevformer_head(mlvl_feats, img_metas)
    
    print(f"✓ Forward pass successful")
    print(f"  - BEV embed shape: {outputs['bev_embed'].shape if outputs['bev_embed'] is not None else 'None'}")
    print(f"  - Classification scores shape: {outputs['all_cls_scores'].shape}")
    print(f"  - Bbox predictions shape: {outputs['all_bbox_preds'].shape}")
    
    # Test loss computation
    print("\nTesting loss computation...")
    
    # Create dummy ground truth
    gt_bboxes_list = []
    gt_labels_list = []
    for i in range(batch_size):
        num_gts = 5
        # Create gt bboxes: [cx, cy, cz, w, l, h, rot, vx, vy]
        gt_bbox = torch.randn(num_gts, 9)
        gt_bbox[:, :3] = torch.rand(num_gts, 3) * 50 - 25  # Center in range
        gt_bbox[:, 3:6] = torch.rand(num_gts, 3) * 5 + 1   # Size
        gt_bboxes_list.append(gt_bbox)
        
        gt_labels = torch.randint(0, num_classes, (num_gts,))
        gt_labels_list.append(gt_labels)
    
    # Compute loss
    loss_dict = bevformer_head.loss(
        gt_bboxes_list=gt_bboxes_list,
        gt_labels_list=gt_labels_list,
        preds_dicts=outputs,
        img_metas=img_metas
    )
    
    print(f"✓ Loss computation successful")
    for key, value in loss_dict.items():
        print(f"  - {key}: {value.item():.4f}")
    
    # Test inference
    print("\nTesting inference...")
    
    # Get bboxes
    results = bevformer_head.get_bboxes(outputs, img_metas)
    
    print(f"✓ Inference successful")
    for i, (bboxes, scores, labels) in enumerate(results):
        print(f"  - Sample {i}: {bboxes.shape[0]} detections")
    
    # Test with previous BEV
    print("\nTesting with temporal BEV...")
    
    prev_bev = torch.randn(bev_h * bev_w, batch_size, embed_dims)
    with torch.no_grad():
        outputs_temporal = bevformer_head(mlvl_feats, img_metas, prev_bev=prev_bev)
    
    print(f"✓ Temporal BEV test passed")
    
    # Test BEV-only mode
    print("\nTesting BEV-only mode...")
    
    with torch.no_grad():
        bev_only = bevformer_head(mlvl_feats, img_metas, only_bev=True)
    
    print(f"✓ BEV-only mode test passed")
    print(f"  - BEV features shape: {bev_only.shape}")
    
    print("\n" + "=" * 60)
    print("All BEVFormerHead tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_bevformer_head()