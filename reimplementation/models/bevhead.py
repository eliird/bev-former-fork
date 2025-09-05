import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict
from dataclasses import dataclass
# from torch.nn import CrossEntropyLoss
from reimplementation.models.cross_entropy import CrossEntropyLoss
# from mmdet.models.losses import CrossEntropyLoss

@dataclass
class TransformerConfig:
    num_classes: int
    embed_dims: int = 256
    num_reg_fcs: int = 2
    sync_cls_avg_factor: bool = False
    loss_cls: CrossEntropyLoss = CrossEntropyLoss(class_weight=0.1, use_sigmoid=False, loss_weight=1.0, class_weight=1.0)
    
    

class BEVFormerHead(nn.Module):
    def __init__(
        self, 
        bev_h, 
        bev_w, 
        num_query,
        bbox_coder=None,
        transformer=None,
        num_cls_fcs=2, 
        with_box_refine=True, 
        as_two_stage=False,
        code_size=10,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
    ):
        super(BEVFormerHead, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_query = num_query
        self.fp16_enabled = False
     
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.code_size = code_size
        self.code_weights = code_weights
        
        self.transformer = transformer
        
        # Need to define the coder model here
        self.bbox_coder = None
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False
        )

        # Additional layers and parameters would be defined here

    def forward(self, x):
        # Forward pass implementation would go here
        pass