"""
BEVFormer Layer implementation based on the config:
type='BEVFormerLayer',
attn_cfgs=[
    dict(
        type='TemporalSelfAttention',
        embed_dims=_dim_,
        num_levels=1),
    dict(
        type='SpatialCrossAttention',
        pc_range=point_cloud_range,
        deformable_attention=dict(
            type='MSDeformableAttention3D',
            embed_dims=_dim_,
            num_points=8,
            num_levels=_num_levels_),
        embed_dims=_dim_,
    )
],
feedforward_channels=_ffn_dim_,
ffn_dropout=0.1,
operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
"""

import copy
import warnings
import torch.nn as nn
import torch
import torch.nn.functional as F
from .spatial_attention import SpatialCrossAttention
from .tempral_attention import TemporalSelfAttention


class FFN(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, embed_dims, feedforward_channels, ffn_dropout=0.0, ffn_num_fcs=2):
        super(FFN, self).__init__()
        assert ffn_num_fcs >= 2, 'num_fcs should be no less than 2. got {}.'.format(ffn_num_fcs)
        
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.ffn_num_fcs = ffn_num_fcs
        
        layers = []
        in_channels = embed_dims
        for _ in range(ffn_num_fcs - 1):
            layers.append(nn.Sequential(
                nn.Linear(in_channels, feedforward_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(ffn_dropout)
            ))
            in_channels = feedforward_channels
        
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_dropout))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x, residual=None):
        out = self.layers(x)
        if residual is not None:
            out += residual
        return out


class BEVFormerLayer(nn.Module):
    """BEVFormer Layer with temporal self-attention and spatial cross-attention.
    
    Args:
        attn_cfgs (list[dict]): List of attention configs
        feedforward_channels (int): The hidden dimension for FFNs
        ffn_dropout (float): Probability of dropout in ffn
        operation_order (tuple[str]): The execution order of operations
        embed_dims (int): The embedding dimension
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 embed_dims=256,
                 ffn_dropout=0.0,
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__()
        
        self.embed_dims = embed_dims
        
        # Build attentions - always temporal then spatial
        self.temporal_attn = TemporalSelfAttention(
            embed_dims=attn_cfgs[0].get('embed_dims', embed_dims),
            num_levels=attn_cfgs[0].get('num_levels', 1)
        )
        
        self.spatial_attn = SpatialCrossAttention(
            embed_dims=attn_cfgs[1].get('embed_dims', embed_dims),
            pc_range=attn_cfgs[1].get('pc_range'),
            deformable_attention=attn_cfgs[1].get('deformable_attention', {})
        )
        
        # Build norms - need 3 norms total
        self.norm1 = nn.LayerNorm(embed_dims)  # after temporal
        self.norm2 = nn.LayerNorm(embed_dims)  # after spatial  
        self.norm3 = nn.LayerNorm(embed_dims)  # after ffn
        
        # Build FFN
        self.ffn = FFN(embed_dims, feedforward_channels, ffn_dropout, ffn_num_fcs)
        
        self.fp16_enabled = False

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                bev_mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for BEVFormerLayer.
        Fixed order: self_attn -> norm -> cross_attn -> norm -> ffn -> norm
        """
        
        # Temporal self attention
        query = self.temporal_attn(
            query,
            prev_bev if prev_bev is not None else query,  # key
            prev_bev if prev_bev is not None else query,  # value
            query_pos=bev_pos,
            key_pos=bev_pos,
            key_padding_mask=query_key_padding_mask,
            reference_points=ref_2d,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device) if bev_h and bev_w else None,
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)
        
        # Norm after temporal
        query = self.norm1(query)
        
        # Spatial cross attention  
        query = self.spatial_attn(
            query,
            key,
            value,
            query_pos=query_pos,
            key_pos=key_pos,
            reference_points=ref_3d,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs)
        
        # Norm after spatial
        query = self.norm2(query)
        
        # FFN
        query = self.ffn(query)
        
        # Final norm
        query = self.norm3(query)

        return query