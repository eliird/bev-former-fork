'''
encoder=dict(
    type='BEVFormerEncoder',
    num_layers=6,
    pc_range=point_cloud_range,
    num_points_in_pillar=4,
    return_intermediate=False,
    transformerlayers=dict(
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
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                            'ffn', 'norm'))),
'''

import torch
import torch.nn as nn
from reimplementation.models.bev_former_layer import BEVFormerLayer


class BEVFormerEncoder(nn.Module):
    """BEVFormer Encoder with multiple transformer layers.
    
    Args:
        num_layers (int): Number of transformer layers. Default: 6
        embed_dims (int): Embedding dimensions. Default: 256  
        pc_range (list): Point cloud range. Default: None
        num_points_in_pillar (int): Number of points in pillar. Default: 4
        return_intermediate (bool): Whether to return intermediate results. Default: False
        feedforward_channels (int): FFN hidden dimensions. Default: 1024
        ffn_dropout (float): FFN dropout rate. Default: 0.1
        **kwargs: Additional arguments
    """
    
    def __init__(self,
                 num_layers=6,
                 embed_dims=256,
                 pc_range=None,
                 num_points_in_pillar=4,
                 return_intermediate=False,
                 feedforward_channels=1024,
                 ffn_dropout=0.1,
                 num_levels=4,
                 **kwargs):
        super(BEVFormerEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.return_intermediate = return_intermediate
        
        # Create attention configs
        attn_cfgs = [
            {
                'type': 'TemporalSelfAttention',
                'embed_dims': embed_dims,
                'num_levels': 1
            },
            {
                'type': 'SpatialCrossAttention', 
                'embed_dims': embed_dims,
                'pc_range': pc_range,
                'deformable_attention': {
                    'type': 'MSDeformableAttention3D',
                    'embed_dims': embed_dims,
                    'num_points': 8,
                    'num_levels': num_levels
                }
            }
        ]
        
        # Build transformer layers
        self.layers = nn.ModuleList([
            BEVFormerLayer(
                attn_cfgs=attn_cfgs,
                feedforward_channels=feedforward_channels,
                embed_dims=embed_dims,
                ffn_dropout=ffn_dropout,
                **kwargs
            ) for _ in range(num_layers)
        ])
        
        self.fp16_enabled = False
    
    def forward(self,
                bev_query,
                key,
                value,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                shift=0,
                **kwargs):
        """Forward function for BEVFormerEncoder.
        
        Args:
            bev_query (Tensor): BEV queries with shape [bs, num_queries, embed_dims]
            key (Tensor): Image feature keys from backbone/neck
            value (Tensor): Image feature values from backbone/neck  
            bev_h (int): Height of BEV grid
            bev_w (int): Width of BEV grid
            bev_pos (Tensor): BEV positional encodings
            spatial_shapes (Tensor): Spatial shapes of multi-scale features
            level_start_index (Tensor): Start indices for each level
            prev_bev (Tensor): Previous BEV features for temporal modeling
            shift (int): Shift for temporal modeling
            **kwargs: Additional arguments
            
        Returns:
            Tensor: Output BEV features with shape [bs, num_queries, embed_dims]
            or list of intermediate results if return_intermediate=True
        """
        
        output = bev_query
        intermediate = []
        
        for lid, layer in enumerate(self.layers):
            output = layer(
                query=output,
                key=key,
                value=value,
                bev_pos=bev_pos,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=prev_bev,
                shift=shift,
                **kwargs
            )
            
            if self.return_intermediate:
                intermediate.append(output)
        
        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output