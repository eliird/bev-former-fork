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
from spatial_attention import SpatialCrossAttention
from tempral_attention import TemporalSelfAttention
from deformable_attention import MSDeformableAttention3D


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
            num_levels=attn_cfgs[0].get('num_levels', 1),
            batch_first=True  # BEVFormerLayer uses (bs, num_query, embed_dims)
        )
        
        # Create deformable attention instance for spatial attention
        deform_attn_cfg = attn_cfgs[1].get('deformable_attention', {})
        deformable_attention = MSDeformableAttention3D(
            embed_dims=deform_attn_cfg.get('embed_dims', embed_dims),
            num_levels=deform_attn_cfg.get('num_levels', 4),
            num_points=deform_attn_cfg.get('num_points', 8)
        )
        
        self.spatial_attn = SpatialCrossAttention(
            embed_dims=attn_cfgs[1].get('embed_dims', embed_dims),
            pc_range=attn_cfgs[1].get('pc_range'),
            deformable_attention=deformable_attention
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
        # The temporal attention always creates internal batch expansion, so we always need to expand reference points
        temporal_ref_points = ref_2d.repeat(2, 1, 1, 1) if ref_2d is not None else None
        
        if prev_bev is not None:
            # Create temporal value by stacking current query and prev_bev
            # Follow original BEVFormer pattern: torch.stack([query, prev_bev], 1).reshape(bs*2, num_query, embed_dims)
            temporal_key_value = torch.stack([query, prev_bev], 1).reshape(query.shape[0]*2, query.shape[1], query.shape[2])
        else:
            temporal_key_value = None  # Will be auto-created by temporal attention
        
        query = self.temporal_attn(
            query,
            temporal_key_value,  # key
            temporal_key_value,  # value
            query_pos=bev_pos,
            key_pos=bev_pos,
            key_padding_mask=query_key_padding_mask,
            reference_points=temporal_ref_points,
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


def test_bev_former_layer():
    """Test BEVFormerLayer module"""
    print("=" * 60)
    print("Testing BEVFormerLayer")
    print("=" * 60)
    
    # Config parameters from BEVFormer
    embed_dims = 256
    feedforward_channels = 1024
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    try:
        # Create attention configs as per BEVFormer
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
                    'num_levels': 4
                }
            }
        ]
        
        # Create model
        model = BEVFormerLayer(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            embed_dims=embed_dims,
            ffn_dropout=0.1
        )
        
        print(f"✓ Model created successfully")
        print(f"  - embed_dims: {embed_dims}")
        print(f"  - feedforward_channels: {feedforward_channels}")
        print(f"  - pc_range: {pc_range}")
        
        # Test inputs
        batch_size = 2
        bev_h, bev_w = 50, 50
        num_queries = bev_h * bev_w  # 2500 BEV queries
        num_cams = 6
        
        # BEV query (current BEV features)
        query = torch.randn(batch_size, num_queries, embed_dims)
        
        # Previous BEV for temporal attention
        prev_bev = torch.randn(batch_size, num_queries, embed_dims)
        
        # Multi-camera image features for spatial attention
        img_h, img_w = 25, 15
        num_levels = 4
        
        # Create multi-scale image features
        key_list = []
        value_list = []
        spatial_shapes_list = []
        
        for level in range(num_levels):
            h, w = img_h // (2 ** level), img_w // (2 ** level)
            h, w = max(h, 1), max(w, 1)
            
            level_key = torch.randn(batch_size, num_cams, embed_dims, h, w)
            level_value = torch.randn(batch_size, num_cams, embed_dims, h, w)
            
            # Reshape to [num_cams, h*w, bs, embed_dims] as expected by spatial attention
            level_key = level_key.permute(1, 3, 4, 0, 2).reshape(num_cams, h * w, batch_size, embed_dims)
            level_value = level_value.permute(1, 3, 4, 0, 2).reshape(num_cams, h * w, batch_size, embed_dims)
            
            key_list.append(level_key)
            value_list.append(level_value)
            spatial_shapes_list.append([h, w])
        
        # Concatenate all levels
        key = torch.cat(key_list, dim=1)  # [num_cams, total_hw, bs, embed_dims]
        value = torch.cat(value_list, dim=1)
        
        spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long)
        level_start_index = torch.cat([
            torch.tensor([0]), 
            torch.tensor([h*w for h, w in spatial_shapes_list]).cumsum(0)[:-1]
        ])
        
        # Reference points for temporal attention (2D BEV grid)
        ref_2d = torch.rand(batch_size, num_queries, 1, 2)  # BEV uses single level
        
        # Reference points for spatial attention (3D with Z-anchors)  
        num_Z_anchors = 4
        ref_3d = torch.rand(batch_size, num_queries, num_levels, 2)
        reference_points_cam = torch.rand(batch_size, num_queries, num_Z_anchors, 2)
        
        # BEV mask for spatial attention
        bev_mask = torch.zeros(num_cams, batch_size, num_queries, dtype=torch.bool)
        for cam_id in range(num_cams):
            # Each camera sees overlapping regions
            start_idx = (cam_id * num_queries // (num_cams + 2))
            end_idx = ((cam_id + 3) * num_queries // (num_cams + 2))
            end_idx = min(end_idx, num_queries)
            bev_mask[cam_id, :, start_idx:end_idx] = True
        
        # BEV positional encoding
        bev_pos = torch.randn(batch_size, num_queries, embed_dims)
        
        print(f"✓ Test inputs created")
        print(f"  - query shape: {query.shape}")
        print(f"  - key shape: {key.shape}")
        print(f"  - value shape: {value.shape}")
        print(f"  - prev_bev shape: {prev_bev.shape}")
        print(f"  - spatial_shapes: {spatial_shapes}")
        print(f"  - bev_mask: {bev_mask.shape}")
        
        # Forward pass through complete BEVFormerLayer
        with torch.no_grad():
            output = model(
                query=query,
                key=key,
                value=value,
                bev_pos=bev_pos,
                bev_h=bev_h,
                bev_w=bev_w,
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=prev_bev
            )
        
        print(f"✓ Forward pass successful")
        print(f"  - output shape: {output.shape}")
        print(f"  - expected shape: {query.shape}")
        
        # Verify output shape
        assert output.shape == query.shape, f"Output shape {output.shape} != expected {query.shape}"
        
        # Verify output is not NaN or Inf
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
        
        # Test without previous BEV (first frame)
        with torch.no_grad():
            output_no_prev = model(
                query=query,
                key=key,
                value=value,
                bev_pos=bev_pos,
                bev_h=bev_h,
                bev_w=bev_w,
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=None  # No previous BEV
            )
        
        print(f"✓ Forward pass without prev_bev successful")
        print(f"  - output shape: {output_no_prev.shape}")
        
        assert output_no_prev.shape == query.shape, f"Output shape {output_no_prev.shape} != expected {query.shape}"
        assert torch.isfinite(output_no_prev).all(), "Output contains NaN or Inf values"
        
        print("✓ All assertions passed")
        print("🎉 BEVFormerLayer test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_bev_former_layer()