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
from .bev_former_layer import BEVFormerLayer


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
                ref_2d=None,
                ref_3d=None,
                reference_points_cam=None,
                bev_mask=None,
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
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                **kwargs
            )
            
            if self.return_intermediate:
                intermediate.append(output)
        
        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output


def test_bev_former_encoder():
    """Test BEVFormerEncoder module"""
    print("=" * 60)
    print("Testing BEVFormerEncoder")
    print("=" * 60)
    
    # Config parameters from BEVFormer
    embed_dims = 256
    num_layers = 6
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    feedforward_channels = 1024
    num_levels = 4
    
    try:
        # Create model
        model = BEVFormerEncoder(
            num_layers=num_layers,
            embed_dims=embed_dims,
            pc_range=pc_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            feedforward_channels=feedforward_channels,
            ffn_dropout=0.1,
            num_levels=num_levels
        )
        
        print(f"✓ Model created successfully")
        print(f"  - num_layers: {num_layers}")
        print(f"  - embed_dims: {embed_dims}")
        print(f"  - pc_range: {pc_range}")
        print(f"  - feedforward_channels: {feedforward_channels}")
        
        # Test inputs
        batch_size = 1  # Reduced for memory
        bev_h, bev_w = 30, 30  # Reduced for memory
        num_queries = bev_h * bev_w  # 900 BEV queries
        
        # BEV query (initial BEV embeddings)
        bev_query = torch.randn(batch_size, num_queries, embed_dims)
        
        # Multi-camera image features
        num_cams = 6
        img_h, img_w = 15, 10  # Reduced for memory
        
        # Create multi-scale image features as keys and values
        key_list = []
        value_list = []
        spatial_shapes_list = []
        
        for level in range(num_levels):
            h, w = img_h // (2 ** level), img_w // (2 ** level)
            h, w = max(h, 1), max(w, 1)
            
            level_key = torch.randn(batch_size, num_cams, embed_dims, h, w)
            level_value = torch.randn(batch_size, num_cams, embed_dims, h, w)
            
            # Reshape to [num_cams, h*w, bs, embed_dims] as expected
            level_key = level_key.permute(1, 3, 4, 0, 2).reshape(num_cams, h * w, batch_size, embed_dims)
            level_value = level_value.permute(1, 3, 4, 0, 2).reshape(num_cams, h * w, batch_size, embed_dims)
            
            key_list.append(level_key)
            value_list.append(level_value)
            spatial_shapes_list.append([h, w])
        
        # Concatenate all levels
        key = torch.cat(key_list, dim=1)
        value = torch.cat(value_list, dim=1)
        
        spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long)
        level_start_index = torch.cat([
            torch.tensor([0]),
            torch.tensor([h*w for h, w in spatial_shapes_list]).cumsum(0)[:-1]
        ])
        
        # BEV positional encoding
        bev_pos = torch.randn(batch_size, num_queries, embed_dims)
        
        # Previous BEV for temporal modeling
        prev_bev = torch.randn(batch_size, num_queries, embed_dims)
        
        # Shift for temporal alignment
        shift = torch.randn(batch_size, 2)  # x, y shift
        
        print(f"✓ Test inputs created")
        print(f"  - bev_query shape: {bev_query.shape}")
        print(f"  - key shape: {key.shape}")
        print(f"  - value shape: {value.shape}")
        print(f"  - spatial_shapes: {spatial_shapes}")
        print(f"  - prev_bev shape: {prev_bev.shape}")
        
        # Reference points for temporal attention (2D BEV grid)
        ref_2d = torch.rand(batch_size, num_queries, 1, 2)  # BEV uses single level
        
        # Reference points for spatial attention (3D with Z-anchors)  
        num_Z_anchors = 4
        ref_3d = torch.rand(batch_size, num_queries, num_levels, 2)
        reference_points_cam = torch.rand(batch_size, num_queries, num_Z_anchors, 2)
        
        # BEV mask for spatial attention
        num_cams = 6
        bev_mask = torch.zeros(num_cams, batch_size, num_queries, dtype=torch.bool)
        for cam_id in range(num_cams):
            # Each camera sees overlapping regions
            start_idx = (cam_id * num_queries // (num_cams + 2))
            end_idx = ((cam_id + 3) * num_queries // (num_cams + 2))
            end_idx = min(end_idx, num_queries)
            bev_mask[cam_id, :, start_idx:end_idx] = True
        
        # Forward pass through encoder
        with torch.no_grad():
            output = model(
                bev_query=bev_query,
                key=key,
                value=value,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_pos=bev_pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=prev_bev,
                shift=shift,
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask
            )
        
        print(f"✓ Forward pass successful")
        print(f"  - output shape: {output.shape}")
        print(f"  - expected shape: {bev_query.shape}")
        
        # Verify output shape
        assert output.shape == bev_query.shape, f"Output shape {output.shape} != expected {bev_query.shape}"
        
        # Verify output is not NaN or Inf
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
        
        # Test without previous BEV (first frame)
        with torch.no_grad():
            output_no_prev = model(
                bev_query=bev_query,
                key=key,
                value=value,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_pos=bev_pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=None,  # No previous BEV
                shift=0,  # No shift for first frame
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask
            )
        
        print(f"✓ Forward pass without prev_bev successful")
        print(f"  - output shape: {output_no_prev.shape}")
        
        assert output_no_prev.shape == bev_query.shape, f"Output shape {output_no_prev.shape} != expected {bev_query.shape}"
        assert torch.isfinite(output_no_prev).all(), "Output contains NaN or Inf values"
        
        # Test with return_intermediate=True
        model.return_intermediate = True
        with torch.no_grad():
            intermediate_outputs = model(
                bev_query=bev_query,
                key=key,
                value=value,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_pos=bev_pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=prev_bev,
                shift=shift,
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask
            )
        
        print(f"✓ Forward pass with return_intermediate=True successful")
        print(f"  - intermediate outputs shape: {intermediate_outputs.shape}")
        print(f"  - expected shape: [{num_layers}, {batch_size}, {num_queries}, {embed_dims}]")
        
        assert intermediate_outputs.shape == torch.Size([num_layers, batch_size, num_queries, embed_dims]), \
            f"Intermediate shape {intermediate_outputs.shape} != expected"
        
        print("✓ All assertions passed")
        print("🎉 BEVFormerEncoder test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_bev_former_encoder()