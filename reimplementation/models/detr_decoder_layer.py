import torch.nn as nn
import torch
from torch import Tensor
from .multi_head_attention import MultiheadAttention
from .custom_deformable_attention import CustomMSDeformableAttention


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


class DetrTransformerDecoderLayer(nn.Module):
    """Decoder layer in DETR transformer.
    
    Based on the config:
    type='DetrTransformerDecoderLayer',
    attn_cfgs=[
        dict(type='MultiheadAttention', embed_dims=_dim_, num_heads=8, dropout=0.1),
        dict(type='CustomMSDeformableAttention', embed_dims=_dim_, num_levels=1)
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
    
    Args:
        attn_cfgs (list[dict]): Attention configs for [self_attn, cross_attn]
        feedforward_channels (int): FFN hidden dimension
        embed_dims (int): Embedding dimension
        ffn_dropout (float): FFN dropout rate
        **kwargs: Additional arguments
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 embed_dims=256,
                 ffn_dropout=0.1,
                 ffn_num_fcs=2,
                 **kwargs):
        super(DetrTransformerDecoderLayer, self).__init__()
        
        self.embed_dims = embed_dims
        self.attn_cfgs = attn_cfgs
        
        # Build self-attention (MultiheadAttention)
        self_attn_cfg = attn_cfgs[0]
        self.self_attn = MultiheadAttention(
            embed_dims=self_attn_cfg.get('embed_dims', embed_dims),
            num_heads=self_attn_cfg.get('num_heads', 8),
            dropout=self_attn_cfg.get('dropout', 0.1),
            batch_first=False  # Decoder uses (num_query, bs, embed_dims)
        )
        
        # Build cross-attention (CustomMSDeformableAttention)
        cross_attn_cfg = attn_cfgs[1]
        self.cross_attn = CustomMSDeformableAttention(
            embed_dims=cross_attn_cfg.get('embed_dims', embed_dims),
            num_levels=cross_attn_cfg.get('num_levels', 1),
            num_heads=8,
            num_points=4,
            batch_first=False  # Decoder convention
        )
        
        # Build normalization layers - need 3 norms
        self.norm1 = nn.LayerNorm(embed_dims)  # after self_attn
        self.norm2 = nn.LayerNorm(embed_dims)  # after cross_attn
        self.norm3 = nn.LayerNorm(embed_dims)  # after ffn
        
        # Build FFN
        self.ffn = FFN(embed_dims, feedforward_channels, ffn_dropout, ffn_num_fcs)
        
        self.fp16_enabled = False

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward function for DetrTransformerDecoderLayer.
        
        Fixed order: self_attn -> norm -> cross_attn -> norm -> ffn -> norm
        
        Args:
            query (Tensor): Object queries with shape (num_queries, bs, embed_dims)
            key (Tensor): BEV features for cross-attention
            value (Tensor): BEV features for cross-attention 
            query_pos (Tensor): Query positional encoding
            key_pos (Tensor): Key positional encoding
            reference_points (Tensor): Reference points for deformable attention
            spatial_shapes (Tensor): Spatial shapes for deformable attention
            level_start_index (Tensor): Level start indices for deformable attention
            **kwargs: Additional arguments
            
        Returns:
            Tensor: Output with same shape as query
        """
        
        # Self-attention
        query = self.self_attn(
            query=query,
            key=None,  # Self-attention
            value=None,  # Self-attention 
            query_pos=query_pos,
            attn_mask=attn_masks[0] if attn_masks is not None else None,
            key_padding_mask=query_key_padding_mask
        )
        
        # Norm after self-attention
        query = self.norm1(query)
        
        # Cross-attention with BEV features
        if key is not None and value is not None:
            query = self.cross_attn(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
        else:
            # Skip cross-attention if no key/value provided
            # This allows testing decoder layer in isolation
            pass
        
        # Norm after cross-attention
        query = self.norm2(query)
        
        # FFN
        query = self.ffn(query)
        
        # Final norm
        query = self.norm3(query)
        
        return query


def test_detr_transformer_decoder_layer():
    """Test DetrTransformerDecoderLayer module"""
    print("=" * 60)
    print("Testing DetrTransformerDecoderLayer")
    print("=" * 60)
    
    # Config parameters from decoder config
    embed_dims = 256
    feedforward_channels = 1024
    
    try:
        # Create attention configs as per decoder config
        attn_cfgs = [
            {
                'type': 'MultiheadAttention',
                'embed_dims': embed_dims,
                'num_heads': 8,
                'dropout': 0.1
            },
            {
                'type': 'CustomMSDeformableAttention',
                'embed_dims': embed_dims,
                'num_levels': 1
            }
        ]
        
        # Create model
        model = DetrTransformerDecoderLayer(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            embed_dims=embed_dims,
            ffn_dropout=0.1
        )
        
        print(f"✓ Model created successfully")
        print(f"  - embed_dims: {embed_dims}")
        print(f"  - feedforward_channels: {feedforward_channels}")
        print(f"  - self_attn: MultiheadAttention")
        print(f"  - cross_attn: CustomMSDeformableAttention")
        
        # Test inputs for decoder layer
        batch_size = 2
        num_queries = 900  # Object queries
        
        # Object queries - shape: (num_queries, bs, embed_dims) 
        query = torch.randn(num_queries, batch_size, embed_dims)
        
        # BEV features from encoder for cross-attention
        bev_h, bev_w = 200, 200
        num_value = bev_h * bev_w
        key = torch.randn(num_value, batch_size, embed_dims)
        value = torch.randn(num_value, batch_size, embed_dims)
        
        # Query positional encoding
        query_pos = torch.randn(num_queries, batch_size, embed_dims)
        
        # Reference points for cross-attention (2D normalized coordinates)
        reference_points = torch.rand(batch_size, num_queries, 1, 2)  # Single level
        
        # Spatial shapes for single BEV level
        spatial_shapes = torch.tensor([[bev_h, bev_w]], dtype=torch.long)
        level_start_index = torch.tensor([0], dtype=torch.long)
        
        print(f"✓ Test inputs created")
        print(f"  - query shape: {query.shape}")
        print(f"  - key shape: {key.shape}")
        print(f"  - value shape: {value.shape}")
        print(f"  - reference_points shape: {reference_points.shape}")
        print(f"  - spatial_shapes: {spatial_shapes}")
        
        # Forward pass through complete decoder layer
        with torch.no_grad():
            output = model(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
        
        print(f"✓ Forward pass successful")
        print(f"  - output shape: {output.shape}")
        print(f"  - expected shape: {query.shape}")
        
        # Verify output shape
        assert output.shape == query.shape, f"Output shape {output.shape} != expected {query.shape}"
        
        # Verify output is not NaN or Inf
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
        
        # Test without key/value (should still work with self-attention)
        with torch.no_grad():
            output_no_cross = model(
                query=query,
                key=None,
                value=None,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
        
        print(f"✓ Forward pass without cross-attention successful")
        print(f"  - output shape: {output_no_cross.shape}")
        
        assert output_no_cross.shape == query.shape, f"Output shape {output_no_cross.shape} != expected {query.shape}"
        assert torch.isfinite(output_no_cross).all(), "Output contains NaN or Inf values"
        
        print("✓ All assertions passed")
        print("🎉 DetrTransformerDecoderLayer test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_detr_transformer_decoder_layer()