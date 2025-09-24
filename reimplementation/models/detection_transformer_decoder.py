import torch
import torch.nn as nn
from .detr_decoder_layer import DetrTransformerDecoderLayer


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, 
                has same shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class DetectionTransformerDecoder(nn.Module):
    """Detection Transformer Decoder.
    
    Based on the config:
    decoder=dict(
        type='DetectionTransformerDecoder',
        num_layers=6,
        return_intermediate=True,
        transformerlayers=dict(
            type='DetrTransformerDecoderLayer',
            attn_cfgs=[...],
            feedforward_channels=_ffn_dim_,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        )
    )
    
    Args:
        num_layers (int): Number of decoder layers. Default: 6
        return_intermediate (bool): Whether to return intermediate outputs. Default: False
        attn_cfgs (list[dict]): Attention configs for decoder layers
        feedforward_channels (int): FFN hidden dimension
        embed_dims (int): Embedding dimension
        ffn_dropout (float): FFN dropout rate
        **kwargs: Additional arguments
    """
    
    def __init__(self,
                 num_layers=6,
                 return_intermediate=False,
                 attn_cfgs=None,
                 feedforward_channels=1024,
                 embed_dims=256,
                 ffn_dropout=0.1,
                 **kwargs):
        super(DetectionTransformerDecoder, self).__init__()
        
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.embed_dims = embed_dims
        
        # Build decoder layers
        self.layers = nn.ModuleList([
            DetrTransformerDecoderLayer(
                attn_cfgs=attn_cfgs,
                feedforward_channels=feedforward_channels,
                embed_dims=embed_dims,
                ffn_dropout=ffn_dropout,
                **kwargs
            ) for _ in range(num_layers)
        ])
        
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
                reg_branches=None,
                **kwargs):
        """Forward function for DetectionTransformerDecoder.
        
        Args:
            query (Tensor): Input query with shape (num_query, bs, embed_dims)
            key (Tensor): BEV features for cross-attention
            value (Tensor): BEV features for cross-attention
            query_pos (Tensor): Query positional encoding
            key_pos (Tensor): Key positional encoding
            reference_points (Tensor): The reference points of offset.
                                     has shape (bs, num_query, 4) when as_two_stage,
                                     otherwise has shape (bs, num_query, 2)
            spatial_shapes (Tensor): Spatial shapes for deformable attention
            level_start_index (Tensor): Level start indices for deformable attention
            reg_branches (nn.ModuleList): Used for refining the regression results.
                                        Only would be passed when with_box_refine is True
            **kwargs: Additional arguments
            
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                   return_intermediate is False, otherwise it has shape
                   [num_layers, num_query, bs, embed_dims].
            Tensor: Reference points with refinement if reg_branches provided
        """
        
        output = query
        intermediate = []
        intermediate_reference_points = []
        
        for lid, layer in enumerate(self.layers):
            # Prepare reference points for this layer
            # Extract 2D coordinates and add num_levels dimension
            reference_points_input = reference_points[..., :2].unsqueeze(2)  # BS NUM_QUERY NUM_LEVEL 2
            
            # Forward through decoder layer
            output = layer(
                query=output,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs
            )
            
            # Change from (num_query, bs, embed_dims) to (bs, num_query, embed_dims)
            # for regression branch processing
            output = output.permute(1, 0, 2)
            
            # Apply regression branch if provided (for iterative refinement)
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                
                assert reference_points.shape[-1] == 3, "Reference points should have 3D coordinates for regression"
                
                new_reference_points = torch.zeros_like(reference_points)
                # Update x, y coordinates
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                # Update z coordinate  
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            
            # Change back to (num_query, bs, embed_dims)
            output = output.permute(1, 0, 2)
            
            # Store intermediate results if needed
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        
        return output, reference_points


def test_detection_transformer_decoder():
    """Test DetectionTransformerDecoder module"""
    print("=" * 60)
    print("Testing DetectionTransformerDecoder")
    print("=" * 60)
    
    # Config parameters from decoder config
    embed_dims = 256
    feedforward_channels = 1024
    num_layers = 6
    
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
        
        # Test without return_intermediate first
        model = DetectionTransformerDecoder(
            num_layers=num_layers,
            return_intermediate=False,
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            embed_dims=embed_dims,
            ffn_dropout=0.1
        )
        
        print(f"✓ Model created successfully")
        print(f"  - num_layers: {num_layers}")
        print(f"  - embed_dims: {embed_dims}")
        print(f"  - feedforward_channels: {feedforward_channels}")
        print(f"  - return_intermediate: False")
        
        # Test inputs for decoder
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
        
        # Reference points for decoder (2D normalized coordinates)
        reference_points = torch.rand(batch_size, num_queries, 2)  # x, y coordinates
        
        # Spatial shapes for single BEV level
        spatial_shapes = torch.tensor([[bev_h, bev_w]], dtype=torch.long)
        level_start_index = torch.tensor([0], dtype=torch.long)
        
        print(f"✓ Test inputs created")
        print(f"  - query shape: {query.shape}")
        print(f"  - key shape: {key.shape}")
        print(f"  - value shape: {value.shape}")
        print(f"  - reference_points shape: {reference_points.shape}")
        print(f"  - spatial_shapes: {spatial_shapes}")
        
        # Forward pass through complete decoder
        with torch.no_grad():
            output, ref_points = model(
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
        print(f"  - ref_points shape: {ref_points.shape}")
        
        # Verify output shape
        assert output.shape == query.shape, f"Output shape {output.shape} != expected {query.shape}"
        assert ref_points.shape == reference_points.shape, f"Ref points shape {ref_points.shape} != expected {reference_points.shape}"
        
        # Verify output is not NaN or Inf
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
        assert torch.isfinite(ref_points).all(), "Reference points contain NaN or Inf values"
        
        # Test with return_intermediate=True
        model.return_intermediate = True
        with torch.no_grad():
            intermediate_outputs, intermediate_ref_points = model(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
        
        print(f"✓ Forward pass with return_intermediate=True successful")
        print(f"  - intermediate outputs shape: {intermediate_outputs.shape}")
        print(f"  - expected shape: [{num_layers}, {num_queries}, {batch_size}, {embed_dims}]")
        print(f"  - intermediate ref_points shape: {intermediate_ref_points.shape}")
        
        expected_intermediate_shape = torch.Size([num_layers, num_queries, batch_size, embed_dims])
        expected_ref_shape = torch.Size([num_layers, batch_size, num_queries, 2])
        
        assert intermediate_outputs.shape == expected_intermediate_shape, \
            f"Intermediate shape {intermediate_outputs.shape} != expected {expected_intermediate_shape}"
        assert intermediate_ref_points.shape == expected_ref_shape, \
            f"Intermediate ref shape {intermediate_ref_points.shape} != expected {expected_ref_shape}"
        
        # Test without cross-attention (key=value=None)
        with torch.no_grad():
            output_no_cross, _ = model(
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
        
        assert output_no_cross.shape == expected_intermediate_shape, \
            f"Output shape {output_no_cross.shape} != expected {expected_intermediate_shape}"
        assert torch.isfinite(output_no_cross).all(), "Output contains NaN or Inf values"
        
        print("✓ All assertions passed")
        print("🎉 DetectionTransformerDecoder test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_detection_transformer_decoder()
