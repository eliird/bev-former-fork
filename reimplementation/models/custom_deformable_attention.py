import math
import torch
import torch.nn as nn
import warnings
from deformable_attention import (
    MultiScaleDeformableAttnFunction_fp16, 
    MultiScaleDeformableAttnFunction_fp32, 
    multi_scale_deformable_attn, 
    _pure_pytorch_deformable_attn,
    MMCV_CUDA_AVAILABLE,
    MMCV_PYTORCH_AVAILABLE
)


class CustomMSDeformableAttention(nn.Module):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if MMCV_CUDA_AVAILABLE and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp16
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        elif MMCV_PYTORCH_AVAILABLE:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        else:
            output = _pure_pytorch_deformable_attn(
                value, spatial_shapes, sampling_locations, attention_weights
            )
        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


def test_custom_msdeformable_attention():
    """Test CustomMSDeformableAttention module"""
    print("=" * 60)
    print("Testing CustomMSDeformableAttention")
    print("=" * 60)
    
    # Config parameters for decoder
    embed_dims = 256
    num_heads = 8
    num_levels = 1  # Decoder typically uses single level BEV
    num_points = 4
    
    try:
        # Create model
        model = CustomMSDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=False  # Decoder uses (num_query, bs, embed_dims)
        )
        
        print(f"✓ Model created successfully")
        print(f"  - embed_dims: {embed_dims}")
        print(f"  - num_heads: {num_heads}")
        print(f"  - num_levels: {num_levels}")
        print(f"  - num_points: {num_points}")
        print(f"  - batch_first: False")
        
        # Test inputs for decoder attention
        batch_size = 2
        num_queries = 900  # Typical number of object queries
        
        # Query tensor (object queries) - shape: (num_query, bs, embed_dims)
        query = torch.randn(num_queries, batch_size, embed_dims)
        
        # Single level BEV features for decoder cross-attention
        bev_h, bev_w = 200, 200  # Full BEV resolution
        num_value = bev_h * bev_w  # 40000 BEV positions
        
        # Key and Value from encoder (BEV features)
        key = torch.randn(num_value, batch_size, embed_dims)
        value = torch.randn(num_value, batch_size, embed_dims)
        
        # Spatial shapes for single BEV level
        spatial_shapes = torch.tensor([[bev_h, bev_w]], dtype=torch.long)
        level_start_index = torch.tensor([0], dtype=torch.long)
        
        # Reference points for object queries (2D normalized coordinates)
        reference_points = torch.rand(batch_size, num_queries, num_levels, 2)
        
        # Query positional encoding
        query_pos = torch.randn(num_queries, batch_size, embed_dims)
        
        print(f"✓ Test inputs created")
        print(f"  - query shape: {query.shape}")
        print(f"  - key shape: {key.shape}")
        print(f"  - value shape: {value.shape}")
        print(f"  - spatial_shapes: {spatial_shapes}")
        print(f"  - reference_points shape: {reference_points.shape}")
        
        # Forward pass
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
        
        # Test without query_pos
        with torch.no_grad():
            output_no_pos = model(
                query=query,
                key=key,
                value=value,
                query_pos=None,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
        
        print(f"✓ Forward pass without query_pos successful")
        print(f"  - output shape: {output_no_pos.shape}")
        
        assert output_no_pos.shape == query.shape, f"Output shape {output_no_pos.shape} != expected {query.shape}"
        assert torch.isfinite(output_no_pos).all(), "Output contains NaN or Inf values"
        
        # Test with 4D reference points (with width/height)
        reference_points_4d = torch.rand(batch_size, num_queries, num_levels, 4)
        
        with torch.no_grad():
            output_4d = model(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_4d,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
        
        print(f"✓ Forward pass with 4D reference points successful")
        print(f"  - output shape: {output_4d.shape}")
        
        assert output_4d.shape == query.shape, f"Output shape {output_4d.shape} != expected {query.shape}"
        assert torch.isfinite(output_4d).all(), "Output contains NaN or Inf values"
        
        print("✓ All assertions passed")
        print("🎉 CustomMSDeformableAttention test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_custom_msdeformable_attention()