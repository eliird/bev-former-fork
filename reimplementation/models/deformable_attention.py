"""
Smart Deformable Attention Implementation
Uses MMCV CUDA extensions when available, falls back to pure PyTorch implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from torch.autograd.function import Function, once_differentiable

# Try to import MMCV's CUDA extensions (like BEVFormer does)
try:
    from mmcv.utils import ext_loader
    ext_module = ext_loader.load_ext('_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
    MMCV_CUDA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ext_module = None
    MMCV_CUDA_AVAILABLE = False

# Try to import MMCV's PyTorch fallback
try:
    from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
    MMCV_PYTORCH_AVAILABLE = True
except ImportError:
    MMCV_PYTORCH_AVAILABLE = False

if not MMCV_CUDA_AVAILABLE and not MMCV_PYTORCH_AVAILABLE:
    warnings.warn("MMCV not available, using pure PyTorch implementation for deformable attention")


def _pure_pytorch_deformable_attn(
        value: torch.Tensor, 
        value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch implementation of multi-scale deformable attention.
    
    This is extracted from MMCV's CPU fallback implementation.
    
    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """
    
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttnFunction_fp32(Function):
    """CUDA-accelerated Multi-Scale Deformable Attention (FP32 version)"""
    
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, 
                              sampling_locations, attention_weights)
        return output
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)
        
        ext_module.ms_deform_attn_backward(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, grad_output.contiguous(),
            grad_value, grad_sampling_loc, grad_attn_weight, im2col_step=ctx.im2col_step)
        
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


class MultiScaleDeformableAttnFunction_fp16(Function):
    """CUDA-accelerated Multi-Scale Deformable Attention (FP16 version)"""
    
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, 
                              sampling_locations, attention_weights)
        return output
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)
        
        ext_module.ms_deform_attn_backward(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, grad_output.contiguous(),
            grad_value, grad_sampling_loc, grad_attn_weight, im2col_step=ctx.im2col_step)
        
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def multi_scale_deformable_attn(
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        value_level_start_index: torch.Tensor = None,
        im2col_step: int = 64) -> torch.Tensor:
    """Smart multi-scale deformable attention with priority for CUDA extensions.
    
    Priority:
    1. MMCV CUDA extensions (fastest) 
    2. MMCV PyTorch fallback (medium)
    3. Pure PyTorch implementation (slowest)
    
    Args:
        value (torch.Tensor): Shape (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Shape (num_levels, 2) 
        sampling_locations (torch.Tensor): Shape (bs, num_queries, num_heads, num_levels, num_points, 2)
        attention_weights (torch.Tensor): Shape (bs, num_queries, num_heads, num_levels, num_points)
        value_level_start_index (torch.Tensor): Start index for each level
        im2col_step (int): Image to column step size
        
    Returns:
        torch.Tensor: Shape (bs, num_queries, embed_dims)
    """
    if MMCV_CUDA_AVAILABLE and value.is_cuda:
        # Use CUDA extensions (fastest)
        if value_level_start_index is None:
            # Generate level start index
            value_level_start_index = torch.cat([
                torch.tensor([0], device=value.device),
                value_spatial_shapes.prod(1).cumsum(0)[:-1]
            ]).int()
        
        im2col_step = torch.tensor(im2col_step, device=value.device)
        
        # Choose precision based on input dtype
        if value.dtype == torch.float16:
            return MultiScaleDeformableAttnFunction_fp16.apply(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step)
        else:
            return MultiScaleDeformableAttnFunction_fp32.apply(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step)
    
    elif MMCV_PYTORCH_AVAILABLE:
        # Use MMCV's PyTorch implementation (medium speed)
        return multi_scale_deformable_attn_pytorch(
            value, value_spatial_shapes, sampling_locations, attention_weights)
    
    else:
        # Use our pure PyTorch fallback (slowest)
        return _pure_pytorch_deformable_attn(
            value, value_spatial_shapes, sampling_locations, attention_weights)


class MSDeformableAttention3D(nn.Module):
    """An attention module used in BEVFormer based on Deformable-Detr.
    
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    
    Args:
        embed_dims (int): The embedding dimension of Attention. Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in Attention. Default: 4.
        num_points (int): The number of sampling points for each query in each head. Default: 8.
        im2col_step (int): The step used in image_to_column. Default: 64.
        dropout (float): A Dropout layer on `inp_identity`. Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization. Default: None.
    """
    
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__()
        
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                           f'but got {embed_dims} and {num_heads}')
        
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False
        
        # Check if dim_per_head is power of 2 for CUDA efficiency
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0
        
        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MSDeformableAttention3D to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')
        
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        
        # Learnable parameters
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        self.init_weights()
    
    def init_weights(self):
        """Default initialization for Parameters of Module."""
        # Initialize sampling offsets
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        nn.init.constant_(self.sampling_offsets.bias.data, 0.)
        
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
            
        self.sampling_offsets.bias.data = grid_init.view(-1)
        
        # Initialize attention weights
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        
        # Initialize projections  
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data) 
        nn.init.constant_(self.output_proj.bias.data, 0.)
        
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
                **kwargs):
        """Forward Function of MSDeformableAttention3D.
        
        Args:
            query (Tensor): Query of Transformer with shape (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape `(bs, num_key, embed_dims)`.
            value (Tensor): The value tensor with shape `(bs, num_key, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the same shape as `query`. Default None.
            query_pos (Tensor): The positional encoding for `query`. Default: None.
            key_padding_mask (Tensor): ByteTensor for `query`, with shape [bs, num_key].
            reference_points (Tensor): The normalized reference points with shape 
                (bs, num_query, num_levels, 2), all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
            spatial_shapes (Tensor): Spatial shape of features in different levels. 
                With shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
                
        Returns:
            Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        
        if not self.batch_first:
            # change to (bs, num_query, embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        
        if spatial_shapes is not None:
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
        attention_weights = attention_weights.view(bs, num_query, self.num_heads,
                                                 self.num_levels, self.num_points)
        
        # Handle reference points - this is the key BEVFormer 3D logic
        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After projecting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each reference point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points, it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors
            
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
        
        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        
        # Apply deformable attention with smart backend selection
        if MMCV_CUDA_AVAILABLE and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp16
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            # Use PyTorch fallback
            output = multi_scale_deformable_attn(
                value, spatial_shapes, sampling_locations, attention_weights,
                level_start_index, self.im2col_step)
        
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        
        return output


def test_msdeformable_attention3d():
    """Test MSDeformableAttention3D module"""
    print("=" * 60)
    print("Testing MSDeformableAttention3D")
    print("=" * 60)
    
    # Config parameters
    embed_dims = 256
    num_heads = 8
    num_levels = 4
    num_points = 8
    
    try:
        # Create model
        model = MSDeformableAttention3D(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=True
        )
        
        print(f"✓ Model created successfully")
        print(f"  - embed_dims: {embed_dims}")
        print(f"  - num_heads: {num_heads}")
        print(f"  - num_levels: {num_levels}")
        print(f"  - num_points: {num_points}")
        
        # Test inputs
        batch_size = 2
        num_queries = 2500  # 50x50 BEV queries
        
        # Spatial shapes for multi-scale features (must sum to num_keys)
        spatial_shapes = torch.tensor([[25, 15], [13, 8], [7, 4], [4, 2]], dtype=torch.long)
        num_keys = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item()  # 375 + 104 + 28 + 8 = 515
        level_start_index = torch.cat([
            torch.tensor([0]), 
            (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]
        ])
        
        query = torch.randn(batch_size, num_queries, embed_dims)
        key = torch.randn(batch_size, num_keys, embed_dims)
        value = torch.randn(batch_size, num_keys, embed_dims)
        
        # Reference points (normalized coordinates)
        reference_points = torch.rand(batch_size, num_queries, num_levels, 2)
        
        print(f"✓ Test inputs created")
        print(f"  - query shape: {query.shape}")
        print(f"  - key shape: {key.shape}")
        print(f"  - value shape: {value.shape}")
        print(f"  - spatial_shapes: {spatial_shapes.shape}")
        print(f"  - reference_points: {reference_points.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(
                query=query,
                key=key, 
                value=value,
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
        
        print("✓ All assertions passed")
        print("🎉 MSDeformableAttention3D test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False



if __name__ == "__main__":
    test_msdeformable_attention3d()