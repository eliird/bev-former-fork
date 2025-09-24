import torch.nn as nn
import torch
from .deformable_attention import MSDeformableAttention3D

class SpatialCrossAttention(nn.Module):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """
    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 batch_first=False,
                 deformable_attention: MSDeformableAttention3D=MSDeformableAttention3D(embed_dims=256, num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = deformable_attention
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask: torch.Tensor=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos
        bs, num_query, _ = query.size()

        num_Z_anchors = reference_points_cam.size(2)  # Number of Z anchors (height levels)
        coord_dim = reference_points_cam.size(3)    # Coordinate dimension (2 for x,y)
        
        indexes = []
        for i in range(len(bev_mask)):
            # Get mask for camera i, first batch
            mask_per_img = bev_mask[i]
            # The original code expects to find which queries are valid for each camera
            # mask_per_img[0] should give us a 1D tensor of shape [num_queries]
            # We want indices of True values (valid queries)
            index_query_per_img = mask_per_img[0].nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, num_Z_anchors, coord_dim])
        
        for j in range(bs):
            for i in range(self.num_cams):  
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_cam[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, num_Z_anchors, coord_dim), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        # Count valid queries per batch and query position
        # bev_mask: [num_cams, bs, num_queries] -> count: [bs, num_queries]
        count = bev_mask.sum(0)  # Sum over camera dimension -> [bs, num_queries]
        count = torch.clamp(count.float(), min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


def test_spatial_cross_attention():
    """Test SpatialCrossAttention module"""
    print("=" * 60)
    print("Testing SpatialCrossAttention")
    print("=" * 60)
    
    # Config parameters
    embed_dims = 256
    num_cams = 6
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    try:
        # Create model
        from deformable_attention import MSDeformableAttention3D
        deformable_attention = MSDeformableAttention3D(
            embed_dims=embed_dims,
            num_levels=4,
            num_points=8
        )
        
        model = SpatialCrossAttention(
            embed_dims=embed_dims,
            num_cams=num_cams,
            pc_range=pc_range,
            deformable_attention=deformable_attention,
            batch_first=True
        )
        
        print(f"✓ Model created successfully")
        print(f"  - embed_dims: {embed_dims}")
        print(f"  - num_cams: {num_cams}")
        print(f"  - pc_range: {pc_range}")
        
        # Test inputs
        batch_size = 2
        bev_h, bev_w = 50, 50
        num_queries = bev_h * bev_w  # 2500 BEV queries
        
        # BEV query (current BEV features)
        query = torch.randn(batch_size, num_queries, embed_dims)
        
        # Multi-camera image features
        img_h, img_w = 25, 15  # Feature map size after backbone+neck
        num_levels = 4
        
        # Multi-scale image features [bs, num_cams, embed_dims, h, w]
        key_list = []
        value_list = []
        spatial_shapes_list = []
        
        for level in range(num_levels):
            h, w = img_h // (2 ** level), img_w // (2 ** level)
            h, w = max(h, 1), max(w, 1)  # Ensure minimum size
            
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
        
        # Reference points in camera coordinates [bs, num_queries, num_Z_anchors, 2]
        num_Z_anchors = 4  # BEVFormer uses multiple height anchors
        reference_points_cam = torch.rand(batch_size, num_queries, num_Z_anchors, 2)
        
        # BEV mask indicating which queries are valid for each camera
        # The forward code does: mask_per_img[0].sum(-1).nonzero().squeeze(-1)
        # This suggests each mask_per_img should be [batch_size, num_queries] or similar
        # Let's create masks where each camera sees a subset of queries to avoid empty results
        
        bev_mask = torch.zeros(num_cams, batch_size, num_queries, dtype=torch.bool)
        for cam_id in range(num_cams):
            # Create overlapping regions so no camera has zero queries
            start_idx = (cam_id * num_queries // (num_cams + 2)) 
            end_idx = ((cam_id + 3) * num_queries // (num_cams + 2))
            end_idx = min(end_idx, num_queries)
            
            bev_mask[cam_id, :, start_idx:end_idx] = True
        
        # Convert to the format expected by the forward method
        
        print(f"✓ Test inputs created")
        print(f"  - query shape: {query.shape}")
        print(f"  - key shape: {key.shape}")
        print(f"  - value shape: {value.shape}")
        print(f"  - spatial_shapes: {spatial_shapes}")
        print(f"  - reference_points_cam: {reference_points_cam.shape}")
        print(f"  - bev_mask: {bev_mask.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(
                query=query,
                key=key,
                value=value,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
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
        print("🎉 SpatialCrossAttention test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_spatial_cross_attention()
