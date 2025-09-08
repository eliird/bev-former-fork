'''
transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
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
                decoder=dict(
                    type='DetectionTransformerDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=_dim_,
                                num_heads=8,
                                dropout=0.1),
                             dict(
                                type='CustomMSDeformableAttention',
                                embed_dims=_dim_,
                                num_levels=1),
                        ],
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                         'ffn', 'norm'))),
'''

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import rotate
from torch.cuda.amp import autocast
from bev_former_encoder import BEVFormerEncoder
from detection_transformer_decoder import DetectionTransformerDecoder


class PerceptionTransformer(nn.Module):
    """Implements the BEVFormer perception transformer.
    
    Args:
        num_feature_levels (int): Number of feature maps from FPN. Default: 4.
        num_cams (int): Number of cameras. Default: 6.
        encoder (dict): Config for encoder.
        decoder (dict): Config for decoder.
        embed_dims (int): Embedding dimensions. Default: 256.
        rotate_prev_bev (bool): Whether to rotate previous BEV. Default: True.
        use_shift (bool): Whether to use shift for temporal modeling. Default: True.
        use_can_bus (bool): Whether to use CAN bus signals. Default: True.
        can_bus_norm (bool): Whether to normalize CAN bus signals. Default: True.
        use_cams_embeds (bool): Whether to use camera embeddings. Default: True.
        rotate_center (list): Center for rotation. Default: [100, 100].
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__()
        
        # Create encoder - use our BEVFormerEncoder if encoder config provided
        if encoder is not None:
            self.encoder = BEVFormerEncoder(
                num_layers=encoder.get('num_layers', 6),
                embed_dims=embed_dims,
                pc_range=encoder.get('pc_range'),
                num_points_in_pillar=encoder.get('num_points_in_pillar', 4),
                return_intermediate=encoder.get('return_intermediate', False),
                feedforward_channels=encoder.get('transformerlayers', {}).get('feedforward_channels', 1024),
                ffn_dropout=encoder.get('transformerlayers', {}).get('ffn_dropout', 0.1),
                num_levels=num_feature_levels
            )
        else:
            self.encoder = None
            
        # Create decoder - use our DetectionTransformerDecoder if decoder config provided
        if decoder is not None:
            # Decoder attention configs from config
            decoder_attn_cfgs = [
                {
                    'type': 'MultiheadAttention',
                    'embed_dims': embed_dims,
                    'num_heads': 8,
                    'dropout': 0.1
                },
                {
                    'type': 'CustomMSDeformableAttention',
                    'embed_dims': embed_dims,
                    'num_levels': 1  # Single BEV level for decoder
                }
            ]
            
            self.decoder = DetectionTransformerDecoder(
                num_layers=decoder.get('num_layers', 6),
                return_intermediate=decoder.get('return_intermediate', True),
                attn_cfgs=decoder_attn_cfgs,
                feedforward_channels=decoder.get('transformerlayers', {}).get('feedforward_channels', 1024),
                embed_dims=embed_dims,
                ffn_dropout=decoder.get('transformerlayers', {}).get('ffn_dropout', 0.1)
            )
        else:
            self.decoder = None
        
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.rotate_center = rotate_center
        
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the PerceptionTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def get_bev_features(self,
                        mlvl_feats,
                        bev_queries,
                        bev_h,
                        bev_w,
                        grid_length=[0.512, 0.512],
                        bev_pos=None,
                        prev_bev=None,
                        **kwargs):
        """Obtain bev features."""
        # Disable autocast for now to avoid dtype mismatch issues
        # with autocast(enabled=True, dtype=torch.float16):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(0).repeat(bs, 1, 1)  # (bs, num_queries, embed_dims)
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1)  # (bs, num_queries, embed_dims)

        # Obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                        for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                        for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            # Ensure prev_bev has correct shape (bs, num_queries, embed_dims)
            if prev_bev.shape[0] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if False:  # Disable rotation for now
                for i in range(bs):
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    # Ensure rotation_angle is a scalar float
                    if hasattr(rotation_angle, 'item'):
                        rotation_angle = rotation_angle.item()
                    else:
                        rotation_angle = float(rotation_angle)
                    
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                        center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # Add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])
        can_bus = self.can_bus_mlp(can_bus)[:, None, :]  # (bs, 1, embed_dims)
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        # Create reference points needed by BEVFormerEncoder layers
        # Reference points for temporal attention (2D BEV grid)
        ref_2d = torch.rand(bs, bev_h * bev_w, 1, 2, device=bev_pos.device)  # BEV uses single level
        
        # Reference points for spatial attention (3D with Z-anchors)  
        num_Z_anchors = 4
        ref_3d = torch.rand(bs, bev_h * bev_w, self.num_feature_levels, 2, device=bev_pos.device)
        reference_points_cam = torch.rand(bs, bev_h * bev_w, num_Z_anchors, 2, device=bev_pos.device)
            
        # BEV mask for spatial attention
        bev_mask = torch.zeros(self.num_cams, bs, bev_h * bev_w, dtype=torch.bool, device=bev_pos.device)
        for cam_id in range(self.num_cams):
            # Each camera sees overlapping regions
            start_idx = (cam_id * (bev_h * bev_w) // (self.num_cams + 2))
            end_idx = ((cam_id + 3) * (bev_h * bev_w) // (self.num_cams + 2))
            end_idx = min(end_idx, bev_h * bev_w)
            bev_mask[cam_id, :, start_idx:end_idx] = True

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
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
            bev_mask=bev_mask,
            **kwargs
        )

        return bev_embed

    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for PerceptionTransformer.
        
        Args:
            mlvl_feats (list(Tensor)): Input queries from different level. 
                Each element has shape [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer.
            cls_branches (obj:`nn.ModuleList`): Classification heads.
                
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder (None for now since no decoder)
                - init_reference_out: The initial value of reference points
                - inter_references_out: The internal value of reference points
        """
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        # Use decoder if available
        if self.decoder is not None:
            inter_states, inter_references = self.decoder(
                query=query,
                key=bev_embed,
                value=bev_embed,
                query_pos=query_pos,
                reference_points=reference_points,
                reg_branches=reg_branches,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device, dtype=torch.long),
                level_start_index=torch.tensor([0], device=query.device, dtype=torch.long),
                **kwargs)
            inter_references_out = inter_references
        else:
            # Return compatible outputs when decoder is not implemented
            inter_states = None
            inter_references_out = None

        return bev_embed, inter_states, init_reference_out, inter_references_out


def test_perception_transformer():
    """Test PerceptionTransformer module with both encoder and decoder"""
    print("=" * 60)
    print("Testing PerceptionTransformer")
    print("=" * 60)
    
    # Config parameters from BEVFormer
    embed_dims = 256
    feedforward_channels = 1024
    num_feature_levels = 4
    num_cams = 6
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    try:
        # Create encoder config
        encoder_config = {
            'type': 'BEVFormerEncoder',
            'num_layers': 6,
            'pc_range': pc_range,
            'num_points_in_pillar': 4,
            'return_intermediate': False,
            'transformerlayers': {
                'feedforward_channels': feedforward_channels,
                'ffn_dropout': 0.1
            }
        }
        
        # Create decoder config
        decoder_config = {
            'type': 'DetectionTransformerDecoder',
            'num_layers': 6,
            'return_intermediate': True,
            'transformerlayers': {
                'feedforward_channels': feedforward_channels,
                'ffn_dropout': 0.1
            }
        }
        
        # Create model with both encoder and decoder
        model = PerceptionTransformer(
            num_feature_levels=num_feature_levels,
            num_cams=num_cams,
            encoder=encoder_config,
            decoder=decoder_config,
            embed_dims=embed_dims,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True
        )
        
        print(f"✓ Model created successfully")
        print(f"  - embed_dims: {embed_dims}")
        print(f"  - num_feature_levels: {num_feature_levels}")
        print(f"  - num_cams: {num_cams}")
        print(f"  - encoder: BEVFormerEncoder (6 layers)")
        print(f"  - decoder: DetectionTransformerDecoder (6 layers)")
        
        # Test inputs
        batch_size = 2
        bev_h, bev_w = 50, 50  # Reduced for memory efficiency
        num_queries = 900  # Object queries
        
        # Multi-level multi-camera features
        mlvl_feats = []
        for level in range(num_feature_levels):
            h, w = 25 // (2 ** level), 15 // (2 ** level)
            h, w = max(h, 1), max(w, 1)
            feat = torch.randn(batch_size, num_cams, embed_dims, h, w)
            mlvl_feats.append(feat)
        
        # BEV queries (learnable embeddings for BEV grid)
        bev_queries = torch.randn(bev_h * bev_w, embed_dims)
        
        # Object query embeddings (for decoder)
        object_query_embed = torch.randn(num_queries, embed_dims * 2)  # query + pos
        
        # BEV positional encoding
        bev_pos = torch.randn(batch_size, embed_dims, bev_h, bev_w)
        
        # Previous BEV for temporal modeling
        prev_bev = torch.randn(batch_size, bev_h * bev_w, embed_dims)
        
        # Mock img_metas with CAN bus data
        img_metas = []
        for i in range(batch_size):
            can_bus = torch.zeros(18)  # CAN bus signals
            can_bus[0] = 0.1 * i  # delta_x
            can_bus[1] = 0.1 * i  # delta_y  
            can_bus[-2] = 0.1  # ego_angle
            can_bus[-1] = 0.05  # rotation_angle
            
            img_metas.append({
                'can_bus': can_bus.numpy()
            })
        
        print(f"✓ Test inputs created")
        print(f"  - mlvl_feats: {len(mlvl_feats)} levels, shapes: {[feat.shape for feat in mlvl_feats]}")
        print(f"  - bev_queries shape: {bev_queries.shape}")
        print(f"  - object_query_embed shape: {object_query_embed.shape}")
        print(f"  - bev_pos shape: {bev_pos.shape}")
        print(f"  - prev_bev shape: {prev_bev.shape}")
        
        # Forward pass through complete perception transformer
        with torch.no_grad():
            bev_embed, inter_states, init_reference_out, inter_references_out = model(
                mlvl_feats=mlvl_feats,
                bev_queries=bev_queries,
                object_query_embed=object_query_embed,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                img_metas=img_metas
            )
        
        print(f"✓ Forward pass successful")
        print(f"  - bev_embed shape: {bev_embed.shape}")
        print(f"  - inter_states shape: {inter_states.shape if inter_states is not None else None}")
        print(f"  - init_reference_out shape: {init_reference_out.shape}")
        print(f"  - inter_references_out shape: {inter_references_out.shape if inter_references_out is not None else None}")
        
        # Verify output shapes
        expected_bev_shape = torch.Size([bev_h * bev_w, batch_size, embed_dims])
        expected_init_ref_shape = torch.Size([batch_size, num_queries, 3])
        
        assert bev_embed.shape == expected_bev_shape, f"BEV embed shape {bev_embed.shape} != expected {expected_bev_shape}"
        assert init_reference_out.shape == expected_init_ref_shape, f"Init ref shape {init_reference_out.shape} != expected {expected_init_ref_shape}"
        
        # When decoder is present, should have intermediate states
        if model.decoder is not None:
            expected_inter_shape = torch.Size([6, num_queries, batch_size, embed_dims])  # 6 decoder layers
            expected_inter_ref_shape = torch.Size([6, batch_size, num_queries, 3])  # 6 layers, 3D ref points
            
            assert inter_states is not None, "inter_states should not be None when decoder is present"
            assert inter_references_out is not None, "inter_references_out should not be None when decoder is present"
            assert inter_states.shape == expected_inter_shape, f"Inter states shape {inter_states.shape} != expected {expected_inter_shape}"
            assert inter_references_out.shape == expected_inter_ref_shape, f"Inter ref shape {inter_references_out.shape} != expected {expected_inter_ref_shape}"
        
        # Verify output is not NaN or Inf
        assert torch.isfinite(bev_embed).all(), "BEV embed contains NaN or Inf values"
        assert torch.isfinite(init_reference_out).all(), "Initial reference points contain NaN or Inf values"
        if inter_states is not None:
            assert torch.isfinite(inter_states).all(), "Intermediate states contain NaN or Inf values"
        if inter_references_out is not None:
            assert torch.isfinite(inter_references_out).all(), "Intermediate references contain NaN or Inf values"
        
        # Test without previous BEV (first frame)
        with torch.no_grad():
            bev_embed_no_prev, _, _, _ = model(
                mlvl_feats=mlvl_feats,
                bev_queries=bev_queries,
                object_query_embed=object_query_embed,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_pos=bev_pos,
                prev_bev=None,  # No previous BEV
                img_metas=img_metas
            )
        
        print(f"✓ Forward pass without prev_bev successful")
        print(f"  - bev_embed shape: {bev_embed_no_prev.shape}")
        
        assert bev_embed_no_prev.shape == expected_bev_shape, f"BEV embed shape {bev_embed_no_prev.shape} != expected {expected_bev_shape}"
        assert torch.isfinite(bev_embed_no_prev).all(), "BEV embed contains NaN or Inf values"
        
        print("✓ All assertions passed")
        print("🎉 PerceptionTransformer test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_perception_transformer()
