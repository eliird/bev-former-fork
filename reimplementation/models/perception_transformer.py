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
'''

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_, xavier_uniform_
from torch.nn.init import xavier_init
from torchvision.transforms.functional import rotate
from torch.cuda.amp import autocast
from reimplementation.models.bev_former_encoder import BEVFormerEncoder


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
            
        # For now, decoder is not implemented in our reimplementation
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
        with autocast(enabled=True, dtype=torch.float16):
            bs = mlvl_feats[0].size(0)
            bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
            bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

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
                if prev_bev.shape[1] == bev_h * bev_w:
                    prev_bev = prev_bev.permute(1, 0, 2)
                if self.rotate_prev_bev:
                    for i in range(bs):
                        rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
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
            can_bus = self.can_bus_mlp(can_bus)[None, :, :]
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

        # For now, we don't have decoder implemented, so return None for decoder outputs
        if self.decoder is not None:
            inter_states, inter_references = self.decoder(
                query=query,
                key=None,
                value=bev_embed,
                query_pos=query_pos,
                reference_points=reference_points,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                level_start_index=torch.tensor([0], device=query.device),
                **kwargs)
            inter_references_out = inter_references
        else:
            # Return compatible outputs when decoder is not implemented
            inter_states = None
            inter_references_out = None

        return bev_embed, inter_states, init_reference_out, inter_references_out