# from mmdet.core.bbox.match_costs import build_match_cost
from .match_cost import BBox3DL1Cost, SmoothL1Cost
from mmengine.registry import build_from_cfg
from mmdet.registry import TASK_UTILS

def build_match_cost(cfg):
    """Build match cost from config."""
    return build_from_cfg(cfg, TASK_UTILS)

__all__ = ['build_match_cost', 'BBox3DL1Cost', 'SmoothL1Cost']