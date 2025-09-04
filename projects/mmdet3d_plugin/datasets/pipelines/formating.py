
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
# from mmcv.parallel import DataContainer as DC

# from mmdet3d.core.bbox import BaseInstance3DBoxes
# from mmdet3d.core.points import BasePoints
from mmdet3d.structures import BasePoints, BaseInstance3DBoxes
# from mmdet.datasets.builder import PIPELINES
from mmdet.registry import TRANSFORMS
# from mmdet.datasets.pipelines import to_tensor
from mmdet.datasets.transforms import ToTensor as to_tensor
# from mmdet3d.datasets.pipelines import DefaultFormatBundle3D
from mmdet3d.datasets.transforms import Pack3DDetInputs
import torch


@TRANSFORMS.register_module()
class CustomDefaultFormatBundle3D(Pack3DDetInputs):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(CustomDefaultFormatBundle3D, self).__call__(results)
        results['gt_map_masks'] = to_tensor(results['gt_map_masks'])
        if isinstance(gt_map_masks, list):
            gt_map_masks = torch.stack(gt_map_masks)
            results['gt_map_masks'] = gt_map_masks

        return results