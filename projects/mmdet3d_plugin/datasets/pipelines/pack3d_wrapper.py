"""
Wrapper to make Pack3DDetInputs compatible with CustomCollect3D output format.
"""

import torch
from mmdet3d.datasets.transforms import Pack3DDetInputs
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Pack3DDetInputsWrapper(Pack3DDetInputs):
    """
    Wrapper around Pack3DDetInputs to maintain compatibility with CustomCollect3D output.
    
    This wrapper:
    1. Uses Pack3DDetInputs for proper data packing
    2. Restructures output to match CustomCollect3D format
    3. Preserves BEVFormer-specific metadata
    """
    
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'lidar2cam',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'scene_token',
                            'can_bus')):
        # Initialize parent with standard keys
        super().__init__(keys=keys, meta_keys=meta_keys)
        self.collect_keys = keys
        self.collect_meta_keys = meta_keys
    
    def transform(self, results):
        """
        Override transform to match CustomCollect3D output format.
        """
        # First apply standard Pack3DDetInputs processing
        packed_results = super().transform(results)
        
        # Now restructure to match CustomCollect3D format
        data = {}
        img_metas = {}
        
        # Extract metadata
        for key in self.collect_meta_keys:
            if key in results:
                img_metas[key] = results[key]
        
        # Store metadata in the expected format
        data['img_metas'] = img_metas
        
        # Extract the main data keys
        for key in self.collect_keys:
            if key in results:
                data[key] = results[key]
            else:
                data[key] = None
        
        # Handle special cases from Pack3DDetInputs output
        if 'inputs' in packed_results:
            # If img is in inputs, use it
            if 'img' in packed_results['inputs'] and 'img' in self.collect_keys:
                data['img'] = packed_results['inputs']['img']
            # If points is in inputs, use it
            if 'points' in packed_results['inputs'] and 'points' in self.collect_keys:
                data['points'] = packed_results['inputs']['points']
        
        # Handle data_samples if present
        if 'data_samples' in packed_results:
            samples = packed_results['data_samples']
            # Extract gt_bboxes_3d and gt_labels_3d if present
            if hasattr(samples, 'gt_instances_3d'):
                if 'gt_bboxes_3d' in self.collect_keys:
                    data['gt_bboxes_3d'] = samples.gt_instances_3d.bboxes_3d
                if 'gt_labels_3d' in self.collect_keys:
                    data['gt_labels_3d'] = samples.gt_instances_3d.labels_3d
        
        return data


@TRANSFORMS.register_module()
class BevFormerCollect3D(object):
    """
    Direct replacement for CustomCollect3D using Pack3DDetInputs internally.
    Maintains exact same interface and behavior as CustomCollect3D.
    """
    
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'lidar2cam',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'scene_token',
                            'can_bus')):
        self.keys = keys
        self.meta_keys = meta_keys
        # Create internal Pack3DDetInputs
        self.packer = Pack3DDetInputs(keys=tuple(keys), meta_keys=tuple(meta_keys))
    
    def __call__(self, results):
        """
        Call function to collect keys in results.
        Maintains CustomCollect3D behavior while using Pack3DDetInputs internally.
        """
        # For compatibility, we don't use Pack3DDetInputs transform directly
        # Instead, we replicate CustomCollect3D behavior exactly
        data = {}
        img_metas = {}
        
        # Collect metadata
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        
        data['img_metas'] = img_metas
        
        # Collect data keys
        for key in self.keys:
            if key not in results:
                data[key] = None
            else:
                data[key] = results[key]
        
        return data
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'