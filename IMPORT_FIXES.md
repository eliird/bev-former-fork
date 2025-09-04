# Import Compatibility Fixes for BEVFormer

This document lists all the import changes needed to make BEVFormer compatible with newer MMDetection/MMDetection3D versions.

## Import Mapping Table

| Old Import | New Import | Notes |
|------------|------------|-------|
| `from mmdet3d.datasets import build_dataset` | `from mmengine.registry import build_from_cfg, DATASETS` | Use `build_from_cfg(config, DATASETS)` |
| `from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps` | `from mmcv.ops import bbox_overlaps` | Moved to mmcv.ops |
| `from mmdet3d.core.bbox.box_np_ops import points_cam2img` | `from mmdet3d.datasets.convert_utils import points_cam2img` | Moved to convert_utils |
| `from mmdet3d.core.bbox import box_np_ops` | `from mmdet3d.datasets.convert_utils import box_np_ops` | Moved to convert_utils |
| `NuScenesDataset.NameMapping` | `NuScenesDataset.METAINFO['classes']` | NameMapping deprecated |
| `from mmcv import Config` | `from mmengine import Config` | Config moved to mmengine |
| `from mmdet.core.bbox.builder import BBOX_ASSIGNERS` | `from mmdet.registry import TASK_UTILS as BBOX_ASSIGNERS` | Registry centralized |
| `from mmdet.core.bbox.assigners import AssignResult` | `from mmdet.structures.bbox import AssignResult` | Moved to structures |
| `from mmdet.core.bbox.assigners import BaseAssigner` | `from mmdet.models.task_modules.assigners import BaseAssigner` | Moved to task_modules |
| `from mmdet.core.bbox.match_costs import build_match_cost` | `from mmengine.registry import build_from_cfg`<br>`from mmdet.registry import TASK_UTILS` | Use `build_from_cfg(config, TASK_UTILS)` |
| `from mmdet.models.utils.transformer import inverse_sigmoid` | `from mmdet.models.layers.transformer import inverse_sigmoid` | Moved to layers |
| `from mmdet.core.bbox import BaseBBoxCoder` | `from mmdet.models.task_modules.coders import BaseBBoxCoder` | Moved to task_modules |
| `from mmdet.core.bbox.builder import BBOX_CODERS` | `from mmdet.registry import TASK_UTILS as BBOX_CODERS` | Registry centralized |
| `from mmdet.core.bbox.match_costs.builder import MATCH_COST` | `from mmdet.registry import TASK_UTILS as MATCH_COST` | Registry centralized |

## Usage Pattern Changes

### Dataset Building
**Old:**
```python
from mmdet3d.datasets import build_dataset
dataset = build_dataset(config)
```

**New:**
```python
from mmengine.registry import build_from_cfg
from mmdet3d.registry import DATASETS
dataset = build_from_cfg(config, DATASETS)
```

### Match Cost Building
**Old:**
```python
from mmdet.core.bbox.match_costs import build_match_cost
cost = build_match_cost(config)
```

**New:**
```python
from mmengine.registry import build_from_cfg
from mmdet.registry import TASK_UTILS
cost = build_from_cfg(config, TASK_UTILS)
```

### Class Name Access
**Old:**
```python
if name in NuScenesDataset.NameMapping:
    mapped_name = NuScenesDataset.NameMapping[name]
```

**New:**
```python
if name in NuScenesDataset.METAINFO['classes']:
    mapped_name = name  # or appropriate mapping
```

## Key Changes Summary

1. **Registry System**: All builders now use centralized registries (`TASK_UTILS`, `DATASETS`) with `build_from_cfg()`
2. **Module Reorganization**: 
   - `mmdet.core.bbox.*` → `mmdet.models.task_modules.*` or `mmdet.structures.bbox.*`
   - `mmdet.models.utils.*` → `mmdet.models.layers.*`
3. **Configuration**: `mmcv.Config` → `mmengine.Config`
4. **Dataset Metadata**: `NameMapping` → `METAINFO['classes']`
5. **Operations**: Some ops moved to `mmcv.ops` or `convert_utils`

## Common Error Patterns

- `AttributeError: type object 'NuScenesDataset' has no attribute 'NameMapping'` → Use `METAINFO['classes']`
- `ImportError: cannot import name 'build_dataset'` → Use registry system
- `ImportError: cannot import name 'bbox_overlaps'` → Import from `mmcv.ops`
- `ImportError: No module named 'mmdet.core.bbox'` → Use `mmdet.models.task_modules` or `mmdet.structures.bbox`