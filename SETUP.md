# BEVFormer Environment Setup and Data Preparation Guide

This document provides step-by-step instructions for setting up the BEVFormer environment and preparing the nuScenes dataset.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Verification](#verification)
4. [Troubleshooting](#troubleshooting)
5. [Version Compatibility](#version-compatibility)

## Environment Setup

### Prerequisites
- NVIDIA GPU with CUDA support
- Miniconda/Anaconda installed
- Minimum 16GB RAM recommended
- At least 50GB free disk space

### Step 1: Create Conda Environment
```bash
conda create -n open-mmlab python=3.10 -y
conda activate open-mmlab
```

### Step 2: Install PyTorch
```bash
# For CUDA 12.6 (adjust according to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Step 3: Install OpenMMLab Ecosystem
```bash
# Install OpenMIM for easy package management
pip install -U openmim

# Install MMCV (compatible version)
mim install mmcv==2.1.0

# Install MMDetection and MMSegmentation
mim install mmdet
mim install mmsegmentation

# Install MMDetection3D
mim install mmdet3d
```

### Step 4: Install Additional Dependencies
```bash
# Core dependencies
pip install einops fvcore seaborn iopath timm typing-extensions pylint ipython numpy matplotlib numba pandas scikit-image setuptools

# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Step 5: Download Pretrained Models
```bash
mkdir -p ckpts
cd ckpts
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
cd ..
```

## Data Preparation

### Step 1: Download nuScenes Dataset
1. Go to [nuScenes website](https://www.nuscenes.org/download)
2. Register and download:
   - **Mini split (v1.0-mini)**: `v1.0-mini.tgz` (~4GB)
   - **CAN bus expansion**: `can_bus.zip` (~10MB)

### Step 2: Extract Dataset
```bash
# Create data directory
mkdir -p data

# Extract mini dataset
tar -xzf v1.0-mini.tgz -C data/

# Extract CAN bus data
unzip can_bus.zip -d data/
```

### Step 3: Verify Data Structure
Your data directory should look like:
```
data/
├── can_bus/
│   ├── scene-0001_meta.json
│   ├── scene-0001_ms_imu.json
│   └── ... (CAN bus files for all scenes)
└── nuscenes/
    ├── maps/
    ├── samples/
    ├── sweeps/
    ├── v1.0-mini/
    └── LICENSE
```

### Step 4: Fix Import Compatibility Issues
The newer MM libraries require some import fixes. Update these files:

**File: `tools/data_converter/create_gt_database.py`**
- Line 15: Change `from mmdet3d.datasets import build_dataset` to `from mmengine.registry import build_from_cfg, DATASETS`
- Line 17: Change `from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps` to `from mmcv.ops import bbox_overlaps`
- Line 225: Change `build_from_cfg(dataset_cfg, registry=DATASETS)` to `dataset = build_from_cfg(dataset_cfg, registry=DATASETS)`

**File: `tools/data_converter/nuscenes_converter.py`**
- Line 16: Change `from mmdet3d.core.bbox.box_np_ops import points_cam2img` to `from mmdet3d.datasets.convert_utils import points_cam2img`
- Line 298: Change `if names[i] in NuScenesDataset.NameMapping:` to `if names[i] in NuScenesDataset.METAINFO['classes']:`

**File: `tools/data_converter/kitti_converter.py`**
- Line 7: Change `from mmdet3d.core.bbox import box_np_ops` to `from mmdet3d.datasets.convert_utils import box_np_ops`

### Step 5: Generate Dataset Info Files
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini --canbus ./data
```

This creates:
- `data/nuscenes/nuscenes_infos_temporal_train.pkl`
- `data/nuscenes/nuscenes_infos_temporal_val.pkl`

## Verification

### Test Environment Setup
```bash
python test_data_loading.py
```

This script will:
- ✅ Verify all packages are installed correctly
- ✅ Test configuration file loading
- ✅ Check dataset directory structure
- ✅ Test data loading pipeline

### Expected Output
```
BEVFormer Data Loading Test
============================================================
Testing basic imports...
✓ MMCV version: 2.1.0
✓ MMDet version: 3.3.0
✓ MMDet3D version: 1.4.0
✓ PyTorch version: 2.5.0
✓ CUDA available: True
...
Overall: 4/4 tests passed
🎉 All tests passed! Your environment is ready.
```

### Test Training (Dry Run)
```bash
# Test with tiny config (lowest memory requirements)
python tools/train.py projects/configs/bevformer/bevformer_tiny.py --work-dir ./work_dirs/test --cfg-options train_dataloader.dataset.ann_file=./data/nuscenes/nuscenes_infos_temporal_train.pkl
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```
ImportError: cannot import name 'build_dataset' from 'mmdet3d.datasets'
```
- **Solution**: Update imports as described in Step 4 above

**2. MMCV Version Incompatibility**
```
AssertionError: MMCV==2.2.0 is used but incompatible
```
- **Solution**: Install specific compatible version: `mim install mmcv==2.1.0`

**3. Numpy Compatibility Issues**
```
ImportError: numpy.core.multiarray failed to import
```
- **Solution**: Reinstall compatible numpy: `pip install numpy==1.24.3`

**4. CUDA Out of Memory**
- **Solution**: Use smaller configs (tiny/small) or reduce batch size
- **Alternative**: Use FP16 training: `projects/configs/bevformer_fp16/bevformer_tiny_fp16.py`

**5. Dataset Files Not Found**
```
FileNotFoundError: data/nuscenes/nuscenes_infos_temporal_train.pkl
```
- **Solution**: Run the data preparation script (Step 5 above)

### Memory Requirements

| Model Variant | Memory Usage | Suitable For |
|---------------|--------------|-------------|
| bevformer_tiny | ~6.5GB | Testing, debugging |
| bevformer_small | ~10.5GB | Small GPUs |
| bevformer_base | ~28.5GB | High-end GPUs |
| bevformer_tiny_fp16 | ~4GB | Memory-constrained setups |

### Performance Tips

1. **Use SSD storage** for faster data loading
2. **Increase num_workers** in data loaders if CPU allows
3. **Use FP16 training** to reduce memory usage
4. **Pin memory** for faster GPU transfers

## Version Compatibility

### Tested Versions
- Python: 3.10
- PyTorch: 2.5.0
- MMCV: 2.1.0
- MMDetection: 3.3.0
- MMDetection3D: 1.4.0
- MMSegmentation: 1.2.2

### Known Working Combinations
- CUDA 12.6 + PyTorch 2.5.0 + MMCV 2.1.0 ✅
- CUDA 11.8 + PyTorch 2.0.0 + MMCV 2.0.1 ✅

### Avoid These Combinations
- MMCV 2.2.0 with MMDet 3.3.0 ❌
- PyTorch < 1.9 with any MM library ❌
- Mixed MM library versions from different releases ❌

## Next Steps

Once setup is complete, you can:
1. **Train a model**: Follow training instructions in `docs/getting_started.md`
2. **Run inference**: Use `tools/test.py` for evaluation
3. **Visualize results**: Use `tools/analysis_tools/visual.py`
4. **Experiment**: Modify configs in `projects/configs/bevformer/`

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Verify your environment with `test_data_loading.py`
3. Check the original [BEVFormer repository](https://github.com/fundamentalvision/BEVFormer)
4. Review MMDetection3D documentation