# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation



**a. Create a conda virtual environment and activate it.**
```shell
conda create -n open-mmlab python=3.10 -y
conda activate open-mmlab
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Recommended torch>=1.9

```

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
# we only need mmcv for some of the custom kernels for speedup will work without it as well
pip install mim
mim install mmengine
mim install mmcv
# mim install mmdet
# mim install mmdet3d
# pip install "mmsegmentation>=1.0.0"
```

**f. Install Detectron2 and Timm.**
```shell
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```


**g. Clone BEVFormer.**
```
git clone https://github.com/eliird/bev-former-fork.git
```

**h. Training**
```
check [reimplementation/README](../reimplementation/README.md) for running the model


