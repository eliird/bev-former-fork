# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation



**a. Create a virtual environment**
```shell
python3 -m venv .venv
. .venv/bin/activate
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install scipy pyquaternion nuscenes-devkit tqdm shapely

```

**c. Install mmcv-full.**

we use mmcv for dataset prepartion and some custom kernels for losses

```shell

pip install openmim
mim install mmengine
mim install mmcv


# mim trouble shooting
# if you get any parsing or python libraries error mim for some reason decreaase the setup tools version reupgrade it 
python -m ensurepip --upgrade
python -m pip install --upgrade setuptools

```

<!-- **f. Install Detectron2 and Timm.**
```shell
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
 -->

**g. Clone BEVFormer.**
```
git clone https://github.com/eliird/bev-former-fork.git
```

**h. Training**
```
check [reimplementation/README](../reimplementation/README.md) for running the model


