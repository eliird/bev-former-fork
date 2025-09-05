"""
Pure PyTorch implementation of ResNet backbone
Converted from MMDetection's ResNet implementation
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from typing import Dict, List, Optional, Union


class BasicBlock(nn.Module):
    """Basic block for ResNet."""
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 with_cp: bool = False):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            inplanes, planes, 3,
            stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet."""
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 with_cp: bool = False):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(
            planes, planes, 3,
            stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    """ResNet backbone for BEVFormer
    
    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        num_stages (int): Number of stages. Default: 4.
        strides (tuple): Strides of the first block of each stage.
        dilations (tuple): Dilation of each stage.
        out_indices (tuple): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
        with_cp (bool): Use checkpoint or not.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth: int = 50,
                 num_stages: int = 4,
                 strides: tuple = (1, 2, 2, 2),
                 dilations: tuple = (1, 1, 1, 1),
                 out_indices: tuple = (0, 1, 2, 3),
                 frozen_stages: int = 1,
                 with_cp: bool = False):
        super(ResNetBackbone, self).__init__()
        
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        
        self.depth = depth
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.with_cp = with_cp
        
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64
        
        # Build stem layer
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build residual layers
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            
            res_layer = self._make_layer(
                self.block,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                with_cp=with_cp)
            
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        
        self._freeze_stages()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, with_cp=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                with_cp=with_cp))
        
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    dilation=dilation,
                    with_cp=with_cp))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        
        return tuple(outs)

    def train(self, mode=True):
        super(ResNetBackbone, self).train(mode)
        self._freeze_stages()
        return self


# Test function
def test_resnet():
    """Test ResNet backbone"""
    model = ResNetBackbone(depth=50, out_indices=(3,))  # BEVFormer uses only C5
    
    # Test with random input
    x = torch.randn(1, 3, 800, 450)  # BEVFormer tiny input size
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
    
    return model


if __name__ == "__main__":
    test_resnet()