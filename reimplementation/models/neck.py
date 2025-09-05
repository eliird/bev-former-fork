"""
Pure PyTorch implementation of FPN neck
Converted from MMDetection's FPN implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union


class FPNNeck(nn.Module):
    """Feature Pyramid Network neck for BEVFormer.
    
    This is a simplified version of FPN specifically for BEVFormer which 
    typically uses only the highest level feature (C5) from ResNet.
    
    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level.
        add_extra_convs (bool | str): Whether to add extra conv layers.
        relu_before_extra_convs (bool): Whether to apply relu before extra conv.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 start_level: int = 0,
                 end_level: int = -1,
                 add_extra_convs: Union[bool, str] = False,
                 relu_before_extra_convs: bool = False):
        super(FPNNeck, self).__init__()
        
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        
        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level + 1
        
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        # Build lateral convs
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(
                in_channels[i], out_channels, 1, bias=False)
            self.lateral_convs.append(l_conv)

        # Build fpn convs
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            fpn_conv = nn.Conv2d(
                out_channels, out_channels, 3, padding=1, bias=False)
            self.fpn_convs.append(fpn_conv)

        # Build extra convs
        if self.num_outs > len(in_channels) - start_level:
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels) - start_level, self.num_outs):
                if self.add_extra_convs == 'on_input':
                    extra_conv = nn.Conv2d(
                        in_channels[-1], out_channels, 3, stride=2, padding=1, bias=False)
                else:
                    extra_conv = nn.Conv2d(
                        out_channels, out_channels, 3, stride=2, padding=1, bias=False)
                self.extra_convs.append(extra_conv)

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # Build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale_factor` (e.g. 2) is preferred, but
            # it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.__dict__.get('upsample_cfg', {}):
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], scale_factor=2, mode='nearest')
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, mode='nearest')

        # Build outputs
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # Part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                
                for i, extra_conv in enumerate(self.extra_convs):
                    if self.relu_before_extra_convs and i == 0:
                        extra_source = F.relu(extra_source, inplace=True)
                    outs.append(extra_conv(extra_source))
                    extra_source = outs[-1]

        return tuple(outs)


# Test function
def test_fpn():
    """Test FPN neck with BEVFormer configuration"""
    # BEVFormer tiny config: uses only C5 (index 3) from ResNet50
    in_channels = [2048]  # Only C5
    out_channels = 256
    num_outs = 1
    
    fpn = FPNNeck(
        in_channels=in_channels,
        out_channels=out_channels,
        num_outs=num_outs,
        start_level=0,
        add_extra_convs='on_output',
        relu_before_extra_convs=True)
    
    # Test input - C5 feature from ResNet50
    # Input size: 800x450 -> after ResNet: 25x15 (stride 32)
    x = torch.randn(1, 2048, 25, 15)
    inputs = (x,)
    
    outputs = fpn(inputs)
    
    print(f"Input shape: {x.shape}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
    
    return fpn


if __name__ == "__main__":
    test_fpn()