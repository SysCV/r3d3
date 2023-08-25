# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def disp_to_depth(
        disp: torch.Tensor,
        min_depth: float,
        max_depth: float,
        focal_length: Optional[torch.Tensor] = None,
        focal_scale: Optional[float] = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Convert network's sigmoid output into depth prediction
    Args:
        disp: Decoder output of shape [B, N, H, W] with values in range [0, 1] ToDo: Check shape
        min_depth: Minimum depth: 1.0 -> min_depth
        max_depth: Maximum depth: 0.0 -> max_depth
        focal_length: Tensor of focal lengths of shape [B]
        focal_scale: Focal length normalization factor -> scale disp by focal_scale / focal_length
    Returns:
        scaled_disp: Scaled disparity of shape [B, N, H, W] with values in range [1/max_depth, 1/min_depth]
        depth: Scaled depth of shape [B, N, H, W] with values in range [min_depth, max_depth]
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp

    if focal_length is not None:  # Focal length scaling of output
        scaled_disp = scaled_disp * focal_scale / focal_length.view(scaled_disp.shape[0], 1, 1, 1)
    depth = 1 / scaled_disp
    return scaled_disp, depth


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")
