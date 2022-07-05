#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/23 11:08
# @File     : dyhead.py
# @Project  : VDAN

import torch
import torch.nn.functional as F
from torch import nn
from head.base import Head
from head.build import HEAD_REGISTRY
from utils import ModulatedConv
# from .deform import ModulatedDeformConv
# from .dyrelu import DYReLU


class HardSigmoid(nn.Module):
    """
    formula: max(0, min(1, (x+1)/2)) in paper. in fact: f(x) = max(-3, min(3, x))
    """
    def __init__(self, inplace=True, h_max=1):
        super(HardSigmoid, self).__init__()
        self.ReLU = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.ReLU(x + 3) * self.h_max / 6


class ConvNorm(nn.Module):
    """
    DNC_V2 + GroupNorm
    Spatial-aware basic convolution block
    """
    def __init__(self, in_channels, out_channels, stride):
        super(ConvNorm, self).__init__()

        self.Conv = ModulatedDeformConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.BN = nn.GroupNorm(num_groups=16, num_channels=out_channels)

    def forward(self, x, **kwargs):
        x = self.Conv(x.contiguous(), **kwargs)
        x = self.BN(x)
        return x


class DyConv(nn.Module):
    """
    This module implements :
        paper:`Dynamic Head: Unifying Object Detection Heads with Attentions`:
         * paper: https://arxiv.org/abs/2106.08322 .
         * blog: https://www.cnblogs.com/dan-baishucaizi/p/16388395.html (Contains specific operation diagrams).
    conv_func: A standard ModulatedDeformConv, which is offset and modulation need to be transmit parameters.

    """
    def __init__(self, in_channels=256, out_channels=256, conv_func=ConvNorm):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))  # High resolution down sampling

        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True))

        self.H_sigmoid = HardSigmoid()
        self.ReLU = DYReLU(in_channels, out_channels)
        self.Offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.AttnConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        next_x = {}
        feature_names = list(x.keys())
        for level, name in enumerate(feature_names):

            feature = x[name]

            offset_mask = self.Offset(feature)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, 18:, :, :].sigmoid()
            conv_args = dict(offset=offset, mask=mask)

            temp_fea = [self.DyConv[1](feature, **conv_args)]
            if level > 0:
                temp_fea.append(self.DyConv[2](x[feature_names[level - 1]], **conv_args))
            if level < len(x) - 1:
                temp_fea.append(F.upsample_bilinear(self.DyConv[0](x[feature_names[level + 1]], **conv_args),
                                                    size=[feature.size(2), feature.size(3)]))
            attn_fea = []
            res_fea = []
            for fea in temp_fea:
                res_fea.append(fea)
                attn_fea.append(self.AttnConv(fea))

            res_fea = torch.stack(res_fea)
            spa_pyr_attn = self.H_sigmoid(torch.stack(attn_fea))
            mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)
            next_x[name] = self.ReLU(mean_fea)

        return next_x


@HEAD_REGISTRY.register()
class DyHead(Head):
    def __init__(self, in_channels, channels, num_blocks, backbone, **kwargs):
        super(DyHead, self).__init__()

        self.backbone = backbone

        dyhead_tower = []
        for i in range(num_blocks):
            dyhead_tower.append(
                DyConv(
                    in_channels if i == 0 else channels,
                    channels,
                    conv_func=ConvNorm,
                )
            )

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))

        self._out_feature_strides = self.backbone._out_feature_strides
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: channels for k in self._out_features}
        self._size_divisibility = list(self._out_feature_strides.values())[-1]

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        x = self.backbone(x)
        dyhead_tower = self.dyhead_tower(x)
        return dyhead_tower


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    x2 = np.linspace(-10, 10, 1000)
    y2 = []
    for j in x2:
        y2.append(max(0, min(1, (j+1)/2)))
    x1 = torch.tensor(x2)
    y1 = nn.ReLU6(inplace=True)(x1 + 3)
    y1 = y1 / 6
    plt.plot(x1, y1)
    plt.show()
    plt.plot(x2, y2)
    plt.show()
    pass
