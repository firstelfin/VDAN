#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/18 17:51
# @File     : van.py
# @Project  : VDAN

from backbones import Backbone
from .build import BACKBONE_REGISTRY
from utils import init_weights, ShapeSpec
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple


class DWConv(nn.Module):
    """depth-wise Conv"""

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.DWConv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.DWConv(x)
        return x


class Mlp(nn.Module):
    """
    1x1 + depth-wise 3x3 + 1x1
    """

    def __init__(self, in_channels, hidden_channels=None, out_channels=None,
                 act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.FC1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.DWConv = DWConv(hidden_channels)
        self.Act = act_layer()
        self.FC2 = nn.Conv2d(hidden_channels, out_channels, 1)
        self.Drop = nn.Dropout(drop)
        self.apply(init_weights)

    def forward(self, x):
        x = self.FC1(x)
        x = self.DWConv(x)
        x = self.Act(x)
        x = self.Drop(x)
        x = self.FC2(x)
        x = self.Drop(x)
        return x


class LKA(nn.Module):
    """
    5x5 + 7x7(dilation=3) + 1x1
    """

    def __init__(self, dim):
        super().__init__()
        self.Conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.Conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.Conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.Conv0(x)
        attn = self.Conv_spatial(attn)
        attn = self.Conv1(attn)

        return u * attn


class BottleneckAttention(nn.Module):
    """
    Bottleneck Attention: 1x1 + LKA(attention)+ 1x1
    """
    def __init__(self, dim, norm=nn.LayerNorm):
        super().__init__()

        self.Conv1 = nn.Conv2d(dim, dim, 1)
        self.Act = nn.GELU()
        self.SpatialAwareUnit = LKA(dim)
        self.Conv2 = nn.Conv2d(dim, dim, 1)
        # self.Norm = norm(dim)

    def forward(self, x):
        shortcut = x.clone()
        x = self.Conv1(x)
        x = self.Act(x)
        x = self.SpatialAwareUnit(x)
        x = self.Conv2(x)
        # x = self.Norm(x)
        x = x + shortcut
        return x


class Block(nn.Module):
    """
    Basic block of each stage!
    BottleneckAttention + Mlp
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm=nn.LayerNorm):
        super().__init__()

        self.Norm1 = nn.BatchNorm2d(dim)
        self.Attn = BottleneckAttention(dim, norm=norm)
        self.DropPath = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.Norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.Mlp = Mlp(in_channels=dim, hidden_channels=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.LayerScale1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim, )), requires_grad=True)
        self.LayerScale2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim, )), requires_grad=True)

        self.apply(init_weights)

    def forward(self, x):
        x = x + self.DropPath(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.Attn(self.Norm1(x)))
        x = x + self.DropPath(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.Mlp(self.Norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, patch_size=7, in_channels=3, out_channels=768, stride=4):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.Norm = nn.LayerNorm(out_channels)
        self.apply(init_weights)

    def forward(self, x):
        x = self.Conv(x)
        _, _, H, W = x.shape
        x = self.Norm(x)
        return x, H, W


@BACKBONE_REGISTRY.register()
class VAN(Backbone):
    """backbone: pyramidal feature hierarchy
    This module implements :paper:`VAN`.
    Large kernel conv attention(LKA) model configured like Swin Transformer model.
    """
    
    def __init__(self, in_channels=3, num_class=9, embed_dims=(64, 128, 256, 512),
                 mlp_ratios=(4, 4, 4, 4), drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=(3, 4, 6, 3), out_features=None, linear=False, **kwargs):
        super(VAN, self).__init__()
        if out_features is None:
            out_features = ["linear"] if linear else ["van4"]
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        self._depths = {}
        self.stage_names, self.stages = [], []

        num_stages = max(
            [{"van1": 1, "van2": 2, "van3": 3, "van4": 4, "linear": 4}.get(f, 0) for f in out_features]
        )
        depths = depths[:num_stages]
        self.depths = depths
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        current_stride, curr_channels = 1, in_channels
        for i, blocks in enumerate(depths):
            assert blocks > 0, blocks
            name = f"van{i+1}"
            stride = 4 if i == 0 else 2
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=stride,
                in_channels=in_channels if i == 0 else embed_dims[i-1],
                out_channels=embed_dims[i]
            )
            block = nn.ModuleList([
                Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur+j])
                for j in range(blocks)
            ])
            norm = norm_layer(embed_dims[i])
            stage = nn.Sequential(
                patch_embed,
                block,
                norm
            )
            cur += blocks
            setattr(self, f"{name}", stage)
            self._out_feature_channels[name] = curr_channels = embed_dims[i]
            self._out_feature_strides[name] = current_stride = int(current_stride * stride)
            self._depths[name] = blocks
            self.stage_names.append(name)
            self.stages.append(stage)
            pass
        self.stage_names = tuple(self.stage_names)

        if linear:
            self.num_class = num_class
            self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
            self.Linear = nn.Linear(curr_channels, num_class)
            nn.init.normal_(self.Linear.weight, std=0.01)
            nn.init.constant_(self.Linear.bias, 0)
            name = "linear"
            self._out_feature_channels[name] = 1
            self._out_feature_strides[name] = int(current_stride * 1)
            self._depths[name] = 1
        if out_features is None:
            out_features = [name]
        self._out_features = out_features

    def forward(self, x):
        assert x.dim() == 4, f"VAN takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_class is not None:
            x = self.AvgPool(x)
            x = torch.flatten(x, 1)
            x = self.Linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
                depth=self._depths[name]
            )
            for name in self._out_features
        }


pass
if __name__ == '__main__':
    va = VAN(out_features=["van3", "van4"])

    print(va.output_shape().get("van3").stride)
    print(BACKBONE_REGISTRY)
