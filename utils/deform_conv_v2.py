#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/24 9:56
# @File     : deform_conv_v2.py
# @Project  : VDAN

import os
import math
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load
try:
    import Deformable
except ModuleNotFoundError:
    # JIT
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sources = ['/vision.cpp', '/cuda/deform_conv_cuda.cu',
               '/cuda/deform_conv_kernel_cuda.cu', '/cuda/SigmoidFocalLoss_cuda.cu']
    sources = [this_dir + "/csrc/" + i for i in sources]
    Deformable = load(name="Deformable", sources=sources, verbose=True)


class ModulatedDeformConvFunction(Function):
    """
    Writing reference: https://pytorch.org/docs/1.10/autograd.html#function
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        ctx.stride = kwargs["stride"]
        ctx.padding = kwargs["padding"]
        ctx.dilation = kwargs["dilation"]
        ctx.groups = kwargs["groups"]
        ctx.deformable_groups = kwargs["deformable_groups"]
        ctx.with_bias = kwargs["bias"] is not None
        if not ctx.with_bias:
            kwargs["bias"] = args[0].new_empty(1)  # fake tensor
        if not args[0].is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or args[0].requires_grad:
            ctx.save_for_backward(args[0], offset, mask, weight, kwargs["bias"])
        output = args[0].new_empty(
            ModulatedDeformConvFunction._infer_shape(ctx, args[0], weight))
        ctx._bufs = [args[0].new_empty(0), args[0].new_empty(0)]
        Deformable.modulated_deform_conv_forward(
            args[0],
            weight,
            kwargs["bias"],
            ctx._bufs[0],
            offset,
            mask,
            output,
            ctx._bufs[1],
            weight.shape[2],
            weight.shape[3],
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            ctx.with_bias
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        x, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(x)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        Deformable.modulated_deform_conv_backward(
            x,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            ctx._bufs[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            weight.shape[2],
            weight.shape[3],
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            ctx.with_bias
        )
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)

    @staticmethod
    def _infer_shape(ctx, x, weight):
        n = x.size(0)
        channels_out = weight.size(0)
        height, width = x.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding -
                      (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding -
                     (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = ModulatedDeformConvFunction.apply


class ModulatedConv(nn.Module):
    """
    1
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv(
            x, offset=offset, mask=mask, weight=self.weight, bias=self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups, deformable_groups=self.deformable_groups
        )

    def __repr__(self):
        return "".join([
            "{}(".format(self.__class__.__name__),
            "in_channels={}, ".format(self.in_channels),
            "out_channels={}, ".format(self.out_channels),
            "kernel_size={}, ".format(self.kernel_size),
            "stride={}, ".format(self.stride),
            "dilation={}, ".format(self.dilation),
            "padding={}, ".format(self.padding),
            "groups={}, ".format(self.groups),
            "deformable_groups={}, ".format(self.deformable_groups),
            "bias={})".format(self.with_bias),
        ])


class ModulatedDeformConv(nn.Module):
    """
    Deformable Conv v2: Offset + Modulated.
    the out_channels of Conv_offset_mask: deformable_groups * 3 * kernel_size[0] * kernel_size[1].
        kernel_size[0] * kernel_size[1] is Sampling number of convolution kernel; 3 means that each
         point needs to generate two offsets and one modulation.
    """

    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.Conv_offset_mask = nn.Conv2d(
            in_channels // groups,
            deformable_groups * 3 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=_pair(stride),
            padding=_pair(padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        """Initialize to 0 according to the implementation method of the paper"""
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        out = self.Conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(
            x, offset=offset, mask=mask, weight=self.weight, bias=self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups, deformable_groups=self.deformable_groups
        )
