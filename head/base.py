#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/23 10:29
# @File     : base.py
# @Project  : VDAN

from abc import ABCMeta, abstractmethod
from typing import Dict
import torch.nn as nn
from utils import ShapeSpec


__all__ = ["Head"]


class Head(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network heads.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """
        Subclasses must override this method, but adhere to the same return type.
        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "p5") to tensor
        """
        pass

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

