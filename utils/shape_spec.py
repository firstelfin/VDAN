#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/18 17:08
# @File     : shape_spec.py
# @Project  : VDAN
# Copy from Facebook

from dataclasses import dataclass
from typing import Optional


@dataclass
class ShapeSpec:
    """
    Data extension of torch.
    """
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None
    depth: Optional[int] = None
