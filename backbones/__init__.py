#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/18 17:31
# @File     : __init__.py.py
# @Project  : VDAN

from .base import Backbone
from .build import build_backbone, BACKBONE_REGISTRY
from .van import (
    VAN,
    DWConv,
    Mlp
)
from .fpn import *

# BACKBONE_REGISTRY.register(VAN)

__call__ = ["Backbone", "BACKBONE_REGISTRY", "build_backbone", "VAN", "DWConv", "Mlp"]
