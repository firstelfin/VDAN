#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/21 15:59
# @File     : train.py
# @Project  : VDAN

from backbones import build_backbone, BACKBONE_REGISTRY
from head import HEAD_REGISTRY, build_head
from configs import CONFIG_REGISTRY

print(CONFIG_REGISTRY)
# model = build_backbone(CONFIG_REGISTRY.get("fpn")())
model = build_head(CONFIG_REGISTRY.get("dyhead")())
print(model.output_shape())
