#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/21 18:15
# @File     : build.py
# @Project  : VDAN
from backbones import Backbone
from utils import Registry

BACKBONE_REGISTRY = Registry("Backbone")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images

The registered object must be a callable that accepts one arguments:

1. A :class:`munch.Munch`

Registered object must return instance of :class:`Backbone`.

"""


def build_backbone(cfg):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(**cfg.MODEL.BACKBONE)
    assert isinstance(backbone, Backbone)
    return backbone
