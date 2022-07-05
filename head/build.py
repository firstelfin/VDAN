#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/23 10:14
# @File     : build.py
# @Project  : VDAN

from utils import Registry
from backbones import BACKBONE_REGISTRY


HEAD_REGISTRY = Registry("Head")
HEAD_REGISTRY.__doc__ = """
Registrar for header networks of different task types.
"""


def build_head(cfg):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Head`
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone_config = cfg.MODEL.BACKBONE.configs
    backbone = BACKBONE_REGISTRY.get(backbone_name)(**backbone_config.MODEL.BACKBONE)
    head_name = cfg.MODEL.HEAD.NAME
    head = HEAD_REGISTRY.get(head_name)(backbone=backbone, **cfg.MODEL.HEAD)
    assert isinstance(head, Head)
    return head
