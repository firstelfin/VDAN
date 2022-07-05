#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/23 15:19
# @File     : dyhead.py
# @Project  : VDAN

import torch.nn as nn
import munch
from configs import CONFIG_REGISTRY

__call__ = ["fpn"]

backbone_cfg = CONFIG_REGISTRY.get("fpn")()

cfg = {
    "MODEL": {
        "HEAD": {
            "NAME": "DyHead",
            "in_channels": backbone_cfg.MODEL.BACKBONE.out_channels,
            "channels": 256,
            "num_blocks": 6,
        },
        "BACKBONE": {
            "NAME": "FPN",
            "configs": backbone_cfg
        }
    }
}


@CONFIG_REGISTRY.register()
def dyhead():
    CONFIG = munch.munchify(cfg)
    return CONFIG
