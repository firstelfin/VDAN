#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/22 13:41
# @File     : fpn.py
# @Project  : VDAN

import torch.nn as nn
import munch
from configs import CONFIG_REGISTRY

__call__ = ["fpn"]

backbone_cfg = CONFIG_REGISTRY.get("van")()

cfg = {
    "MODEL": {
        "BACKBONE": {
            "NAME": "FPN",
            "in_features": backbone_cfg.MODEL.BACKBONE.out_features,
            "out_channels": 512,
            "norm": "GN",
            "bottom_up": "VAN",
            "top_block": None,
            "fuse_type": "sum",
            "square_pad": 0,
            "configs": {
                "VAN": backbone_cfg
            }
        }
    }
}


@CONFIG_REGISTRY.register()
def fpn():
    CONFIG = munch.munchify(cfg)
    return CONFIG
