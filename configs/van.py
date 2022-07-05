#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/22 9:11
# @File     : van.py
# @Project  : VDAN

import torch.nn as nn
import munch
from configs import CONFIG_REGISTRY

__call__ = ["van"]

cfg = {
    "MODEL": {
        "BACKBONE": {
            "NAME": "VAN",
            "in_channels": 3,
            "num_class": 9,
            "embed_dims": (64, 128, 256, 512),
            "mlp_ratios": (4, 4, 4, 4),
            "drop_rate": 0.,
            "drop_path_rate": 0.,
            "norm_layer": nn.LayerNorm,
            "depths": (3, 4, 6, 3),
            "out_features": ["van2", "van3", "van4"],
            "linear": False
        }
    }
}


@CONFIG_REGISTRY.register()
def van():
    CONFIG = munch.munchify(cfg)
    return CONFIG


