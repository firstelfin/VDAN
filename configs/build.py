#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/22 9:21
# @File     : build.py
# @Project  : VDAN

from utils import Registry


CONFIG_REGISTRY = Registry("Config")
CONFIG_REGISTRY.__doc__ = """
Registry for configs, which records all configuration parameters of the model.
"""