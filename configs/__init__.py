#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/22 9:10
# @File     : __init__.py.py
# @Project  : VDAN

from configs.build import CONFIG_REGISTRY
from configs.van import *
from configs.fpn import *
from configs.dyhead import *

__call__ = ["CONFIG_REGISTRY"]
