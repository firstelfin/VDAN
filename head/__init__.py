#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2022/6/23 10:12
# @File     : __init__.py.py
# @Project  : VDAN

from .build import HEAD_REGISTRY, build_head
from .base import Head
from .dyhead import DyHead


__call__ = ["Head", "HEAD_REGISTRY", "build_head"]
