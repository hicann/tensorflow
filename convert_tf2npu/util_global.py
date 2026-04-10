#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""functions for global varabiles manegement"""

import json
import time
import os
import collections

ParamConfig = collections.namedtuple('ParamConfig', \
    ['short_opts', 'long_opts', 'opt_err_prompt', 'opt_help', 'support_list_filename', 'main_arg_not_set_promt'])


def init():
    global _global_dict
    _global_dict = {}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mappings', 'ast.json')) as f:
        load_dict = json.load(f)
        items = load_dict.items()
        for key, value in items:
            set_value(key, value)
    value = "_npu_" + time.strftime('%Y%m%d%H%M%S')
    set_value('timestap', value)


def set_value(key, value):
    """Set value for global dictionary"""
    _global_dict[key] = value


def get_value(key, def_value=None):
    """Get value by key from global dictionary"""
    try:
        return _global_dict[key]
    except KeyError:
        return def_value
