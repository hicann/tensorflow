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

"""NPU hccl functions"""

import os
import ctypes
from npu_bridge.estimator.npu import util as util_lib


hccl_graph_adp_ctypes = ctypes.CDLL('libhcom_graph_adaptor.so')


def c_str(string):
    return ctypes.c_char_p(string.encode('utf-8'))


def get_actual_rank_size(group="hccl_world_group"):
    c_group = c_str(group)
    c_rank_size = ctypes.c_uint()
    ret = hccl_graph_adp_ctypes.HcomGetActualRankSize(c_group, ctypes.byref(c_rank_size))
    if ret != 0:
        raise ValueError('get actual rank size error.')
    return c_rank_size.value


def get_user_rank_size():
    rank_size = int(util_lib.get_ranksize())
    return rank_size


def get_user_rank_id():
    rank_id = int(os.getenv('RANK_ID'))
    return rank_id
