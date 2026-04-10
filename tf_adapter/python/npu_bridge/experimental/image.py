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

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from npu_bridge.helper import helper

gen_npu_cpu_ops = helper.get_gen_ops();


def decode_image(contents, channels=0, dtype=dtypes.uint8, expand_animations=True):
    """
    Decode image.

    :param contents: string 类型.
    :param channels int 类型.
    :param expand_animations bool 类型.
    :return image
    """
    return gen_npu_cpu_ops.decode_image_v3(
        contents=contents,
        channels=channels,
        dtype=dtype,
        expand_animations=expand_animations)
