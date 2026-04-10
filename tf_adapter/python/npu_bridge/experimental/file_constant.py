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

gen_npu_cpu_ops = helper.get_gen_ops()


## 提供FileConstant功能
#  @param shape list(int) 类型
#  @param dtype float, float16, int8, int16, uint16,
#               uint8, int32, int64, uint32, uint64, bool, double 类型
#  @param file_path string 类型
#  @param file_id string 类型
#  @return y float, float16, int8, int16, uint16,
#            uint8, int32, int64, uint32, uint64, bool, double 类型
def file_constant(shape, dtype, file_path=None, file_id=None, name=None):
    """ file constant. """
    result = gen_npu_cpu_ops.file_constant(
        file_path=file_path,
        file_id=file_id,
        shape=shape,
        dtype=dtype,
        name=name)
    return result
