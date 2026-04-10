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

"""NPU scope management"""

from tensorflow.python.framework import ops
from tensorflow.python.util import tf_contextlib
from tensorflow.core.framework import attr_value_pb2


@tf_contextlib.contextmanager
def keep_dtype_scope():
    """Execute in keep_dtype_scope"""
    with ops.get_default_graph()._attr_scope({'_keep_dtype': attr_value_pb2.AttrValue(i=1)}):
        yield


@tf_contextlib.contextmanager
def npu_optimizer_scope():
    """
    add _optimizer attr to node within the scope.
    """
    with ops.get_default_graph()._attr_scope({"_optimizer": attr_value_pb2.AttrValue(b=True)}):
        yield


@tf_contextlib.contextmanager
def npu_gradients_scope():
    """
    add _backward attr to node within the scope.
    """
    with ops.get_default_graph()._attr_scope({"_backward": attr_value_pb2.AttrValue(b=True)}):
        yield


@tf_contextlib.contextmanager
def npu_recompute_scope():
    """
    add _recompute attr to node within the scope.
    """
    with ops.get_default_graph()._attr_scope({"_recompute": attr_value_pb2.AttrValue(b=True)}):
        yield
