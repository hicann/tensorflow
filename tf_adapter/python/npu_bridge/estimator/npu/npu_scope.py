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

"""
Config the non npu compilation scope for NPU in mix compute mode.
"""
import contextlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops
from tensorflow.python.util import compat
from npu_bridge.estimator.npu.npu_config import NpuExecutePlacement
from npu_bridge.estimator.npu import util


@contextlib.contextmanager
def without_npu_compile_scope():
    """
    Enable the non npu compilation of operators within the scope.
    """
    attrs = {
        "_without_npu_compile": attr_value_pb2.AttrValue(b=True)
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


@contextlib.contextmanager
def npu_variable_scope(placement=NpuExecutePlacement.ALL):
    """
    Enable the node in the scope adding _variable_placement attr.
    """
    if placement not in NpuExecutePlacement:
        raise ValueError("placement vaule must be in NpuExecutePlacement's vaule")
    attrs = {
        "_variable_placement": attr_value_pb2.AttrValue(s=compat.as_bytes(placement.value))
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


@contextlib.contextmanager
def keep_dtype_scope():
    """
    Specify which layers retain the original precision.
    """
    attrs = {
        "_keep_dtype": attr_value_pb2.AttrValue(i=1)
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


@contextlib.contextmanager
def npu_stage_scope(stage):
    """
    npu_stage_scope provides a graph-building space, all nodes under this space form a 'Stage'.
    The graph will be divided into different execution units according to different 'Stage'
    Stage's sequential execution is equivalent to graph execution，
    When the graph is executed multiple times, there is a possibility of parallelism between stages of different Steps.

    Input:
        stage: stage id of current scope, all nodes under this scope will have attr '_stage_level' with value 'stage'

    Usage example：
        We want to perform a unique calculation on the input data first, and then sum， just like
        def my_model(x):
            ux, _ = tf.unique(x)
            sum = tf.reduce_sum(ux)
        The executed prof data is as follows：
        ┌─────────────────┐ ┌─────────────────────────┐
        │  Unique @AICPU  │ │    ReduceSum @AICORE    │
        └─────────────────┘ └─────────────────────────┘
        We can optimize it at execution time through npu stage scope，and with 'iterations_per_loop' = '2'
        def my_model(x):
            with npu_stage_scope(0):
               ux, _ = tf.unique(x)
            sum = tf.reduce_sum(ux)
        Then the executed prof data is as follows：
        ┌─────────────────┐ ┌───────────────────────┐
        │  Unique @CPU    │ │    ReduceSum @CORE    │
        └─────────────────┘ └───────────────────────┘
                           ┌───────────────┐         ┌───────────────────────┐
                           │  Unique @CPU  │         │    ReduceSum @CORE    │
                           └───────────────┘         └───────────────────────┘

    Use constraints:
        1. The 'iterations_per_loop' config must be configured to be greater than 1 during pipeline execution
        2. Performance gains are possible only when different computing resources are used between different stages
        3. Communication under the same communication domain is not allowed between different Stages
    """
    attrs = {
        "_stage_level": attr_value_pb2.AttrValue(i=stage)
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


@contextlib.contextmanager
def npu_mem_type_scope():
    """
    Enable the node in the scope adding _output_memory_type attr.
    """
    attrs = {
        "_output_memory_type": attr_value_pb2.AttrValue(i=1)
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


@contextlib.contextmanager
def npu_weight_prefetch_scope(buffer_pool_id=0, buffer_pool_size=536870912):
    """
    Enable the PREFETCH node in the scope to use buffer pool memory.
    buffer_pool_id: Specifies the id of buffer pool to enable,
                    it is a integer, default is 0;
    buffer_pool_size: Specifies the size of this buffer pool in bytes,
                      default is 512MB.

    Use constraints:
    1. BufferPoolMemory is only supported for PREFETCH node with single
       input and single output;
    2. Buffer pool size of the same ID must be the same;
    3. The size of the buffer pool should be able to meet the requirements
       of the PREFETCH node with the largest memory (note that alignment
       and complement are included, for example, 512 bytes alignment of
       the HCOM node with an additional 512 bytes before and after each);
    4. Prefetch is not supported if it is located in a subgraph or
       in a control flow branch.
    """
    attrs = {
        "_buffer_pool_id": attr_value_pb2.AttrValue(i=buffer_pool_id),
        "_buffer_pool_size": attr_value_pb2.AttrValue(i=buffer_pool_size)
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


@contextlib.contextmanager
def op_specified_engine_scope(engine_name, kernel_lib_name):
    """
    Enable the node in the scope adding _specified_engine_name and _specified_kernel_lib_name attr.
    """
    attrs = {
        "_specified_engine_name": attr_value_pb2.AttrValue(s=compat.as_bytes(engine_name)),
        "_specified_kernel_lib_name": attr_value_pb2.AttrValue(s=compat.as_bytes(kernel_lib_name))
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


@contextlib.contextmanager
def npu_qos_label_scope(label):
    """
    Enable the node in the scope adding _qos_service_label attr.
    """
    attrs = {
        "_qos_service_label": attr_value_pb2.AttrValue(i=label)
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


@contextlib.contextmanager
def subgraph_multi_dims_scope(index):
    """
    Enable the node in the scope adding subgraph multi dims index.
    """
    attrs = {
        "_subgraph_multi_dims_index": attr_value_pb2.AttrValue(i=index)
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


@contextlib.contextmanager
def npu_optimizer_scope():
    """
    Enable the non npu compilation of operators within the scope.
    """
    attrs = {
        "_optimizer": attr_value_pb2.AttrValue(b=True)
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


def npu_recompute_scope():
    return ops.name_scope_v2("NpuRecompute")


def npu_graph_slice_scope(slice_num=None):
    if slice_num is not None:
        util.check_positive_integer(slice_num, "slice_num")
    return ops.name_scope_v2("".join(["SliceNum-", str(slice_num), "-NpuGraphSlicing", str(slice_num)]))


@contextlib.contextmanager
def disable_autofuse():
    """
    Disable the autofuse of operators within the scope.
    """
    attrs = {
        "_disable_autofuse_scope": attr_value_pb2.AttrValue(b=True)
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


@contextlib.contextmanager
def limit_core_num_scope(op_aicore_num="0", op_vectorore_num="0"):
    """
    Limit the aic and aiv core num of autofuse operators within the scope, only support aiv core num now.
    """
    if not isinstance(op_vectorore_num, str):
        raise ValueError("Param op_vectorore_num must be string.")
    try:
        int_vector_core_num = int(op_vectorore_num)
    except ValueError:
        raise ValueError("Param op_vectorore_num can not be converted into a valid int number.")
    if int_vector_core_num <= 0:
        raise ValueError("Param op_vectorore_num must be greater than zero.")
    attrs = {
        "_op_vectorcore_num": attr_value_pb2.AttrValue(s=compat.as_bytes(op_vectorore_num))
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield
