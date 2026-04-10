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

"""NPU training ops"""

from npu_device.npu_device import gen_npu_ops
from tensorflow.python.eager import context
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
from keras.engine import keras_tensor
from npu_device.utils import npu_wrapper


@ops.RegisterGradient("FastGelu")
def fast_gelu_grad(op, grad):
    """The gradient for `fast_gelu`.

    Args:
        op: The `fast_gelu` `Operation` that we are differentiating, which we can use
            to find the inputs and outputs of the original op.
        grad: Gradient with respect to the output of the `fast_gelu` op.

    Returns:
        Gradients with respect to the input of `fast_gelu`.
    """
    return [gen_npu_ops.fast_gelu_grad(grad, op.inputs[0])]


npu_wrapper.npu_symbol_register("npu.ops.gelu", gen_npu_ops.fast_gelu)


def gelu(x):
    """ fast_gelu operator interface implementation.

    An approximate implementation of Computing the Gaussian Error Linear Unit (GELU) activation function.
    It is used to replace `tf.nn.gelu` to get performance improvements, but the result will have some difference
    in precision from the origin.

    Args:
        x: A input tensor representing preactivation values, with type is float16 or float32.

    Returns:
         A tensor with the same type as `x`.
    """
    if any(isinstance(e, keras_tensor.KerasTensor) for e in nest.flatten([x])):
        # in case that Functional API construction
        return npu_wrapper.NpuOpLambda(gen_npu_ops.fast_gelu)(x)

    if context.executing_eagerly():
        return nn_ops.gelu(x)
    x = ops.convert_to_tensor(x)
    return gen_npu_ops.fast_gelu(x)
