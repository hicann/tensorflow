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

"""LossScaleManager classes for mixed precision training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import state_ops
from tensorflow.contrib.mixed_precision.python import loss_scale_manager as lsm_lib


class FixedLossScaleManager(lsm_lib.FixedLossScaleManager):
    """Loss scale manager with a fixed loss scale.
    """

    def __init__(self, loss_scale, enable_overflow_check=True):
        """Creates the fixed loss scale manager.
        """
        if loss_scale < 1:
            raise ValueError("loss scale must be at least 1.")
        self._loss_scale = ops.convert_to_tensor(loss_scale, dtype=dtypes.float32, name="loss_scale")
        self._enable_overflow_check = enable_overflow_check
        super(FixedLossScaleManager, self).__init__(loss_scale=loss_scale)

    def get_enable_overflow_check(self):
        """Enable overflow check"""
        return self._enable_overflow_check


class ExponentialUpdateLossScaleManager(lsm_lib.ExponentialUpdateLossScaleManager):
    """Loss scale manager uses an exponential update strategy.
    """

    def __init__(self,
                 init_loss_scale,
                 incr_every_n_steps,
                 decr_every_n_nan_or_inf=2,
                 incr_ratio=2,
                 decr_ratio=0.8):
        """Constructor of exponential-update loss scale manager.
        """
        super(ExponentialUpdateLossScaleManager, self).__init__(
            init_loss_scale=init_loss_scale,
            incr_every_n_steps=incr_every_n_steps,
            decr_every_n_nan_or_inf=decr_every_n_nan_or_inf,
            incr_ratio=incr_ratio,
            decr_ratio=decr_ratio)

    def update_loss_scale(self, finite_grads):
        """Used to update loss scale"""
        def update_if_finite_grads():
            def increase_loss_scale():
                incr_result_finite = gen_math_ops.less(self._loss_scale, (3.4e+38) / self._incr_ratio)
                new_loss_scale_value = control_flow_ops.cond(
                    incr_result_finite,
                    lambda: self._loss_scale * self._incr_ratio,
                    lambda: self._loss_scale)
                update_loss_scale = state_ops.assign(self._loss_scale, new_loss_scale_value)
                return control_flow_ops.group(update_loss_scale, self._reset_stats())

            is_incr_good_steps = self._num_good_steps + 1 >= self._incr_every_n_steps
            return control_flow_ops.cond(is_incr_good_steps, increase_loss_scale,
                                         lambda: state_ops.assign_add(self._num_good_steps, 1).op)

        def update_if_not_finite_grads():
            def decrease_loss_scale():
                new_loss_scale_value = gen_math_ops.maximum(1., self._loss_scale * self._decr_ratio)
                update_loss_scale = state_ops.assign(self._loss_scale, new_loss_scale_value)
                return control_flow_ops.group(update_loss_scale, self._reset_stats())

            def only_update_steps():
                return control_flow_ops.group(state_ops.assign_add(self._num_bad_steps, 1),
                                              state_ops.assign(self._num_good_steps, 0))

            is_incr_bad_steps = self._num_bad_steps + 1 >= self._decr_every_n_nan_or_inf
            return control_flow_ops.cond(is_incr_bad_steps, decrease_loss_scale, only_update_steps)

        return control_flow_ops.cond(finite_grads, update_if_finite_grads, update_if_not_finite_grads)
