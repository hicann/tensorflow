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

"""NPU implemented rnn"""

import tensorflow as tf


def npu_dynamic_rnn(cell,
                    inputs,
                    initial_state=None,
                    dtype=None,
                    sequence_length=None,
                    scope=None):
    """Creates a high performance neural network specified by RNNCell `cell`.
    """
    # tf origin static_rnn
    inputs = tf.unstack(inputs, axis=0)
    encoder_outputs, encoder_state = tf.nn.static_rnn(
        cell,
        inputs,
        initial_state=initial_state,
        dtype=dtype,
        sequence_length=sequence_length,
        scope=scope)
    encoder_outputs = tf.stack(encoder_outputs, axis=0)

    return encoder_outputs, encoder_state
