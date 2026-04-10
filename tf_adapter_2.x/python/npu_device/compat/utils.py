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

"""Public functions for NPU compat"""

import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


class SessionConfigBuilder:
    def __init__(self, config=None):
        if isinstance(config, tf.compat.v1.ConfigProto):
            self._session_config = config
        else:
            self._session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        self._session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        self._npu_config = self._session_config.graph_options.rewrite_options.custom_optimizers.add()
        self._npu_config.name = 'NpuOptimizer'

    @property
    def parameter_map(self):
        return self._npu_config.parameter_map

    @property
    def session_config(self):
        return self._session_config
