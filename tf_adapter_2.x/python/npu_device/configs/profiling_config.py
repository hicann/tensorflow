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

"""Configuration for NPU profiling"""

from npu_device.configs.option_base import OptionValue
from npu_device.configs.option_base import NpuBaseConfig


class NpuProfilingConfig(NpuBaseConfig):
    """"NPU profiling configurations"""
    def __init__(self):
        self.enable_profiling = OptionValue(False, [True, False])
        self.profiling_options = OptionValue("{\"output\":\".\\/\",\"training_trace\":\"on\",\"task_time\":\"on\",\
                                             \"hccl\":\"on\",\"aicpu\":\"on\",\"aic_metrics\":\"PipeUtilization\",\
                                             \"msproftx\":\"off\"}", None)

        super(NpuProfilingConfig, self).__init__()
