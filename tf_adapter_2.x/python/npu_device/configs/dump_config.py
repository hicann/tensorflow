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

"""Configuration for dumping NPU data"""

from npu_device.configs.option_base import OptionValue
from npu_device.configs.option_base import NpuBaseConfig


class NpuDumpConfig(NpuBaseConfig):
    """Config for dumping npu training data"""
    def __init__(self):
        self.enable_dump = OptionValue(False, [True, False])
        self.dump_path = OptionValue(None, None)
        self.dump_step = OptionValue(None, None)
        self.dump_mode = OptionValue('output', ['input', 'output', 'all'])
        self.enable_dump_debug = OptionValue(False, [True, False])
        self.dump_debug_mode = OptionValue('all', ['aicore_overflow', 'atomic_overflow', 'all'])
        self.dump_data = OptionValue('tensor', ['tensor', 'stats'])
        self.dump_layer = OptionValue(None, None)

        super(NpuDumpConfig, self).__init__()
