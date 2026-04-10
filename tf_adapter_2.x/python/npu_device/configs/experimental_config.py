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

"""Configuration for experiment"""

from npu_device.configs.option_base import NpuBaseConfig
from npu_device.configs.option_base import OptionValue
from npu_device.configs.multi_branches_config import NpuMultiBranchesConfig
from npu_device.configs.logical_device_deploy_config import LogicalDeviceDeployConfig
from npu_device.configs.model_deploy_config import ModelDeployConfig
from npu_device.configs.memory_optimize_config import GraphMemoryOptimizeConfig


class NpuExperimentalConfig(NpuBaseConfig):
    """Config for experiment"""
    def __init__(self):
        self.multi_branches_config = NpuMultiBranchesConfig()
        self.logical_device_deploy_config = LogicalDeviceDeployConfig()
        self.model_deploy_config = ModelDeployConfig()

        # run context options
        self.graph_memory_optimize_config = GraphMemoryOptimizeConfig()
        self.resource_config_path = OptionValue(None, None)
        self.graph_parallel_option_path = OptionValue(None, None)
        self.enable_graph_parallel = OptionValue(False, [True, False])

        super(NpuExperimentalConfig, self).__init__()
