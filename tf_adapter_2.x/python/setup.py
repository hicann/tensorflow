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

"""Basic configration for setting up NPU device"""

from setuptools import setup, find_namespace_packages

setup(name='npu_device',
      version='2.6.5',
      description='npu device for tensorflow v2.6.5, tag version v0.0.33',
      long_description='npu device for tensorflow v2.6.5, tag version v0.0.33',
      packages=find_namespace_packages(include=['npu_device*']),
      include_package_data=True,
      ext_modules=[],
      zip_safe=False)
