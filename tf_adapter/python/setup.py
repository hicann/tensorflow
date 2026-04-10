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

"""npu bridge for tensorflow v1.15.0, tag version v0.0.33
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_namespace_packages

DOCLINES = __doc__.split('\n')

# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
_VERSION = '1.15.0'

setup(
    name='npu_bridge',
    version=_VERSION,
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    author='HuaWei Inc.',
    # Contained modules and scripts.
    packages=find_namespace_packages(include=['npu_bridge*']),
    # Add in any packaged data.
    include_package_data=True,
    keywords='tensorflow tensor machine learning',
)
