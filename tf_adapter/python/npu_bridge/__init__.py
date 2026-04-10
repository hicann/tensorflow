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

import atexit
from npu_bridge.helper.helper import npu_bridge_handle
from npu_bridge.helper.helper import version as __version__
from npu_bridge.helper import helper
from npu_bridge.estimator.npu import npu_estimator
from npu_bridge.hccl import hccl_ops
from npu_bridge.estimator.npu.npu_plugin import npu_close

atexit.register(npu_close)
__all__ = [_s for _s in dir() if not _s.startswith('_')]
