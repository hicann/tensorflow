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

from npu_device.npu_device import open
from npu_device.npu_device import npu_compat_function
from npu_device.npu_device import gen_npu_ops
from npu_device.npu_device import global_options
from npu_device.npu_device import set_npu_loop_size
from npu_device.npu_device import npu_run_context
from npu_device.npu_device import set_device_sat_mode
from npu_device.npu_device import is_inf_nan_enabled

from npu_device.utils.scope import keep_dtype_scope
from npu_device.utils.scope import npu_recompute_scope

from npu_device._api import distribute
from npu_device._api import train
from npu_device._api import ops
from npu_device._api import compat
from npu_device._api import configs
