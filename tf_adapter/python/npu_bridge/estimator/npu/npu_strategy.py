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

"""NPU distributed strategy"""

import os
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import one_device_strategy
from npu_bridge.estimator.npu import util as util_lib

from hccl.manage.api import get_rank_size
from hccl.manage.api import get_rank_id


class NPUExtended(one_device_strategy.OneDeviceExtended):
    """NPU implemented oneDevice strategy"""

    def __init__(self, container_strategy, device):
        super(NPUExtended, self).__init__(container_strategy, device)

    @property
    def _num_replicas_in_sync(self):
        rank_size = util_lib.get_ranksize()
        return int(rank_size)

    def _experimental_distribute_dataset(self, dataset):
        return dataset.shard(get_rank_size(), get_rank_id())


class NPUStrategy(distribute_lib.StrategyV1):
    """NPU distribute strategy"""

    def __init__(self, device="/cpu:0"):
        if device != "/cpu:0":
            raise ValueError('"device" only support "/cpu:0"')
        super(NPUStrategy, self).__init__(NPUExtended(self, device))
