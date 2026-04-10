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

from npu_bridge import tf_adapter


class Profiler(object):
    def __init__(
        self,
        *,
        level: str = "L0",
        aic_metrics: str = "",
        output_path: str = ""
    ):
        if not isinstance(level, str):
            raise ValueError(f"Option level should be str, but get type: {type(level)}")
        if not isinstance(aic_metrics, str):
            raise ValueError(f"Option aic_metrics should be str, but get type: {type(aic_metrics)}")
        if not isinstance(output_path, str):
            raise ValueError(f"Option output_path should be str, but get type: {type(output_path)}")
        self._level = level
        self._aic_metrics = aic_metrics
        self._output_path = output_path

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exe_type, exe_val, exc_tb):
        self.stop()

    def start(self):
        status = tf_adapter.ProfilerStart(self._level, self._aic_metrics, self._output_path)
        if len(status) != 0:
            raise RuntimeError(status)

    def stop(self):
        status = tf_adapter.ProfilerStop()
        if len(status) != 0:
            raise RuntimeError(status)
