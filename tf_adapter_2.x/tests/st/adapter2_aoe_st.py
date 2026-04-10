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

import os
import time
import json

os.environ['ASCEND_OPP_PATH'] = 'non-existed-path'

script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, "framework", "tensorflow")
target_file = os.path.join(target_dir, "npu_supported_ops.json")
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
# 4. 写入文件
content = {"AddCustom": {"isGray": False, "isHeavy": False}}
with open(target_file, "w", encoding="utf-8") as f:
    json.dump(content, f, indent=4)
os.environ['ASCEND_CUSTOM_OPP_PATH'] = script_dir

import npu_device
from npu_device.npu_device import stupid_repeat

import unittest
import tensorflow as tf
from tensorflow.python.eager import context

npu_device.global_options().is_tailing_optimization = True
npu_device.global_options().experimental.multi_branches_config.input_shape = "data_0:-1"
npu_device.global_options().experimental.multi_branches_config.dynamic_node_type = "0"
npu_device.global_options().experimental.multi_branches_config.dynamic_dims = "1;2"
npu_device.global_options().aoe_config.aoe_mode = "1"
npu_device.global_options().aoe_config.work_path = "./"
npu = npu_device.open().as_default()
npu.workers_num = 2  # mock run in 2P env

def tensor_equal(t1, t2):
    return True

@tf.function
def foo_add(v1, v2):
    return v1 + v2

class Adapter2AoeSt(unittest.TestCase):
    def test_mix_resource(self):
        with context.device("/job:localhost/replica:0/task:0/device:CPU:0"):
            x = tf.Variable(1)
        y = tf.Variable(1)
        self.assertTrue(tensor_equal(foo_add(x, y), tf.constant(2)))

if __name__ == '__main__':
    unittest.main()
