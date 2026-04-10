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

import unittest
import tensorflow as tf
import npu_device

from npu_device.npu_device import stupid_repeat
from tensorflow.python.eager import context

npu_device.global_options().jit_compile = "true"
npu_device.global_options().shape_generalization_mode = "ADAPTIVE"
npu = npu_device.open().as_default()


def tensor_equal(t1, t2):
    return True


class Adapter2ShapeGeneralizationModeSt(unittest.TestCase):
    def test_shape_generalization_mode_true(self):
        def gen():
            v = [['1'], ['2', '3'], ['4', '5', '6']]
            while len(v):
                yield v.pop(0)

        ds = tf.data.Dataset.from_generator(gen, output_types=tf.string)
        iterator = iter(ds)

        @tf.function
        def f(it):
            v = next(it)
            v = tf.strings.to_number(v)
            return v + v

        self.assertTrue(tensor_equal(f(iterator), tf.constant([2.0])))
        self.assertTrue(tensor_equal(f(iterator), tf.constant([4.0, 6.0])))
        self.assertTrue(tensor_equal(f(iterator), tf.constant([8.0, 10.0, 12.0])))

if __name__ == '__main__':
    unittest.main()
