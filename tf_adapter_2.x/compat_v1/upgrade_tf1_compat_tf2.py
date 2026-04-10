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

"""Make adapter 1.x compat for tf2"""

import os
import sys
import shutil
import re

REGEXP_RULES = dict()
REPLACE_RULES = dict()
REMOVED_LINES = []

REPLACE_RULES['from npu_bridge'] = 'from npu_device.compat.v1'
REPLACE_RULES['import tensorflow'] = 'import tensorflow.compat.v1'
REPLACE_RULES[
    'from tensorflow.distribute.experimental import ParameterServerStrategy'] = \
    'from tensorflow.python.distribute.parameter_server_strategy ' \
    'import ParameterServerStrategyV1 as ParameterServerStrategy'
REPLACE_RULES[
    'from tensorflow.contrib.distribute import DistributeConfig'] = \
    'from tensorflow.python.distribute.distribute_config import DistributeConfig'
REPLACE_RULES['from tensorflow.python.keras import backend'] = 'from keras import backend'

REMOVED_LINES.append("@ops.RegisterGradient('HcomAllReduce')")
REMOVED_LINES.append("@ops.RegisterGradient(\"FastGelu\")")
REMOVED_LINES.append("from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer")
REMOVED_LINES.append("from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager")
REMOVED_LINES.append("from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager")

REGEXP_RULES['import npu_bridge$'] = 'from npu_device.compat import v1 as npu_bridge'

FILE_REPLACED = (
    'helper/helper.py',
)

FILE_REMOVED = (
    'estimator/npu/npu_loss_scale_manager.py',
    'estimator/npu/npu_loss_scale_optimizer.py'
)

REPLACE_BASE = os.path.join(os.path.dirname(__file__), 'replacement')


def make_compat(root, absf):
    fn = os.path.relpath(absf, root)
    if fn in FILE_REMOVED:
        print('>>> File removed', flush=True)
        os.remove(absf)
    elif fn in FILE_REPLACED:
        print('>>> File replaced with', os.path.join('replacement', fn), flush=True)
        shutil.copyfile(os.path.join(REPLACE_BASE, fn), absf)
    else:
        with open(absf, 'r') as f:
            lines = f.readlines()
        with open(absf, 'w+') as f:
            for line in lines:
                origin_line = line
                if line.strip() in REMOVED_LINES:
                    line = ''
                else:
                    for k, v in REPLACE_RULES.items():
                        line = line.replace(k, v, 1)
                    for k, v in REGEXP_RULES.items():
                        line = re.sub(k, v, line)
                if origin_line != line:
                    if line:
                        f.writelines(line)
                        print('>>> Replace', origin_line, "with", line, flush=True)
                    else:
                        print('>>> Remove', origin_line, flush=True)
                else:
                    f.writelines(line)


def main():
    tree = sys.argv[1]
    for path, _, files in os.walk(tree):
        for fn in files:
            if fn.endswith('.py'):
                print("--- Processing", os.path.join(path, fn), '---', flush=True)
                make_compat(tree, os.path.join(path, fn))


if __name__ == '__main__':
    main()
