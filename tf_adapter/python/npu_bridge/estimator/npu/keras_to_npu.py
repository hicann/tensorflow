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

"""NPU estimator for keras model"""

from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow_estimator.python.estimator import run_config
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator


def model_to_npu_estimator(keras_model=None,
                           keras_model_path=None,
                           custom_objects=None,
                           model_dir=None,
                           checkpoint_format='saver',
                           config=None,
                           job_start_file=''):
    """Constructs an `NPUEstimator` instance from given keras model.
    """
    tf_estimator = model_to_estimator(keras_model=keras_model,
                                      keras_model_path=keras_model_path,
                                      custom_objects=custom_objects,
                                      model_dir=model_dir,
                                      config=run_config.RunConfig(model_dir=model_dir),
                                      checkpoint_format=checkpoint_format)

    estimator = NPUEstimator(model_fn=tf_estimator._model_fn,
                             model_dir=model_dir,
                             config=config,
                             job_start_file=job_start_file,
                             warm_start_from=tf_estimator._warm_start_settings)

    return estimator
