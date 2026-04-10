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

from npu_bridge.helper import helper
gen_npu_image_ops = helper.get_gen_ops()


def decode_and_resize_jpeg(image, size):
    """
    Decode and resize JPEG-encoded image.

    :param image: The JPEG-encoded image.
    :param size: A 1-D int32 Tensor of 2 elements:
    new_height, new_width. The new size for the images.
    :return Resized image, a 3-D uint8 tensor:
    [new_height, new_width, channel=3] .
    """
    return gen_npu_image_ops.decode_and_resize_jpeg(image, size)


def decode_and_crop_and_resize_jpeg(image, crop_size, size):
    """
    Decode, crop and resize JPEG-encoded image.

    :param image: The JPEG-encoded image.
    :param crop_size: A 1-D int32 Tensor of 4 elements:
    [y_min, x_min, crop_height, crop_width].
    :param size: A 1-D int32 Tensor of 2 elements:
    new_height, new_width. The new size for the images.
    :return Cropped and Resized image, a 3-D uint8 tensor:
    [new_height, new_width, channel=3].
    """
    return gen_npu_image_ops.decode_and_crop_and_resize_jpeg(image, crop_size, size)
