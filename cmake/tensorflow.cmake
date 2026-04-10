# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include(FetchContent)
if(TF_PKG_SERVER)
    set(_tf_url "${TF_PKG_SERVER}/libs/tensorflow/v1.15.0.zip")
    FetchContent_Declare(
        tensorflow
        URL ${_tf_url}
        URL_HASH MD5=0ad811d8d59f56ecc1a6032af997ec1d
    )
else()
  FetchContent_Declare(
        tensorflow
        URL https://github.com/tensorflow/tensorflow/archive/v1.15.0.zip
        URL_HASH MD5=0ad811d8d59f56ecc1a6032af997ec1d
  )
endif()
FetchContent_GetProperties(tensorflow)
if (NOT tensorflow_POPULATED)
    FetchContent_Populate(tensorflow)
    include_directories(${tensorflow_SOURCE_DIR})
endif ()
