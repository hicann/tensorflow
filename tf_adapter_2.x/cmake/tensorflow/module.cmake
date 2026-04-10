# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(tensorflow_libs INTERFACE)

if(DEFINED TF_INSTALLED_PATH)
    SET(TF_INCLUDE_DIR ${TF_INSTALLED_PATH}/include)
    target_link_libraries(tensorflow_libs INTERFACE
        ${TF_INSTALLED_PATH}/python/_pywrap_tensorflow_internal.so
        ${TF_INSTALLED_PATH}/libtensorflow_framework.so.2)
else()
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
        COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
    )

    set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

    add_library(tensorflow_framework SHARED ${fake_sources})
    set_target_properties(tensorflow_framework PROPERTIES VERSION 2)

    add_library(pywrap_tensorflow_internal SHARED ${fake_sources})
    set_target_properties(pywrap_tensorflow_internal PROPERTIES PREFIX _)

    SET(TF_INCLUDE_DIR ${TF2X_SOURCE_PATH}/include/)
    target_link_libraries(tensorflow_libs INTERFACE
        tensorflow_framework
        pywrap_tensorflow_internal)
endif()

include_directories(${TF_INCLUDE_DIR})
include_directories(${TF_INCLUDE_DIR}/external/farmhash_archive/src)
include_directories(${TF_INCLUDE_DIR}/external/pybind11/_virtual_includes/pybind11)
