# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)
project(NpuDevice)

include_directories(${CMAKE_CURRENT_LIST_DIR}/core)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
foreach (CPP_SOURCE ${SOURCES})
    file(RELATIVE_PATH RELATIVE_CPP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR} ${CPP_SOURCE})
    set_property(SOURCE ${CPP_SOURCE} PROPERTY COMPILE_DEFINITIONS __FILE__=\"${RELATIVE_CPP_SOURCE}\")
endforeach (CPP_SOURCE)

add_library(_npu_device_backends SHARED ${SOURCES})
set_target_properties(_npu_device_backends PROPERTIES PREFIX "")

target_link_libraries(_npu_device_backends PRIVATE
    tensorflow_libs
    c_sec
    ge_libs
    acl_libs)

if (DEFINED ASCEND_CI_BUILD_DIR)
    target_include_directories(_npu_device_backends PRIVATE
        ${ASCEND_CI_BUILD_DIR}/runtime/pkg_inc/runtime
        ${ASCEND_CI_BUILD_DIR}/runtime/pkg_inc/profiling
        ${CMAKE_CURRENT_LIST_DIR}/../../inc/)
else ()
    target_include_directories(_npu_device_backends PRIVATE
        ${ASCEND_INSTALLED_PATH}/pkg_inc/runtime
        ${ASCEND_INSTALLED_PATH}/pkg_inc/profiling
        ${CMAKE_CURRENT_LIST_DIR}/../../inc/)
endif ()
