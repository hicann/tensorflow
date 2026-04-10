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
project(NpuCompatV1)

set(ADAPTER_ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)

if (DEFINED ASCEND_CI_BUILD_DIR)
    include_directories(${ASCEND_CI_BUILD_DIR}/abl/libc_sec/include)
    include_directories(${ADAPTER_ROOT}/inc/parser/inc)
    include_directories(${ADAPTER_ROOT}/inc/parser/inc/external)
else ()
    include(${ADAPTER_ROOT}/cmake/secure_c.cmake)
    include(${ADAPTER_ROOT}/cmake/nlohmann_json.cmake)
endif ()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ge_plugin_wrap.cxx
    COMMAND ${SWIG_BIN_PATH} -python -c++ -threads -o ${CMAKE_CURRENT_BINARY_DIR}/ge_plugin_wrap.cxx -outdir ${COMPAT_V1_DST} ${ADAPTER_ROOT}/tf_adapter/swig/ge_plugin.i
    DEPENDS ${ADAPTER_ROOT}/tf_adapter/swig/ge_plugin.i generate_compat_python
    VERBATIM
)

add_custom_target(generate_compat_python
    COMMAND cp -r ${ADAPTER_ROOT}/tf_adapter/python/npu_bridge/. ${COMPAT_V1_DST}
    COMMAND ${PYTHON_BIN_PATH} ${CMAKE_CURRENT_LIST_DIR}/upgrade_tf1_compat_tf2.py ${COMPAT_V1_DST}
)

add_custom_target(generate_compat_headers
    COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core/graph/
    COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core/kernels/data/
    COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/compiler/jit/graphcycles/
    COMMAND ln -sf ${TF_INCLUDE_DIR}/tensorflow/core/common_runtime/graph_constructor.h ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core/graph/graph_constructor.h
    COMMAND ln -sf ${TF_INCLUDE_DIR}/tensorflow/core/data/captured_function.h ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core/kernels/data/captured_function.h
    COMMAND ln -sf ${TF_INCLUDE_DIR}/tensorflow/core/framework/dataset.h ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core/kernels/data/dataset.h
    COMMAND ln -sf ${TF_INCLUDE_DIR}/tensorflow/compiler/xla/service/graphcycles/graphcycles.h ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/compiler/jit/graphcycles/graphcycles.h
)

file(GLOB_RECURSE SOURCES
    ${ADAPTER_ROOT}/tf_adapter/common/*.cc
    ${ADAPTER_ROOT}/tf_adapter/kernels/*.cc
    ${ADAPTER_ROOT}/tf_adapter/optimizers/*.cc
    ${ADAPTER_ROOT}/tf_adapter/util/*.cc
)
add_library(_tf_adapter SHARED ${SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/ge_plugin_wrap.cxx)
set_target_properties(_tf_adapter PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${COMPAT_V1_DST})

add_dependencies(_tf_adapter generate_compat_headers)

target_compile_definitions(_tf_adapter PRIVATE LOG_CPP TF_VERSION_TF2)

if (DEFINED ASCEND_CI_BUILD_DIR)
    target_include_directories(_tf_adapter PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${ADAPTER_ROOT}/
        ${ADAPTER_ROOT}/inc/
        ${ASCEND_CI_BUILD_DIR}/abl/mmpa/inc
        ${ASCEND_CI_BUILD_DIR}/asl/aoetools/inc/aoe
        ${ASCEND_CI_BUILD_DIR}/abl/slog/inc
        ${ASCEND_CI_BUILD_DIR}/abl/slog/inc/toolchain
        ${ASCEND_CI_BUILD_DIR}/ace/npuruntime/inc
        ${ASCEND_CI_BUILD_DIR}/abl/msprof/inc
        ${ASCEND_CI_BUILD_DIR}/parser/inc/external
        ${ASCEND_CI_BUILD_DIR}/runtime/pkg_inc
    )
else ()
    add_dependencies(_tf_adapter download-acl_compat)
    include(${ADAPTER_ROOT}/cmake/acl_compat.cmake)
    target_include_directories(_tf_adapter PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${ADAPTER_ROOT}/
        ${ADAPTER_ROOT}/inc/
        ${ASCEND_INSTALLED_PATH}/include
        ${ASCEND_INSTALLED_PATH}/include/aoe
        ${ASCEND_INSTALLED_PATH}/pkg_inc/runtime
        ${ASCEND_INSTALLED_PATH}/pkg_inc/profiling
        ${acl_compat_dir}/include
    )
endif ()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
    COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
)

set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

add_library(fmk_onnx_parser SHARED ${fake_sources})
add_library(c_sec SHARED ${fake_sources})
add_library(datatransfer SHARED ${fake_sources})
add_library(indextransform SHARED ${fake_sources})

add_library(compat_v1_addon_libs INTERFACE)
target_link_libraries(compat_v1_addon_libs INTERFACE
    fmk_onnx_parser
    c_sec
    datatransfer
    indextransform
)

target_link_libraries(_tf_adapter PRIVATE
    tensorflow_libs
    ge_libs
    acl_libs
    compat_v1_addon_libs
)
