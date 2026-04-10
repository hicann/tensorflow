# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(ge_libs INTERFACE)

if(DEFINED ASCEND_INSTALLED_PATH)
    include_directories(${ASCEND_INSTALLED_PATH}/include)
    include_directories(${ASCEND_INSTALLED_PATH}/include/external)
    include_directories(${ASCEND_INSTALLED_PATH}/pkg_inc)
    target_link_libraries(ge_libs INTERFACE
        ${ASCEND_INSTALLED_PATH}/lib64/libge_runner.so
        ${ASCEND_INSTALLED_PATH}/lib64/libfmk_parser.so)
else()
    include_directories(${ASCEND_CI_BUILD_DIR}/graphengine/inc)
    include_directories(${ASCEND_CI_BUILD_DIR}/graphengine/inc/external)
    include_directories(${ASCEND_CI_BUILD_DIR}/metadef/inc)
    include_directories(${ASCEND_CI_BUILD_DIR}/metadef/pkg_inc)
    include_directories(${ASCEND_CI_BUILD_DIR}/metadef/inc/graph)
    include_directories(${ASCEND_CI_BUILD_DIR}/metadef/inc/external)
    include_directories(${ASCEND_CI_BUILD_DIR}/metadef/inc/external/graph)
    include_directories(${ASCEND_CI_BUILD_DIR}/air/inc/graph_metadef)
    include_directories(${ASCEND_CI_BUILD_DIR}/air/inc/graph_metadef/graph)
    include_directories(${ASCEND_CI_BUILD_DIR}/air/inc/graph_metadef/external)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
        COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
    )

    set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

    add_library(ge_runner SHARED ${fake_sources})
    add_library(fmk_parser SHARED ${fake_sources})
    target_link_libraries(ge_libs INTERFACE
        ge_runner
        fmk_parser)
endif()
