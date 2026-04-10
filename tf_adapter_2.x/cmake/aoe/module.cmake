# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(aoe_libs INTERFACE)

if (DEFINED ASCEND_INSTALLED_PATH)
    include_directories(${ASCEND_INSTALLED_PATH}/include/aoe)
    target_link_libraries(aoe_libs INTERFACE
        ${ASCEND_INSTALLED_PATH}/tools/aoe/lib64/libaoe_tuning.so)
else ()
    include_directories(${ASCEND_CI_BUILD_DIR}/asl/aoetools/inc/aoe)
    include_directories(${ASCEND_CI_BUILD_DIR}/abl/slog/inc)
    include_directories(${ASCEND_CI_BUILD_DIR}/abl/msprof/inc)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
        COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
    )

    set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

    add_library(aoe_tuning SHARED ${fake_sources})
    target_link_libraries(aoe_libs INTERFACE
        aoe_tuning)
endif ()
