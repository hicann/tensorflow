# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

project(runtime_stub)

file(GLOB_RECURSE SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "src/*.cc"
)

add_library(runtime_stub SHARED ${SRC_FILES})

target_compile_options(runtime_stub PRIVATE
    ${TFADAPTER_COMMON_COMPILE_OPTION}
)

target_include_directories(runtime_stub PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/
    ${TFADAPTER_DIR}/inc/
    ${TF_INSTALL_LIB_PATH}/include/
)

target_link_libraries(runtime_stub PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ${TFADAPTER_COMMON_LINK_OPTION}
)
