# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(acl_libs INTERFACE)

if(DEFINED ASCEND_INSTALLED_PATH)
    include_directories(${ASCEND_INSTALLED_PATH}/include)
    include_directories(${ASCEND_INSTALLED_PATH}/include/acl/error_codes)
    include_directories(${ASCEND_INSTALLED_PATH}/include)
    include_directories(${ASCEND_INSTALLED_PATH}/include/acl)
    target_link_libraries(acl_libs INTERFACE
        ${ASCEND_INSTALLED_PATH}/lib64/libascendcl.so
        ${ASCEND_INSTALLED_PATH}/lib64/libacl_tdt_channel.so
        ${ASCEND_INSTALLED_PATH}/lib64/libacl_op_compiler.so)
else()
    include_directories(${ASCEND_CI_BUILD_DIR}/inc/external)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
        COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
    )

    set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

    add_library(ascendcl SHARED ${fake_sources})
    add_library(acl_op_compiler SHARED ${fake_sources})
    add_library(acl_tdt_channel SHARED ${fake_sources})
    target_link_libraries(acl_libs INTERFACE
        ascendcl
        acl_op_compiler
        acl_tdt_channel)
endif()
