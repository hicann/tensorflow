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

add_library(ge_stub STATIC
    ${CMAKE_CURRENT_LIST_DIR}/../../stub/ge_stub.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../stub/parser_stub.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../stub/register_base_stub.cpp)

target_include_directories(ge_stub PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../../inc)

target_link_libraries(ge_libs INTERFACE ge_stub)
