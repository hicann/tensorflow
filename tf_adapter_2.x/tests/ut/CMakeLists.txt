# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/../../npu_device/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../../tf_adapter/ops/*.cc
    ${CMAKE_CURRENT_LIST_DIR}/testcases/*.cpp
)

add_executable(adapter2_ut_main ${TEST_SOURCES})

target_include_directories(adapter2_ut_main PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../inc)

target_compile_options(adapter2_ut_main PRIVATE
    -O2 -g --coverage -fprofile-arcs -ftest-coverage)

target_link_libraries(adapter2_ut_main PRIVATE
    tensorflow_libs gtest_main acl_stub aoe_stub ge_stub c_sec ${PYTHON_LD_LIBRARY} -lrt -ldl -lutil)

add_custom_target(adapter2_ut ALL
    COMMAND ./adapter2_ut_main
    COMMAND lcov -c -q -d . -o ut_addons.coverage
    COMMAND lcov -q -e ut_addons.coverage "*tf_adapter_2.x/npu_device*" -o ut_addons.coverage
    COMMAND lcov -o ut.coverage -a ${CMAKE_BINARY_DIR}/st/st.coverage -a ut_addons.coverage
    DEPENDS adapter2_ut_main adapter2_st
    VERBATIM)
