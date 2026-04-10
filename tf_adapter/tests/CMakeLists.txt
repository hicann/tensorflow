# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

project(tests CXX C)

set(TFADAPTER_COMMON_COMPILE_OPTION -g -O0 -fsanitize=address -fsanitize=leak -fsanitize-recover=address)
set(TFADAPTER_COMMON_LINK_OPTION -fsanitize=address -fsanitize=leak -fsanitize-recover=address)

string(REPLACE "-s " "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE "-s " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include(${TFADAPTER_DIR}/cmake/tests/gtest.cmake)
include(${TFADAPTER_DIR}/cmake/tests/securec.cmake)
include(${TFADAPTER_DIR}/cmake/tests/intf_pub_linux.cmake)

execute_process(COMMAND ${PYTHON_BIN_PATH} -c "import tensorflow as tf; print(tf.sysconfig.get_lib(), end='')"
    OUTPUT_VARIABLE TF_INSTALL_LIB_PATH)

execute_process(COMMAND ${PYTHON_BIN_PATH} -c "from distutils import sysconfig; import os.path as op; \
    v=sysconfig.get_config_vars(); print(op.join(v['LIBDIR'], v['LDLIBRARY']))"
    OUTPUT_VARIABLE PYTHON_LIB_PATH)
message("python lib path:${PYTHON_LIB_PATH}")

add_subdirectory(depends/mmpa)
add_subdirectory(depends/indextransform)
add_subdirectory(depends/alog)
add_subdirectory(depends/datatransfer)
add_subdirectory(depends/ge_runner)
add_subdirectory(depends/aoe)
add_subdirectory(depends/runtime)
add_subdirectory(depends/ascendcl)

if (ENABLE_TFADAPTER_UT)
    add_subdirectory(ut)
endif()

if (ENABLE_TFADAPTER_ST)
    add_subdirectory(st)
endif()
