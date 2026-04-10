# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(COPY ${CMAKE_CURRENT_LIST_DIR}/../../python DESTINATION ${CMAKE_BINARY_DIR}/dist)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/adapter2_st.py DESTINATION ${CMAKE_BINARY_DIR}/dist/python)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/adapter2_aoe_st.py DESTINATION ${CMAKE_BINARY_DIR}/dist/python)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/adapter2_jit_compile_st.py DESTINATION ${CMAKE_BINARY_DIR}/dist/python)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/adapter2_compile_dynamic_mode_st.py DESTINATION ${CMAKE_BINARY_DIR}/dist/python)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/adapter2_shape_generalization_mode_st.py DESTINATION ${CMAKE_BINARY_DIR}/dist/python)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/adapter2_shape_generalization_mode_ignore_st.py DESTINATION ${CMAKE_BINARY_DIR}/dist/python)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/../stub/hccl DESTINATION ${CMAKE_BINARY_DIR}/dist/python)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/adapter2_options.py DESTINATION ${CMAKE_BINARY_DIR}/dist/python)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/dist/python/npu_device)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../npu_device npu_device)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../npu_ops npu_ops)

add_custom_target(adapter2_st ALL
    COMMAND export NPU_DEBUG=true && export NPU_DUMP_GRAPH=true && cd ${CMAKE_BINARY_DIR}/dist/python/ && ${PYTHON_BIN_PATH} -m unittest adapter2_st
    COMMAND lcov -c -q -d . -o st_p1.coverage && lcov -q -e st_p1.coverage "*npu_device*" -o st_p1.coverage
    COMMAND export GE_USE_STATIC_MEMORY=1 && cd ${CMAKE_BINARY_DIR}/dist/python/ && ${PYTHON_BIN_PATH} -m unittest adapter2_st.Adapter2St_EnvGeStaticMemory
    COMMAND lcov -c -q -d . -o st_p2.coverage && lcov -q -e st_p2.coverage "*npu_device*" -o st_p2.coverage
    COMMAND cd ${CMAKE_BINARY_DIR}/dist/python/ && ${PYTHON_BIN_PATH} -m unittest adapter2_aoe_st
    COMMAND lcov -c -q -d . -o st_p3.coverage && lcov -q -e st_p3.coverage "*npu_device*" -o st_p3.coverage
    COMMAND export NPU_DEBUG=true && export NPU_DUMP_GRAPH=true && cd ${CMAKE_BINARY_DIR}/dist/python/ && ${PYTHON_BIN_PATH} -m unittest adapter2_jit_compile_st
    COMMAND lcov -c -q -d . -o st_p4.coverage && lcov -q -e st_p4.coverage "*npu_device*" -o st_p4.coverage
    COMMAND cd ${CMAKE_BINARY_DIR}/dist/python/ && ${PYTHON_BIN_PATH} -m unittest adapter2_options.Adapter2Options
    COMMAND lcov -c -q -d . -o st_p5.coverage && lcov -q -e st_p5.coverage "*npu_device*" -o st_p5.coverage
    COMMAND export NPU_DEBUG=true && export NPU_DUMP_GRAPH=true && cd ${CMAKE_BINARY_DIR}/dist/python/ && ${PYTHON_BIN_PATH} -m unittest adapter2_compile_dynamic_mode_st
    COMMAND lcov -c -q -d . -o st_p6.coverage && lcov -q -e st_p6.coverage "*npu_device*" -o st_p6.coverage
    COMMAND export NPU_DEBUG=true && export NPU_DUMP_GRAPH=true && cd ${CMAKE_BINARY_DIR}/dist/python/ && ${PYTHON_BIN_PATH} -m unittest adapter2_shape_generalization_mode_st
    COMMAND lcov -c -q -d . -o st_p7.coverage && lcov -q -e st_p7.coverage "*npu_device*" -o st_p7.coverage
    COMMAND export NPU_DEBUG=true && export NPU_DUMP_GRAPH=true && cd ${CMAKE_BINARY_DIR}/dist/python/ && ${PYTHON_BIN_PATH} -m unittest adapter2_shape_generalization_mode_ignore_st
    COMMAND lcov -c -q -d . -o st_p8.coverage && lcov -q -e st_p7.coverage "*npu_device*" -o st_p8.coverage
    COMMAND lcov -o st.coverage -a st_p1.coverage -a st_p2.coverage -a st_p3.coverage -a st_p4.coverage -a st_p5.coverage -a st_p6.coverage -a st_p7.coverage -a st_p8.coverage
    DEPENDS _npu_ops _npu_device_backends aoe_tuning
    VERBATIM)
