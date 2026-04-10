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
project(NpuConvertTool)

get_filename_component(ADAPTER_V2_DIR ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)
get_filename_component(PROJECT_ROOT ${ADAPTER_V2_DIR} DIRECTORY)
set(CONVERT_TOOL_DIST ${CMAKE_BINARY_DIR}/dist/python/npu_device/convert_tf2npu)

add_custom_target(convert_tool_v2 ALL
    COMMAND cp -r ${PROJECT_ROOT}/convert_tf2npu ${CMAKE_BINARY_DIR}/dist/python/npu_device
    && rm -f ${CONVERT_TOOL_DIST}/mappings/ast.json
    && rm -f ${CONVERT_TOOL_DIST}/config.py
    && rm -f ${CONVERT_TOOL_DIST}/tf1.15_api_support_list.xlsx
    && rm -f ${CONVERT_TOOL_DIST}/main_win.py
    && mv ${CONVERT_TOOL_DIST}/ast_impl.py ${CONVERT_TOOL_DIST}/ast_impl_v1.py
    && mv ${CONVERT_TOOL_DIST}/v2/mappings/ast_v2.json  ${CONVERT_TOOL_DIST}/mappings/ast.json
    && mv ${CONVERT_TOOL_DIST}/v2/ast_impl_v2.py  ${CONVERT_TOOL_DIST}/ast_impl.py
    && mv ${CONVERT_TOOL_DIST}/v2/config_v2.py  ${CONVERT_TOOL_DIST}/config.py
    && mv ${CONVERT_TOOL_DIST}/v2/tf_func_def_v2.py ${CONVERT_TOOL_DIST}/tf_func_def_v2.py
    && mv ${CONVERT_TOOL_DIST}/v2/TF2.6_api_support_list.xlsx ${CONVERT_TOOL_DIST}/TF2.6_api_support_list.xlsx
    && rm -r ${CONVERT_TOOL_DIST}/v2
    VERBATIM)
