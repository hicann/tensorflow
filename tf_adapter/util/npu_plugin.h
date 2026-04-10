/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_NPU_PLUGIN_H_
#define TENSORFLOW_NPU_PLUGIN_H_

#include <map>
#include <string>
#include "ge_common/ge_api_types.h"
#include "ge_plugin.h"
#include "ge/ge_api_wrapper.h"

const char *const OP_DEBUG_LEVEL = "ge.opDebugLevel";
const char *const OPTION_EXEC_ENABLE_SCOPE_FUSION_PASSES = ge::OPTION_EXEC_ENABLE_SCOPE_FUSION_PASSES;
const char *const OPTION_EXEC_PROFILING_MODE = ge::OPTION_EXEC_PROFILING_MODE;
const char *const OPTION_EXEC_PROFILING_OPTIONS = ge::OPTION_EXEC_PROFILING_OPTIONS;
const char *const OPTION_GRAPH_RUN_MODE = ge::OPTION_GRAPH_RUN_MODE;
const char* const OPTION_EXEC_HCCL_FLAG = ge::OPTION_EXEC_HCCL_FLAG;
const char* const OPTION_EXEC_PROFILING_FPPONIT_OPTIONS = ge::OPTION_EXEC_PROFILING_FPPONIT_OPTIONS;
const char* const OPTION_EXEC_PROFILING_BPPONIT_OPTIONS = ge::OPTION_EXEC_PROFILING_BPPONIT_OPTIONS;

void PluginInit(std::map<std::string, std::string> &init_options);

void PluginFinalize();

void NpuClose();

void AoeFinalizeIfNeed();

int32_t InitRdmaPool(size_t size);

int32_t RegistRdmaRemoteAddr(const std::vector<std::pair<uint64_t, uint64_t>> &var_info);

int32_t RdmaInitAndRegister(const std::vector<std::pair<uint64_t, uint64_t>> &var_info, size_t size);

int32_t GetVarAddrAndSize(const std::string &var_name, uint64_t &base_addr, uint64_t &var_size);

int32_t MallocSharedMem(const std::string &var_name, const std::vector<int64_t> &dims, ge::DataType data_type,
                        uint64_t &dev_addr, uint64_t &memory_size);

int32_t SetDeviceSatMode(uint32_t mode);

int32_t GetDeviceSatMode();
#endif  // TENSORFLOW_NPU_PLUGIN_H_
