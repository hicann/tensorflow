/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include <mmpa/mmpa_api.h>
#include "nlohmann/json.hpp"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/framework/cancellation.h"
#include "graph/types.h"
#include "ge/ge_api.h"
#include "tdt/tdt_host_interface.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/npu_attrs.h"
#include "acl/acl_rt.h"
#include "tf_adapter/util/npu_plugin.h"
#include "ge/ge_api_wrapper.h"
#include "external/aoe.h"
#include "external/aoe_errcodes.h"

using AoeStatus = int32_t;
using AoeFinalizeFunc = AoeStatus (*)();
using json = nlohmann::json;

using namespace tdt;
using namespace tensorflow;
namespace {
const int kFatalSleepTime = 3000;
const int64 kInvalidRankSize = -1;
const int64 kDefaultRankSize = 1;
// 仅ge错误码可用
inline string ToString(ge::Status status) {
  return ::ge::StatusFactory::Instance()->GetErrDescV2(status).GetString();
}
void GeFinalize() {
  // 先等待可能的异步初始化结束
  (void) GePlugin::GetInstance()->GetInitStatus();
  // ge finalize
  ge::Status status = ge::GEFinalize();
  if (status != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] GE finalize failed, ret : " << ToString(status);
    LOG(ERROR) << "[GePlugin] GE finalize failed, ret : " << ToString(status) << std::endl
               << "Error Message is : " << std::endl << ge::GEGetErrorMsgV2().GetString();
  }

  // parser finalize
  ge::Status status_parser = GeApiWrapper_ParserFinalize();
  if (status_parser != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] Parser finalize failed, ret : " << ToString(status_parser);
    LOG(ERROR) << "[GePlugin] Parser finalize failed, ret : " << ToString(status_parser);
  }
}

void SetOptionNameMap(json &option_name_map) {
  option_name_map.emplace(ge::OPTION_GRAPH_RUN_MODE, "graph_run_mode");
  option_name_map.emplace(ge::GRAPH_MEMORY_MAX_SIZE, "graph_memory_max_size");
  option_name_map.emplace(ge::VARIABLE_MEMORY_MAX_SIZE, "variable_memory_max_size");
  option_name_map.emplace("ge.exec.variable_acc", "variable_format_optimize");
  option_name_map.emplace(ge::OPTION_EXEC_ENABLE_SCOPE_FUSION_PASSES, "enable_scope_fusion_passes");
  option_name_map.emplace(ge::FUSION_SWITCH_FILE, "fusion_switch_file");
  option_name_map.emplace(ge::PRECISION_MODE, "precision_mode");
  option_name_map.emplace(ge::PRECISION_MODE_V2, "precision_mode_v2");
  option_name_map.emplace(ge::OP_SELECT_IMPL_MODE, "op_select_implmode");
  option_name_map.emplace(ge::OPTYPELIST_FOR_IMPLMODE, "optypelist_for_implmode");
  option_name_map.emplace(ge::OP_COMPILER_CACHE_MODE, "op_compiler_cache_mode");
  option_name_map.emplace(ge::OP_COMPILER_CACHE_DIR, "op_compiler_cache_dir");
  option_name_map.emplace(ge::STREAM_MAX_PARALLEL_NUM, "stream_max_parallel_num");
  option_name_map.emplace(ge::AC_PARALLEL_ENABLE, "ac_parallel_enable");
  option_name_map.emplace(ge::QUANT_DUMPABLE, "quant_dumpable");
  option_name_map.emplace(ge::HCOM_PARALLEL, "hcom_parallel");
  option_name_map.emplace(ge::HCOM_MULTI_MODE, "hcom_multi_mode");
  option_name_map.emplace(ge::OPTION_EXEC_ENABLE_TAILING_OPTIMIZATION, "is_tailing_optimization");
  option_name_map.emplace(ge::OP_DEBUG_LEVEL, "op_debug_level");
  option_name_map.emplace(ge::DEBUG_DIR, "debug_dir");
  option_name_map.emplace(ge::MODIFY_MIXLIST, "modify_mixlist");
  option_name_map.emplace(ge::OPTION_EXEC_ENABLE_EXCEPTION_DUMP, "enable_exception_dump");
  option_name_map.emplace(ge::OPTION_EXEC_ENABLE_DUMP, "enable_dump");
  option_name_map.emplace(ge::OPTION_EXEC_DUMP_PATH, "dump_path");
  option_name_map.emplace(ge::OPTION_EXEC_DUMP_STEP, "dump_step");
  option_name_map.emplace(ge::OPTION_EXEC_DUMP_MODE, "dump_mode");
  option_name_map.emplace(ge::OPTION_EXEC_ENABLE_DUMP_DEBUG, "enable_dump_debug");
  option_name_map.emplace(ge::OPTION_EXEC_DUMP_DEBUG_MODE, "dump_debug_mode");
  option_name_map.emplace(ge::OPTION_EXEC_PROFILING_MODE, "enable_profiling");
  option_name_map.emplace(ge::OPTION_EXEC_PROFILING_OPTIONS, "profiling_options");
  option_name_map.emplace("ge.jobType", "aoe_mode");
  option_name_map.emplace("ge.tuningPath", "work_path");
  option_name_map.emplace(ge::INPUT_SHAPE, "input_shape");
  option_name_map.emplace(ge::DYNAMIC_NODE_TYPE, "dynamic_node_type");
  option_name_map.emplace(ge::kDynamicDims, "dynamic_dims");
  option_name_map.emplace(ge::ENABLE_SMALL_CHANNEL, "enable_small_channel");
  option_name_map.emplace("ge.deterministic", "deterministic");
  option_name_map.emplace("ge.exec.op_precision_mode", "op_precision_mode");
  option_name_map.emplace("ge.exec.graphExecTimeout", "graph_exec_timeout");
  option_name_map.emplace(ge::OPTION_EXEC_LOGICAL_DEVICE_CLUSTER_DEPLOY_MODE, "logical_device_cluster_deploy_mode");
  option_name_map.emplace(ge::OPTION_EXEC_LOGICAL_DEVICE_ID, "logical_device_id");
  option_name_map.emplace("ge.exec.modelDeployMode", "model_deploy_mode");
  option_name_map.emplace("ge.exec.modelDeployDevicelist", "model_deploy_devicelist");
  option_name_map.emplace("ge.topoSortingMode", "topo_sorting_mode");
  option_name_map.emplace("ge.exec.overflow", "overflow_flag");
  option_name_map.emplace("ge.insertOpFile", "insert_op_file");
  option_name_map.emplace("ge.customizeDtypes", "customize_dtypes");
  option_name_map.emplace("ge.exec.dumpData", "dump_data");
  option_name_map.emplace("ge.exec.dumpLayer", "dump_layer");
  option_name_map.emplace("ge.aoe_config_file", "aoe_config_file");
  option_name_map.emplace("ge.externalWeight", "external_weight");
  option_name_map.emplace("ge.autoTuneMode", "auto_tune_mode");
  option_name_map.emplace("ge.deviceType", "device_type");
  option_name_map.emplace("ge.exec.hcclExecuteTimeOut", "hccl_timeout");
  option_name_map.emplace("ge.exec.opWaitTimeout", "op_wait_timeout");
  option_name_map.emplace("ge.exec.opExecuteTimeout", "op_execute_timeout");
  option_name_map.emplace("op_debug_config", "op_debug_config");
  option_name_map.emplace("ge.exec.staticMemoryPolicy", "static_memory_policy");
  option_name_map.emplace("ge.variableUse1gHugePage", "variable_use_1g_huge_page");
  option_name_map.emplace("ge.socVersion", "soc_config");
  option_name_map.emplace(ge::OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "dynamic_graph_execute_mode");
  option_name_map.emplace(ge::OPTION_EXEC_DYNAMIC_INPUT, "dynamic_input");
  option_name_map.emplace(ge::AICORE_NUM, "aicore_num");
  option_name_map.emplace("ge.inputBatchCpy", "input_batch_cpy");
  option_name_map.emplace(ge::OPTION_ALL_TENSOR_NOT_EMPTY, "all_tensor_not_empty");
  option_name_map.emplace("ge.autoMultistreamParallelMode", "auto_multistream_parallel_mode");
  option_name_map.emplace("ge.oo.level", "oo_level");
  option_name_map.emplace("ge.optimizationSwitch", "optimization_switch");
}
}  // namespace

GePlugin::GePlugin()

    : device_id_(0), isInit_(false), isGlobal_(false) {
  ADP_LOG(INFO) << "[GePlugin] New constructor";
}

GePlugin::~GePlugin() {
  ADP_LOG(INFO) << "[GePlugin] Destroy constructor begin";
  Finalize();
  ADP_LOG(INFO) << "[GePlugin] Destroy constructor end";
}

/**
 * @brief: get instance
 */
GePlugin *GePlugin::GetInstance() {
  static GePlugin instance;
  return &instance;
}

void GePlugin::Init(std::map<std::string, std::string> &init_options, const bool is_global, const bool is_async) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (isInit_) {
    ADP_LOG(INFO) << "[GePlugin] Ge has already initialized";
    return;
  }
  ADP_LOG(INFO) << "Init options: ";
  NpuAttrs::LogOptions(init_options);
  init_options_ = init_options;

  std::string enable_hf32_execution;
  (void) ReadStringFromEnvVar("ENABLE_HF32_EXECUTION", "", &enable_hf32_execution);
  if (!enable_hf32_execution.empty()) {
    init_options["ge.exec.allow_hf32"] = enable_hf32_execution;
    ADP_LOG(INFO) << "[GePlugin] allow_hf32 : " << init_options["ge.exec.allow_hf32"];
  }

  std::string tf_config;
  (void) ReadStringFromEnvVar("TF_CONFIG", "", &tf_config);
  int exec_hccl_flag = 1;
  if (!tf_config.empty()) {
    json config_info;
    try {
      config_info = json::parse(tf_config);
    } catch (json::exception &e) {
      ADP_LOG(WARNING) << "[GePlugin] Failed to convert TF_CONFIG info from string to json ,reason: " << e.what();
      LOG(WARNING) << "[GePlugin] Failed to convert TF_CONFIG info from string to json ,reason: " << e.what();
    }
    if (config_info.is_object()) {
      if (config_info["task"]["type"] == "ps") {
        ADP_LOG(INFO) << "The ps process does not need to be initialized";
        return;
      }
      if (config_info["task"]["type"] == "evaluator") {
        exec_hccl_flag = 0;
      }
    }
  }
  init_options[OPTION_EXEC_HCCL_FLAG] = std::to_string(exec_hccl_flag);

  ADP_LOG(INFO) << "[GePlugin] graph run mode : " << init_options[ge::OPTION_GRAPH_RUN_MODE];

  Status s = GetEnvDeviceID(device_id_);
  if (!s.ok()) {
    ADP_LOG(FATAL) << s.error_message();
    LOG(FATAL) << s.error_message();
  }
  init_options[ge::OPTION_EXEC_DEVICE_ID] = std::to_string(device_id_);
  ADP_LOG(INFO) << "[GePlugin] device id : " << init_options[ge::OPTION_EXEC_DEVICE_ID];

  std::string env_job_id;
  (void) ReadStringFromEnvVar("JOB_ID", "", &env_job_id);
  if (!env_job_id.empty()) {
    init_options[ge::OPTION_EXEC_JOB_ID] = env_job_id;
  } else {
    ADP_LOG(WARNING) << "[GePlugin] can not find Environment variable : JOB_ID";
    LOG(WARNING) << "[GePlugin] can not find Environment variable : JOB_ID";
  }

  std::string cm_chief_ip;
  (void) ReadStringFromEnvVar("CM_CHIEF_IP", "", &cm_chief_ip);
  (void) ReadInt64FromEnvVar("CM_WORKER_SIZE", kInvalidRankSize, &work_size_num);
  std::string env_rank_table_file;
  (void) ReadStringFromEnvVar("RANK_TABLE_FILE", "", &env_rank_table_file);
  (void) ReadInt64FromEnvVar("RANK_SIZE", kInvalidRankSize, &rank_size_num);
  if (!cm_chief_ip.empty() && !env_rank_table_file.empty()) {
    ADP_LOG(ERROR) << "[GePlugin] CM_CHIEF_IP and RANK_TABLE_FILE cannot be configured at the same time.";
    LOG(ERROR) << "[GePlugin] CM_CHIEF_IP and RANK_TABLE_FILE cannot be configured at the same time.";
  } else if (!cm_chief_ip.empty()) {
    SetCmChiefWorkSizeEnv(init_options, cm_chief_ip);
  } else if (!env_rank_table_file.empty()) {
    SetRankTableFileEnv(init_options, env_rank_table_file);
  } else {
    ADP_LOG(INFO) << "[GePlugin] CM_CHIEF_IP and RANK_TABLE_FILE are all not be configured.";
  }

  std::string cluster_info;
  (void) ReadStringFromEnvVar("HELP_CLUSTER", "", &cluster_info);
  if (!cluster_info.empty()) {
    is_use_hcom = true;
  }

  init_options[ge::OPTION_EXEC_IS_USEHCOM] = std::to_string(is_use_hcom);

  // is use hcom configuration
  ADP_LOG(INFO) << "[GePlugin] is_usehcom : " << init_options[ge::OPTION_EXEC_IS_USEHCOM]
                << ", deploy_mode :" << init_options[ge::OPTION_EXEC_DEPLOY_MODE];

  // profiling configuration
  ADP_LOG(INFO) << "[GePlugin] profiling_mode : " << init_options[ge::OPTION_EXEC_PROFILING_MODE]
                << ", profiling_options:" << init_options[ge::OPTION_EXEC_PROFILING_OPTIONS];

  // mix precision configuration
  if (init_options.find(ge::PRECISION_MODE) != init_options.end()) {
    ADP_LOG(INFO) << "[GePlugin] precision_mode : " << init_options[ge::PRECISION_MODE];
  }
  if (init_options.find("ge.exec.precision_mode_v2") != init_options.end()) {
    ADP_LOG(INFO) << "[GePlugin] precision_mode_v2 : " << init_options["ge.exec.precision_mode_v2"];
  }

  // debug configuration
  ADP_LOG(INFO) << "[GePlugin] op_debug_level : " << init_options[ge::OP_DEBUG_LEVEL];

  ADP_LOG(INFO) << "[GePlugin] ge.deterministic : " << init_options["ge.deterministic"];

  // scope fusion configuration
  ADP_LOG(INFO) << "[GePlugin] enable_scope_fusion_passes : "
                << init_options[ge::OPTION_EXEC_ENABLE_SCOPE_FUSION_PASSES];

  // exception dump configuration
  ADP_LOG(INFO) << "[GePlugin] enable_exception_dump : " << init_options["ge.exec.enable_exception_dump"];

  ADP_LOG(INFO) << "[GePlugin] job_id : " << init_options[ge::OPTION_EXEC_JOB_ID];

  ADP_LOG(INFO) << "[GePlugin] op_compiler_cache_mode : " << init_options["ge.op_compiler_cache_mode"];

  ADP_LOG(INFO) << "[GePlugin] op_compiler_cache_dir : " << init_options["ge.op_compiler_cache_dir"];

  ADP_LOG(INFO) << "[GePlugin] debugDir : " << init_options["ge.debugDir"];

  ADP_LOG(INFO) << "[GePlugin] hcom_multi_mode : " << init_options["ge.hcomMultiMode"];

  init_options["ge.fusionTensorSize"] = std::to_string(GetFusionTensorSize());
  ADP_LOG(INFO) << "[GePlugin] fusionTensorSize : " << init_options["ge.fusionTensorSize"];

  // aoe mode and work path
  if (!init_options["ge.jobType"].empty()) {
    init_options["ge.buildMode"] = "tuning";
  }
  ADP_LOG(INFO) << "[GePlugin] aoe mode : " << init_options["ge.jobType"]
                << ", work path : " << init_options["ge.tuningPath"]
                << ", distribute_config : " << init_options["distribute_config"];

  ADP_LOG(INFO) << "[GePlugin] fusion_switch_file :" << init_options["ge.fusionSwitchFile"];

  ADP_LOG(INFO) << "[GePlugin] op_precision_mode :" << init_options[ge::OP_PRECISION_MODE];

  ADP_LOG(INFO) << "[GePlugin] op_select_implmode :" << init_options[ge::OP_SELECT_IMPL_MODE];

  ADP_LOG(INFO) << "[GePlugin] optypelist_for_implmode :" << init_options[ge::OPTYPELIST_FOR_IMPLMODE];

  if (init_options.find("ge.exportCompileStat") != init_options.end()) {
    ADP_LOG(INFO) << "[GePlugin] export_compile_stat : " << init_options["ge.exportCompileStat"];
  }

  if (init_options.find("ge.aicoreNum") != init_options.end()) {
    ADP_LOG(INFO) << "[GePlugin] aicoreNum : " << init_options["ge.aicoreNum"];
  }

  if (init_options.find("ge.oo.constantFolding") != init_options.end()) {
    ADP_LOG(INFO) << "[GePlugin] oo_constant_folding : " << init_options["ge.oo.constantFolding"];
  }

  if (init_options.find("ge.inputBatchCpy") != init_options.end()) {
    ADP_LOG(INFO) << "[GePlugin] input_batch_cpy : " << init_options["ge.inputBatchCpy"];
  }

  if (init_options.find("ge.oo.level") != init_options.end()) {
    ADP_LOG(INFO) << "[GePlugin] oo_level : " << init_options["ge.oo.level"];
  }

  if (init_options.find("ge.optimizationSwitch") != init_options.end()) {
    ADP_LOG(INFO) << "[GePlugin] optimization_switch : " << init_options["ge.optimizationSwitch"];
  }

  bool tdt_uninit_env = false;
  (void) ReadBoolFromEnvVar("ASCEND_TDT_UNINIT", false, &tdt_uninit_env);
  if (!kIsHeterogeneous && !tdt_uninit_env) {
    // Open TsdClient first, then call GEInitialize
    ADP_LOG(INFO) << "[GePlugin] Open TsdClient and Init tdt host.";
    int32_t ret = tdt::TdtOutFeedInit(static_cast<uint32_t>(device_id_));
    if (ret != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kFatalSleepTime));
      ADP_LOG(FATAL) << "[GePlugin] Tdt host init failed, tdt error code : " << ret;
      LOG(FATAL) << "[GePlugin] Tdt host init failed, tdt error code : " << ret;
    }
  }

  json option_name_map;
  SetOptionNameMap(option_name_map);
  init_options["ge.optionNameMap"] = option_name_map.dump();

  // parser Initialize
  auto const init_options_ascend_string = ChangeStringToAscendString(init_options);
  ge::Status status_parser = GeApiWrapper_ParserInitialize(init_options_ascend_string);
  if (status_parser != ge::SUCCESS) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kFatalSleepTime));
    ADP_LOG(FATAL) << "[GePlugin] Initialize parser failed, ret : " << ToString(status_parser);
    LOG(FATAL) << "[GePlugin] Initialize parser failed, ret : " << ToString(status_parser);
  }
  ADP_LOG(INFO) << "[GePlugin] Initialize parser success.";
  if (is_async) {
    future_ = std::async(
                  std::launch::async,
                  [this](const std::map<std::string, std::string> &init_options) -> ge::Status {
                    const auto init_ascend_string_options = ChangeStringToAscendString(init_options);
                    const auto init_ret = ge::GEInitialize(init_ascend_string_options);
                    error_message_ = std::string(ge::GEGetErrorMsgV2().GetString());
                    warning_message_ = std::string(ge::GEGetWarningMsgV2().GetString());
                    return init_ret;
                  },
                  init_options)
                  .share();
  } else {
    ge::Status status = ge::GEInitialize(init_options_ascend_string);
    warning_message_ = std::string(ge::GEGetWarningMsgV2().GetString());
    if (!warning_message_.empty()) {
        LOG(WARNING) << "[GePlugin] GEInitialize warning message: " << std::endl
                     << warning_message_;
    }
    if (status != ge::SUCCESS) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kFatalSleepTime));
      ADP_LOG(FATAL) << "[GePlugin] Initialize ge failed, ret : " << ToString(status);
      error_message_ = std::string(ge::GEGetErrorMsgV2().GetString());
      LOG(FATAL) << "[GePlugin] Initialize ge failed, ret : " << ToString(status) << std::endl
                 << "Error Message is : " << std::endl
                 << error_message_;
    }
    ADP_LOG(INFO) << "[GePlugin] Initialize ge success.";
  }
  GeApiWrapper_SetDomiContextTrainFlag(true);
  isInit_ = true;
  isGlobal_ = is_global;
}

void GePlugin::SetRankTableFileEnv(std::map<std::string, std::string> &init_options, std::string &rankTableFile) {
  rank_size_num = (rank_size_num == kInvalidRankSize) ? kDefaultRankSize : rank_size_num;
  if (rank_size_num > UINT32_MAX) {
    rank_size_num = UINT32_MAX;
    ADP_LOG(WARNING) << "[GePlugin] RANK_SIZE is larger than UINT32_MAX, set to UINT32_MAX.";
    LOG(WARNING) << "[GePlugin] RANK_SIZE is larger than UINT32_MAX, set to UINT32_MAX.";
  }
  if (!rankTableFile.empty() && (rank_size_num > 0) && (work_size_num == kInvalidRankSize)) {
    ADP_LOG(INFO) << "[GePlugin] env RANK_TABLE_FILE:" << rankTableFile;
    is_use_hcom = true;
    init_options[ge::OPTION_EXEC_RANK_TABLE_FILE] = rankTableFile;
    std::string env_pod_name;
    (void) ReadStringFromEnvVar("POD_NAME", "", &env_pod_name);
    if (!env_pod_name.empty()) {
      init_options[ge::OPTION_EXEC_POD_NAME] = env_pod_name;
    }
    std::string env_rank_id;
    (void) ReadStringFromEnvVar("RANK_ID", "", &env_rank_id);
    if (!env_rank_id.empty()) {
      ADP_LOG(INFO) << "[GePlugin] env RANK_ID:" << env_rank_id;
      init_options[ge::OPTION_EXEC_RANK_ID] = env_rank_id;
    }
  }
}

void GePlugin::SetCmChiefWorkSizeEnv(std::map<std::string, std::string> &init_options, std::string &cmChiefIp) {
  std::string cm_chief_port;
  (void) ReadStringFromEnvVar("CM_CHIEF_PORT", "", &cm_chief_port);
  std::string cm_chief_device;
  (void) ReadStringFromEnvVar("CM_CHIEF_DEVICE", "", &cm_chief_device);
  std::string cm_worker_ip;
  (void) ReadStringFromEnvVar("CM_WORKER_IP", "", &cm_worker_ip);
  std::string cm_worker_size;
  (void) ReadStringFromEnvVar("CM_WORKER_SIZE", "", &cm_worker_size);
  work_size_num = (work_size_num == kInvalidRankSize) ? kDefaultRankSize : work_size_num;
  if (work_size_num > UINT32_MAX) {
    work_size_num = UINT32_MAX;
    ADP_LOG(WARNING) << "[GePlugin] RANK_SIZE is larger than UINT32_MAX, set to UINT32_MAX.";
    LOG(WARNING) << "[GePlugin] RANK_SIZE is larger than UINT32_MAX, set to UINT32_MAX.";
  }
  if (!cmChiefIp.empty() && !cm_chief_port.empty() && !cm_chief_device.empty() && (work_size_num > 0) &&
      (rank_size_num == kInvalidRankSize)) {
    is_use_hcom = true;
    init_options["ge.cmChiefIp"] = cmChiefIp;
    init_options["ge.cmChiefPort"] = cm_chief_port;
    init_options["ge.cmChiefWorkerDevice"] = cm_chief_device;
    if (!cm_worker_ip.empty()) {
      init_options["ge.cmWorkerIp"] = cm_worker_ip;
    }
    if (!cm_worker_size.empty()) {
      init_options["ge.cmWorkerSize"] = cm_worker_size;
    }
  }
}

std::map<std::string, std::string> GePlugin::GetInitOptions() {
  return init_options_;
}

uint64_t GePlugin::GetFusionTensorSize() const {
  const int64 fusion_tensor_size_default = 524288000;
  int64 fusion_tensor_size = fusion_tensor_size_default;
  Status s = ReadInt64FromEnvVar("FUSION_TENSOR_SIZE", fusion_tensor_size_default, &fusion_tensor_size);
  if (s.ok() && fusion_tensor_size >= 0) {
    return static_cast<uint64_t>(fusion_tensor_size);
  }
  return static_cast<uint64_t>(fusion_tensor_size_default);
}

void GePlugin::Finalize() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!isInit_) {
    ADP_LOG(INFO) << "[GePlugin] Ge has already finalized.";
    return;
  }
  // ge finalize
  GeFinalize();

  const char *tdt_uninit_env = std::getenv("ASCEND_TDT_UNINIT");
  bool tdt_init = true;
  if (tdt_uninit_env != nullptr && std::atoi(tdt_uninit_env) == 1) {
    tdt_init = false;
  }
  if (!kIsHeterogeneous && tdt_init) {
    ADP_LOG(INFO) << "[GePlugin] Close TsdClient and destroy tdt.";
    int32_t tdt_ret = tdt::TdtOutFeedDestroy();
    if (tdt_ret != 0) {
      LOG(ERROR) << "[GePlugin] Close tdt host failed.";
      ADP_LOG(ERROR) << "[GePlugin] Close tdt host failed.";
    }
  }
  isInit_ = false;
}

bool GePlugin::IsGlobal() {
  std::lock_guard<std::mutex> lock(mutex_);
  return isGlobal_;
}

static CancellationManager g_cancellationManager;
Status RegisterNpuCancellationCallback(std::function<void()> callback, std::function<void()> *deregister_fn) {
  CancellationToken token = g_cancellationManager.get_cancellation_token();
  if (!g_cancellationManager.RegisterCallback(token, std::move(callback))) {
    return errors::Cancelled("Operation was cancelled");
  }
  *deregister_fn = [token]() { g_cancellationManager.DeregisterCallback(token); };
  return Status::OK();
}

void PluginInit(std::map<std::string, std::string> &init_options) {
  GePlugin::GetInstance()->Init(init_options, true);
  ADP_LOG(INFO) << "[GePlugin] npu plugin init success.";
}

void PluginFinalize() {
  GePlugin::GetInstance()->Finalize();
  ADP_LOG(INFO) << "[GePlugin] npu plugin finalize success.";
}

void AoeFinalizeIfNeed() {
  auto attr = GePlugin::GetInstance()->GetInitOptions();
  if (attr["ge.jobType"].empty() || attr["ge.tuningPath"].empty()) {
    return;
  }

  ADP_LOG(INFO) << "Start to call aoe finalize when npu close.";
  void *handle = mmDlopen("libaoe_tuning.so", MMPA_RTLD_NOW);
  if (handle == nullptr) {
    ADP_LOG(WARNING) << "open libaoe_tuning.so failed.";
    return;
  }

  auto aoe_finalize = reinterpret_cast<AoeFinalizeFunc>(mmDlsym(handle, "AoeFinalize"));
  if (aoe_finalize == nullptr) {
    ADP_LOG(WARNING) << "load aoe finalize function failed.";
    return;
  }

  (void) aoe_finalize();
  (void) mmDlclose(handle);

  ADP_LOG(INFO) << "Finish to call aoe finalize when npu close.";
}

void NpuClose() {
  ADP_LOG(INFO) << "[GePlugin] Npu close.";
  g_cancellationManager.StartCancel();
  GeFinalize();
  AoeFinalizeIfNeed();
  uint32_t device_id = 0;
  (void) GetEnvDeviceID(device_id);
  if (NpuAttrs::GetUseTdtStatus(device_id)) {
    ADP_LOG(INFO) << "[GePlugin] the process has turned on TDT resource, finalize resource at exit.";
    int32_t tdt_status = TdtInFeedDestroy(device_id);
    if (tdt_status != 0) {
      ADP_LOG(ERROR) << "[GePlugin] Tdt client close failed.";
      LOG(ERROR) << "[GePlugin] Tdt client close failed.";
    } else {
      ADP_LOG(INFO) << "[GePlugin] Tdt client close success.";
      NpuAttrs::SetUseTdtStatus(device_id, false);
    }
  }
  ADP_LOG(INFO) << "[GePlugin] npu finalize resource success.";
}

int32_t InitRdmaPool(size_t size) {
  ge::Status ret = GeApiWrapper_InitRdmaPool(size, RT_MEMORY_HBM);
  if (ret != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] init rdma pool failed, ret : " << ToString(ret);
    LOG(ERROR) << "[GePlugin] init rdma pool failed, ret : " << ToString(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] init rdma pool success.";
  return 0;
}

int32_t RegistRdmaRemoteAddr(const std::vector<std::pair<uint64_t, uint64_t>> &var_info) {
  ge::Status ret = GeApiWrapper_RdmaRemoteRegister(var_info, RT_MEMORY_HBM);
  if (ret != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] rdma remote register failed, ret : " << ToString(ret);
    LOG(ERROR) << "[GePlugin] rdma remote register failed, ret : " << ToString(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] rdma remote register success.";
  return 0;
}

int32_t RdmaInitAndRegister(const std::vector<std::pair<uint64_t, uint64_t>> &var_info, size_t size) {
  ge::Status ret = GeApiWrapper_InitRdmaPool(size, RT_MEMORY_HBM);
  if (ret != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] init rdma pool failed, ret : " << ToString(ret);
    LOG(ERROR) << "[GePlugin] init rdma pool failed, ret : " << ToString(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] init rdma pool success.";
  ret = GeApiWrapper_RdmaRemoteRegister(var_info, RT_MEMORY_HBM);
  if (ret != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] rdma remote register failed, ret : " << ToString(ret);
    LOG(ERROR) << "[GePlugin] rdma remote register failed, ret : " << ToString(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] rdma remote register success.";
  return 0;
}

int32_t GetVarAddrAndSize(const string &var_name, uint64_t &base_addr, uint64_t &var_size) {
  ge::Status ret = GeApiWrapper_GetVarBaseAddrAndSize(var_name.c_str(), base_addr, var_size);
  if (ret != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] get " << var_name << " base addr and size failed, ret : " << ToString(ret);
    LOG(ERROR) << "[GePlugin] get " << var_name << " base addr and size failed, ret : " << ToString(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] get " << var_name << " base addr and size success.";
  return 0;
}

int32_t MallocSharedMem(const std::string &var_name, const std::vector<int64_t> &dims, ge::DataType data_type,
                        uint64_t &dev_addr, uint64_t &memory_size) {
  ge::Status ret = GeApiWrapper_MallocSharedMemory(var_name, dims, data_type, dev_addr, memory_size);
  if (ret != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] malloc shared memory failed, ret : " << ToString(ret);
    LOG(ERROR) << "[GePlugin] malloc shared memory failed, ret : " << ToString(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] malloc shared memory success.";
  return 0;
}

int32_t SetDeviceSatMode(uint32_t mode) {
  aclError ret = aclrtSetDeviceSatMode(aclrtFloatOverflowMode(mode));
  if (ret != ACL_SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] set device sat mode failed, ret : " << static_cast<int32_t>(ret);
    LOG(ERROR) << "[GePlugin] set device sat mode failed, ret : " << static_cast<int32_t>(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] set device sat mode success.";
  return 0;
}

int32_t GetDeviceSatMode() {
  aclrtFloatOverflowMode floatOverflowMode = ACL_RT_OVERFLOW_MODE_UNDEF;
  aclError ret = aclrtGetDeviceSatMode(&floatOverflowMode);
  if (ret != ACL_SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] get device sat mode failed, ret : " << static_cast<int32_t>(ret);
    LOG(ERROR) << "[GePlugin] get device sat mode failed, ret : " << static_cast<int32_t>(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] get device sat mode success.";
  return static_cast<int32_t>(floatOverflowMode);
}
std::atomic_int GePlugin::graph_counter_ = {0};
