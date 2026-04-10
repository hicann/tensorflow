/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_adapter/util/session_manager.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/common/adapter_logger.h"

using namespace tensorflow;
namespace {
std::string GetOptionVal(const std::map<std::string, std::string> &options, const std::string option_key) {
  auto it = options.find(option_key);
  return (it != options.end() ? it->second : "");
}
}
/**
 * @brief: get instance
 */
SessionManager &SessionManager::GetInstance() {
  static SessionManager instance;
  return instance;
}

// Returns True if get ge session success.
bool SessionManager::GetOrCreateGeSession(const std::string &tf_session, ge::Session *&ge_session,
                                          std::map<std::string, std::string> &sess_options) {
  // find valid tf session handle
  if (tf_session.empty()) {
    ADP_LOG(ERROR) << "tf session is empty, get ge session failed.";
    LOG(ERROR) << "tf session is empty, get ge session failed.";
    return false;
  }

  // find valid ge session
  auto it = ge_sessions_.find(tf_session);
  if (it != ge_sessions_.end()) {
    ge_session = it->second;
    ADP_LOG(INFO) << "tf session " << tf_session << " get ge session success.";
    return true;
  }
  ADP_LOG(INFO) << "Session options: ";
  NpuAttrs::LogOptions(sess_options);
  PrintGeSessionOptions(sess_options);
  bool ret = SessionManager::CreateGeSession(tf_session, ge_session, sess_options);
  if (!ret) {
    ADP_LOG(ERROR) << "tf session " << tf_session << " create ge session failed.";
    LOG(ERROR) << "tf session " << tf_session << " create ge session failed.";
    return false;
  }
  return true;
}

/**
 * @brief: destroy ge session.
 * @param tf_session: tf session
 */
void SessionManager::DestroyGeSession(const std::string &tf_session) {
  if (tf_session.empty()) {
    ADP_LOG(ERROR) << "tf session is empty, can not destroy ge session.";
    LOG(ERROR) << "tf session is empty, can not destroy ge session.";
  }
  auto it = ge_sessions_.find(tf_session);
  if (it != ge_sessions_.end()) {
    if (it->second != nullptr) {
      ADP_LOG(INFO) << "find ge session connect with tf session " << tf_session;
      delete it->second;
      it->second = nullptr;
    }
    (void)ge_sessions_.erase(it);
    ADP_LOG(INFO) << "destroy ge session connect with tf session " << tf_session << " success.";
  }
}

// Returns True if create ge session success.
bool SessionManager::CreateGeSession(const std::string &tf_session, ge::Session *&ge_session,
                                     std::map<std::string, std::string> &sess_options) {
  // hcom parallel
  ADP_LOG(INFO) << "[GEOP] hcom_parallel :" << sess_options[ge::HCOM_PARALLEL];

  // stream max parallel num
  ADP_LOG(INFO) << "[GEOP] stream_max_parallel_num :" << sess_options[ge::STREAM_MAX_PARALLEL_NUM];
  const auto sess_options_ascend_string = ChangeStringToAscendString(sess_options);
  ge_session = new (std::nothrow) ge::Session(sess_options_ascend_string);
  if (ge_session == nullptr) {
    ADP_LOG(ERROR) << "tf session " << tf_session << " create ge session failed.";
    LOG(ERROR) << "tf session " << tf_session << " create ge session failed.";
    return false;
  }
  (void)ge_sessions_.insert(std::make_pair(tf_session, ge_session));
  return true;
}

// Returns True if any ge session exist.
bool SessionManager::IsGeSessionExist() const {
  return !ge_sessions_.empty();
}

void SessionManager::PrintGeSessionOptions(std::map<std::string, std::string> &sess_options) const {
  // variable acceleration configuration
  ADP_LOG(INFO) << "[GEOP] variable_acceleration :" << GetOptionVal(sess_options, "ge.exec.variable_acc");
  // hcom parallel
  ADP_LOG(INFO) << "[GEOP] hcom_parallel :" << GetOptionVal(sess_options, ge::HCOM_PARALLEL);

  // stream max parallel num
  ADP_LOG(INFO) << "[GEOP] stream_max_parallel_num :" << GetOptionVal(sess_options, ge::STREAM_MAX_PARALLEL_NUM);
  // ac parallel enable
  ADP_LOG(INFO) << "[GEOP] ac_parallel_enable :" << GetOptionVal(sess_options, ge::AC_PARALLEL_ENABLE);
  // quant dumpable
  ADP_LOG(INFO) << "[GEOP] quant_dumpable :" << GetOptionVal(sess_options, ge::QUANT_DUMPABLE);

  // graph memory configuration
  if (!GetOptionVal(sess_options, ge::GRAPH_MEMORY_MAX_SIZE).empty()) {
    ADP_LOG(INFO) << "[GEOP] set graph_memory_max_size: " << sess_options[ge::GRAPH_MEMORY_MAX_SIZE];
  } else {
    (void)sess_options.erase(ge::GRAPH_MEMORY_MAX_SIZE);
  }

  // variable memory configuration
  if (!GetOptionVal(sess_options, ge::VARIABLE_MEMORY_MAX_SIZE).empty()) {
    ADP_LOG(INFO) << "[GEOP] set variable_memory_max_size: " << sess_options[ge::VARIABLE_MEMORY_MAX_SIZE];
  } else {
    (void)sess_options.erase(ge::VARIABLE_MEMORY_MAX_SIZE);
  }

  // tailing optimization
  ADP_LOG(INFO) << "[GEOP] is_tailing_optimization : " << GetOptionVal(sess_options, "ge.exec.isTailingOptimization");

  ADP_LOG(INFO) << "[GEOP] op_select_implmode : " << GetOptionVal(sess_options, ge::OP_SELECT_IMPL_MODE);

  ADP_LOG(INFO) << "[GEOP] optypelist_for_implmode : " << GetOptionVal(sess_options, ge::OPTYPELIST_FOR_IMPLMODE);

  // dump configuration
  string dump_step = GetOptionVal(sess_options, ge::OPTION_EXEC_DUMP_STEP);
  ADP_LOG(INFO) << "[GEOP] enable_dump :" << GetOptionVal(sess_options, ge::OPTION_EXEC_ENABLE_DUMP)
                << ", dump_path :" << GetOptionVal(sess_options, ge::OPTION_EXEC_DUMP_PATH)
                << ", dump_step :" << (dump_step.empty() ? "NA" : dump_step)
                << ", dump_mode :" << GetOptionVal(sess_options, ge::OPTION_EXEC_DUMP_MODE)
                << ", enable_dump_debug :" << GetOptionVal(sess_options, ge::OPTION_EXEC_ENABLE_DUMP_DEBUG)
                << ", dump_debug_mode :" << GetOptionVal(sess_options, ge::OPTION_EXEC_DUMP_DEBUG_MODE);

  ADP_LOG(INFO) << "[GEOP] buffer_optimize :" << GetOptionVal(sess_options, "ge.bufferOptimize");

  ADP_LOG(INFO) << "[GEOP] enable_small_channel :" << GetOptionVal(sess_options, "ge.enableSmallChannel");

  ADP_LOG(INFO) << "[GEOP] fusion_switch_file :" << GetOptionVal(sess_options, "ge.fusionSwitchFile");

  ADP_LOG(INFO) << "[GEOP] enable_compress_weight :" << GetOptionVal(sess_options, "ge.enableCompressWeight");

  ADP_LOG(INFO) << "[GEOP] compress_weight_conf :" << GetOptionVal(sess_options, "compress_weight_conf");
}
