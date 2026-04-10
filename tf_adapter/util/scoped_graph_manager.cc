/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scoped_graph_manager.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/util/session_manager.h"
#include "npu_attrs.h"

namespace tensorflow {
uint32_t ScopedGraphManager::graph_id_ = UINT32_MAX;
std::string ScopedGraphManager::tf_session_;
std::mutex ScopedGraphManager::mutex_;
bool ScopedGraphManager::graph_life_control_enabled_;

ScopedGraphManager& ScopedGraphManager::Instance() {
    static ScopedGraphManager instance;
    return instance;
}

void ScopedGraphManager::EnableControl() {
    std::lock_guard<std::mutex> lock(mutex_);
    graph_life_control_enabled_ = true;
    ADP_LOG(INFO) << "[ScopedGraphManager] Set graph_life_control_enabled_ true";
}

void ScopedGraphManager::DisableControl() {
    std::lock_guard<std::mutex> lock(mutex_);
    graph_life_control_enabled_ = false;
    ADP_LOG(INFO) << "[ScopedGraphManager] Set graph_life_control_enabled_ false";
}

bool ScopedGraphManager::IsControlEnabled() const {
    ADP_LOG(INFO) << "[ScopedGraphManager] Get graph_life_control_enabled_: " << graph_life_control_enabled_;
    return graph_life_control_enabled_;
}

bool ScopedGraphManager::SetGraph(const std::string& tf_session, const uint32_t& graph_id) {
    if (graph_id_ != UINT32_MAX) {
        ADP_LOG(ERROR) << "[ScopedGraphManager] Only support call sess.run once in scope of ScopedGraphManager.";
        return false;
    }
    ADP_LOG(INFO) << "[ScopedGraphManager] SetGraph tf_session: " << tf_session << ", graph_id: " << graph_id;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        tf_session_ = tf_session;
        graph_id_ = graph_id;
    }

    ADP_LOG(INFO) << "[ScopedGraphManager] SetGraph success for tf_session: " << tf_session << ", graph_id: " << graph_id;
    return true;
}

void ScopedGraphManager::Clear() {
  ADP_LOG(INFO) << "[ScopedGraphManager] Begin to clear after graph run";
  DisableControl();

  std::lock_guard<std::mutex> lock(mutex_);
  ge::Session* global_ge_session = nullptr;
  std::map<std::string, std::string> global_sess_options;
  // 空图，未注册图，tf_session_为空
  if (tf_session_.empty()) {
    ADP_LOG(WARNING) << "[ScopedGraphManager] No need to RemoveGraph, tf_session is empty";
    graph_id_ = UINT32_MAX;
    ADP_LOG(INFO) << "[ScopedGraphManager] Clear finished";
    return;
  }
  if (!SessionManager::GetInstance().GetOrCreateGeSession(tf_session_, global_ge_session, global_sess_options)) {
    ADP_LOG(WARNING) << "[ScopedGraphManager] Failed to get session for tf_session: " << tf_session_;
  }
  if (global_ge_session != nullptr) {
    global_ge_session->RemoveGraph(graph_id_);
    ADP_LOG(INFO) << "[ScopedGraphManager] RemoveGraph success for tf_session: "
                    << tf_session_ << ", graph_id: " << graph_id_;
  }
  graph_id_ = UINT32_MAX;
  tf_session_.clear();
  ADP_LOG(INFO) << "[ScopedGraphManager] Clear finished";
}
}
