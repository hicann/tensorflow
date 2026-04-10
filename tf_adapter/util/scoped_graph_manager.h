/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_UTILS_SCOPED_GRAPH_MANAGER_H_
#define TENSORFLOW_UTILS_SCOPED_GRAPH_MANAGER_H_
#include "acl/acl_prof.h"
#include <string>
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/lib/core/status.h"
#include <thread>


namespace tensorflow {

class ScopedGraphManager {
public:
  static ScopedGraphManager& Instance();

  // 启用图生命周期控制
  void EnableControl();

  // 禁用图生命周期控制
  void DisableControl();

  bool IsControlEnabled() const;

  // 注册图
  bool SetGraph(const std::string& tf_session, const uint32_t& graph_id);

  // 清理状态，卸载图并释放其占用内存
  void Clear();

private:
  ScopedGraphManager() = default;

  static uint32_t graph_id_;

  static std::string tf_session_;

  static std::mutex mutex_;

  static bool graph_life_control_enabled_;
};
}

#endif
