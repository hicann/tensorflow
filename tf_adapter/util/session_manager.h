/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_SESSION_MANAGER_H_
#define TENSORFLOW_SESSION_MANAGER_H_

#include <mutex>
#include <string>
#include <unordered_map>
#include "ge/ge_api.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/mutex.h"

// Singleton class for manage the relationship between
// tf session and ge session
class SessionManager {
 public:
  static SessionManager &GetInstance();

  // Retrieves an already existing ge session to run the compute graph,
  // or create a ge session for future use.
  bool GetOrCreateGeSession(const std::string &tf_session, ge::Session *&ge_session,
                            std::map<std::string, std::string> &sess_options);

  // Destroy a ge session divided by tf session.
  void DestroyGeSession(const std::string &tf_session);

  // Whether a ge session exist.
  bool IsGeSessionExist() const ;

 private:
  // Create a ge session to run the compute graph divided by tf session.
  bool CreateGeSession(const std::string &tf_session, ge::Session *&ge_session,
                       std::map<std::string, std::string> &sess_options);
  // Print ge session options
  void PrintGeSessionOptions(std::map<std::string, std::string> &sess_options) const;
  // Mapping relationship between tf session and ge session.
  std::unordered_map<std::string, ge::Session *> ge_sessions_;
};
#endif
