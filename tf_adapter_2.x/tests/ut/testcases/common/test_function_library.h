/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WORKSPACE_TEST_FUNCTION_LIBRARY_H
#define WORKSPACE_TEST_FUNCTION_LIBRARY_H

#include <mutex>
#include <string>
#include <vector>

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"

class FunctionStrLibrary {
 public:
  static FunctionStrLibrary &Instance() {
    static FunctionStrLibrary library;
    return library;
  }

  void Add(const std::string &readable_str) {
    tensorflow::FunctionDef def;
    CHECK(tensorflow::protobuf::TextFormat::ParseFromString(readable_str, &def));
    std::unique_lock<std::mutex> lk(mu_);
    function_defs_.emplace_back(def.SerializeAsString());
  }

  std::vector<std::string> Get() {
    std::unique_lock<std::mutex> lk(mu_);
    return function_defs_;
  }

 private:
  FunctionStrLibrary() = default;
  ~FunctionStrLibrary() = default;
  std::mutex mu_;
  std::vector<std::string> function_defs_;
};

#define REGISTER_TEST_FUNC(str) REGISTER_TEST_FUNC_1(__COUNTER__, (str))
#define REGISTER_TEST_FUNC_1(ctr, str) REGISTER_TEST_FUNC_2(ctr, (str))
#define REGISTER_TEST_FUNC_2(ctr, str)         \
  static int __registered_func##ctr = []() {   \
    FunctionStrLibrary::Instance().Add((str)); \
    return 0;                                  \
  }();

#endif  // WORKSPACE_TEST_FUNCTION_LIBRARY_H
