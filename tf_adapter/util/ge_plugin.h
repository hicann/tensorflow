/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_GE_PLUGIN_H_
#define TENSORFLOW_GE_PLUGIN_H_

#include <map>
#include <mutex>
#include <string>
#include <atomic>
#include <future>
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/lib/core/status.h"

#include "ge_common/ge_api_error_codes.h"
// Singleton class for manage the relationship between
// tf session and ge session
class GePlugin {
 public:
  static GePlugin *GetInstance();

  void Init(std::map<std::string, std::string> &init_options, const bool is_global = false,
            const bool is_async = false);

  void Finalize();

  bool IsGlobal();

  ge::Status GetInitStatus() {
    if (future_.valid()) {
      return future_.get();
    }
    return ge::SUCCESS;
  }

  std::string GetInitErrorMessage() {
    return error_message_;
  }

  std::string GetInitWarningMessage() {
    return warning_message_;
  }

  std::map<std::string, std::string> GetInitOptions();

  void SetRankTableFileEnv(std::map<std::string, std::string> &init_options, std::string &rankTableFile);

  void SetCmChiefWorkSizeEnv(std::map<std::string, std::string> &init_options, std::string &cmChiefIp);

 private:
  GePlugin();

  ~GePlugin();

  uint64_t GetFusionTensorSize() const;

  uint32_t device_id_;
  bool isInit_;
  bool isGlobal_;
  bool is_use_hcom = false;
  bool deploy_mode = false;
  tensorflow::int64 work_size_num;
  tensorflow::int64 rank_size_num;
  std::map<std::string, std::string> init_options_;
  std::mutex mutex_;
  static std::atomic_int graph_counter_;
  std::shared_future<ge::Status> future_;
  std::string error_message_;
  std::string warning_message_;
};

tensorflow::Status RegisterNpuCancellationCallback(std::function<void()> callback,
                                                   std::function<void()> *deregister_fn);
// } // tensorflow

#endif
