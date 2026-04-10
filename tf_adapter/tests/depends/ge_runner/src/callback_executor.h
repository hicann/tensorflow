/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_DEPENDS_GE_RUNNER_SRC_HOST_THREAD_POOL_H_
#define TESTS_DEPENDS_GE_RUNNER_SRC_HOST_THREAD_POOL_H_

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <functional>
#include <condition_variable>
#include <thread>
#include "ge_common/ge_api_types.h"
#include "graph/tensor.h"

namespace tensorflow {
struct CallbackPack {
  ge::RunAsyncCallback callback;
  ge::Status ge_status;
  std::vector<ge::Tensor> outputs;
};
class CallbackExecutor {
 public:
  static CallbackExecutor &GetInstance();
  void Init();
  void PushTask(const CallbackPack &closure);
  void StopThreadPool();
  int32_t GetRunNum();
 private:
  void CallbackHandler();
  std::mutex queue_lock_;
  std::condition_variable queue_var_;
  std::vector<std::unique_ptr<std::thread>> copy_thread_pool_;
  std::queue<CallbackPack> task_queue_;
  std::atomic<bool> thread_stop_flag_{false};
  uint32_t thread_num_ = 1U;
  int32_t run_num_ = 0;
};
}
#endif // TESTS_DEPENDS_GE_RUNNER_SRC_HOST_THREAD_POOL_H_
