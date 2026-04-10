/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNELS_UTIL_HOST_THREAD_POOL_H_
#define KERNELS_UTIL_HOST_THREAD_POOL_H_

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <functional>
#include <condition_variable>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
class HostThreadPool {
 public:
  HostThreadPool();
  Status Init(uint32_t device_id);
  void PushTask(const std::function<void()> &closure);
  void StopThreadPool();
  ~HostThreadPool();
 private:
  void ParallelForCopyThread();
  std::mutex queue_lock_;
  std::condition_variable queue_var_;
  std::vector<std::unique_ptr<Thread>> copy_thread_pool_;
  std::queue<std::function<void()>> task_queue_;
  std::atomic<bool> thread_stop_flag_;
  uint32_t device_id_;
};
}
#endif
