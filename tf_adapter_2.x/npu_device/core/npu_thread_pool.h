/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_DEVICE_CORE_NPU_THREAD_POOL_H
#define NPU_DEVICE_CORE_NPU_THREAD_POOL_H

#include <queue>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"

namespace npu {
const uint32_t kDefaultThreadNum = 4U;

class NpuThreadPool {
 public:
  static NpuThreadPool &GetInstance() {
    static NpuThreadPool Instance;
    return Instance;
  }

  tensorflow::Status EnqueueTask(std::function<void()> closure) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      if (request_stop_) {
        return tensorflow::errors::Internal("thread pool is stopping");
      }
      requests_.emplace(closure);
    }
    cv_.notify_one();
    return tensorflow::Status::OK();
  }

  void Init(size_t thread_num) {
    workers_.resize(thread_num);
    for (size_t i = 0; i < thread_num; i++) {
      workers_[i].reset(tensorflow::Env::Default()->StartThread(
        tensorflow::ThreadOptions{}, "thread_pool_" + std::to_string(i), [this]() {
          while (true) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this]() -> bool { return !requests_.empty() || request_stop_; });
            if (request_stop_ && requests_.empty()) {
              return;
            }
            do {
              auto task = requests_.front();
              requests_.pop();
              lk.unlock();
              task();
            } while (request_stop_ && !requests_.empty());
          }
        }));
    }
  }

  void Destroy() {
    {
      std::unique_lock<std::mutex> lk(mu_);
      request_stop_ = true;
    }
    cv_.notify_all();
    workers_.clear();
  }

  NpuThreadPool() : request_stop_(false) {}

  ~NpuThreadPool() { Destroy(); }

 private:
  bool request_stop_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::queue<std::function<void()>> requests_;
  std::vector<std::unique_ptr<tensorflow::Thread>> workers_;
};
}  // namespace npu

#endif  // NPU_DEVICE_CORE_NPU_THREAD_POOL_H
