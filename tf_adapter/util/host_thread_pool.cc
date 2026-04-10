/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_thread_pool.h"
#include "acl/acl_rt.h"
#include "tf_adapter/common/adapter_logger.h"

namespace {
  const uint32_t MAX_THREAD_NUM = 4U;
}
namespace tensorflow {
  HostThreadPool::HostThreadPool() : thread_stop_flag_(false), device_id_(0U) {}

  HostThreadPool::~HostThreadPool() {}

  Status HostThreadPool::Init(uint32_t device_id) {
    ADP_LOG(INFO) << "Start to start thread pool.";
    device_id_ = device_id;
    copy_thread_pool_.resize(MAX_THREAD_NUM);
    if (Env::Default() == nullptr) {
      ADP_LOG(ERROR) << "Env default is nullptr.";
      return errors::InvalidArgument("Init memory pool failed");
    }
    for (size_t idx = 0UL; idx < copy_thread_pool_.size(); idx++) {
      if (copy_thread_pool_[idx] == nullptr) {
        std::string thread_name = "thread_pool" + std::to_string(idx);
        copy_thread_pool_[idx].reset(
            Env::Default()->StartThread({}, thread_name, [this]() { ParallelForCopyThread(); }));
      }
    }
    return Status::OK();
  }

  void HostThreadPool::ParallelForCopyThread() {
    ADP_LOG(INFO) << "Start parallel copy thread.";
    std::function<void()> closure;
    while (!thread_stop_flag_.load()) {
      {
        std::unique_lock<std::mutex> lck(queue_lock_);
        queue_var_.wait(lck, [this]() { return ((!task_queue_.empty()) || (thread_stop_flag_.load())); });
        if (thread_stop_flag_.load()) {
          queue_var_.notify_all();
          break;
        }
        closure = task_queue_.front();
        task_queue_.pop();
      }
      closure();
    }
    ADP_LOG(INFO) << "Copy thread is finished.";
  }

  void HostThreadPool::PushTask(const std::function<void()> &closure) {
    std::unique_lock<std::mutex> lck(queue_lock_);
    task_queue_.push(closure);
    queue_var_.notify_one();
  }

  void HostThreadPool::StopThreadPool() {
    thread_stop_flag_.store(true);
    queue_var_.notify_all();
  }
}
