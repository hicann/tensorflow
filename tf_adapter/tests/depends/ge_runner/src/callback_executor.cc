/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "callback_executor.h"
#include <iostream>
#include "acl/acl_rt.h"

namespace tensorflow {
  CallbackExecutor &CallbackExecutor::GetInstance() {
    static CallbackExecutor instance;
    return instance;
  }
  void CallbackExecutor::Init() {
    std::cout << "Start callback thread pool." << std::endl;
    copy_thread_pool_.resize(thread_num_);
    for (size_t idx = 0UL; idx < copy_thread_pool_.size(); idx++) {
      if (copy_thread_pool_[idx] == nullptr) {
        std::string thread_name = "thread_pool" + std::to_string(idx);
        copy_thread_pool_[idx].reset(new std::thread(std::bind(&CallbackExecutor::CallbackHandler, this)));
      }
    }
    thread_stop_flag_.store(false);
  }

  void CallbackExecutor::CallbackHandler() {
    std::cout << "Start callback thread." << std::endl;
    CallbackPack closure;
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
        std::cout << "Run callback" << std::endl;
      }
      closure.callback(closure.ge_status, closure.outputs);
      std::unique_lock<std::mutex> lck(queue_lock_);
      run_num_--;
    }
    std::cout << "Callback thread is finished." << std::endl;
  }

  void CallbackExecutor::PushTask(const CallbackPack &closure) {
    std::unique_lock<std::mutex> lck(queue_lock_);
    std::cout << "Push closure" << std::endl;
    task_queue_.push(closure);
    run_num_++;
    queue_var_.notify_all();
  }

  void CallbackExecutor::StopThreadPool() {
    {
      std::unique_lock<std::mutex> lck(queue_lock_);
      queue_var_.wait(lck, [this]() { return run_num_ <= 0; });
      std::cout << "Stop callback thread." << std::endl;
      thread_stop_flag_.store(true);
      queue_var_.notify_all();
    }
    for (size_t i = 0UL; i < copy_thread_pool_.size(); i++) {
      if (copy_thread_pool_[i]->joinable()) {
        copy_thread_pool_[i]->join();
      }
    }
  }
  int32_t CallbackExecutor::GetRunNum() {
    std::unique_lock<std::mutex> lck(queue_lock_);
    return run_num_;
  }
}
