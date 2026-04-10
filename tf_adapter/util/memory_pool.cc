/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "memory_pool.h"
#include <vector>
#include <string>
#include "securec.h"
#include "tf_adapter/common/adapter_logger.h"

namespace tensorflow {
  constexpr uint64_t kMemAlignSize = 128;
  MemoryPool::MemoryPool() {}

  MemoryPool::~MemoryPool() {}

  Status MemoryPool::MallocMemory(void *&buffer,
                                  uint64_t args_size) {
    MemoryBlock temp_block(nullptr, args_size);
    {
      std::lock_guard<std::mutex> lck(memory_pool_lock_);
      auto free_it = free_memory_list_.begin();
      while (free_it != free_memory_list_.end()) {
        if (free_it->data_size >= args_size) {
          temp_block = (*free_it);
          free_it = free_memory_list_.erase(free_it);
          break;
        }
        ++free_it;
      }
      if ((temp_block.ptr == nullptr) && (!free_memory_list_.empty())) {
        if (!FreeMemoryList(free_memory_list_)) {
          ADP_LOG(ERROR) << "Release free host memory failed";
          return errors::InvalidArgument("Release free host memory failed");
        }
      }
    }

    if (temp_block.ptr == nullptr) {
      int ret = posix_memalign(&temp_block.ptr, kMemAlignSize, args_size);
      if ((ret != 0) || (temp_block.ptr == nullptr)) {
        ADP_LOG(ERROR) << "rtMalloc host memory failed";
        return errors::InvalidArgument("rtMalloc host memory failed");
      }
    }
    buffer = temp_block.ptr;
    std::lock_guard<std::mutex> lck(memory_pool_lock_);
    used_memory_list_.push_back(temp_block);
    return Status::OK();
  }

  void MemoryPool::ReleaseMemory() {
    std::lock_guard<std::mutex> lck(memory_pool_lock_);
    if (used_memory_list_.empty()) {
      return;
    }
    MemoryBlock head = used_memory_list_.front();
    used_memory_list_.pop_front();
    free_memory_list_.push_back(head);
  }

  Status MemoryPool::FreeAllMemory() {
    std::lock_guard<std::mutex> lck(memory_pool_lock_);
    if ((!FreeMemoryList(free_memory_list_)) || !FreeMemoryList(used_memory_list_)) {
      ADP_LOG(ERROR) << "Release host memory pool failed";
      return errors::InvalidArgument("Release host memory pool failed");
    }
    return Status::OK();
  }

  bool MemoryPool::FreeMemoryList(std::list<MemoryBlock> &memory_list) const {
    auto memory_it = memory_list.begin();
    while (memory_it != memory_list.end()) {
      free(memory_it->ptr);
      ++memory_it;
    }
    memory_list.clear();
    return true;
  }
}
