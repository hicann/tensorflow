/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNELS_UTIL_MEMORY_POOL_H_
#define KERNELS_UTIL_MEMORY_POOL_H_

#include <cstdlib>
#include <cstdint>
#include <memory>
#include <atomic>
#include <list>
#include <thread>
#include <mutex>
#include <queue>
#include <functional>
#include <condition_variable>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
struct MemoryBlock {
  void *ptr;
  uint64_t data_size;
  MemoryBlock(void *in_ptr, uint64_t in_size) : ptr(in_ptr), data_size(in_size) {}
};

class MemoryPool {
 public:
  MemoryPool();
  Status MallocMemory(void *&buffer,
                      uint64_t args_size);
  void ReleaseMemory();
  Status FreeAllMemory();
  ~MemoryPool();
 private:
  bool FreeMemoryList(std::list<MemoryBlock> &memory_list) const;
  std::mutex memory_pool_lock_;
  std::list<MemoryBlock> used_memory_list_;
  std::list<MemoryBlock> free_memory_list_;
};
}
#endif
