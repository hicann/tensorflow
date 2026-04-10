/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_HOST_ALLOCATOR_H_
#define TENSORFLOW_HOST_ALLOCATOR_H_

#include <string>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {
  class HostAllocator : public Allocator, public tensorflow::core::RefCounted {
  public:
    explicit HostAllocator(void *addr);
    ~HostAllocator() override;
    std::string Name() override;
    void *AllocateRaw(size_t alignment, size_t num_bytes) override;
    void *AllocateRaw(size_t alignment, size_t num_bytes,
                      const AllocationAttributes &allocation_attr) override;
    void DeallocateRaw(void *ptr) override;
  private:
    void *addr_;
  };
}
#endif
