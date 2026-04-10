/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_allocator.h"

namespace tensorflow {
  HostAllocator::HostAllocator(void *addr) : addr_(addr) {}
  HostAllocator::~HostAllocator() {
    addr_ = nullptr;
  }
  std::string HostAllocator::Name() {
    return "host_allocator";
  }
  void *HostAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
    (void) alignment;
    (void) num_bytes;
    return addr_;
  }
  void *HostAllocator::AllocateRaw(size_t alignment, size_t num_bytes,
                                   const AllocationAttributes &allocation_attr) {
    (void) alignment;
    (void) num_bytes;
    (void) allocation_attr;
    return addr_;
  }
  void HostAllocator::DeallocateRaw(void *ptr) {
    (void) ptr;
    Unref();
  }
}
