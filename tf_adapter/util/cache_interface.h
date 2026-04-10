/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_CORE_FRAMEWORK_CACHE_INTERFACE_H_
#define TENSORFLOW_CORE_FRAMEWORK_CACHE_INTERFACE_H_

#include <string>
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
// All implementations must be thread-safe.
class CacheInterface : public ResourceBase {
 public:
  virtual void add(std::vector<uint64> &ids_vec, std::vector<uint64> &swap_in_id_temp,
                   std::vector<uint64> &swap_in_idx_temp, std::vector<uint64> &swap_out_id_temp,
                   std::vector<uint64> &swap_out_idx_temp, int64 &swap_in_num, int64 &swap_out_num) = 0;

  virtual void remoteIndexToLocal(const std::vector<uint64> &ids_vec, Tensor &local_idx) = 0;
  // Return the num of elements in cache
  virtual int64 size() const = 0;
  // Return a debug string for *this
  string DebugString() const override {
    return strings::StrCat("A Cache of size: ", size());
  }

  virtual ~CacheInterface() override {}
};
}  // namespace tensorflow
#endif  // TENSORFLOW_CACHE_INTERFACE_H_
