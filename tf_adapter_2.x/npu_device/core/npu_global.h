/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_DEVICE_CORE_NPU_GLOBAL_H
#define NPU_DEVICE_CORE_NPU_GLOBAL_H

#include <map>
#include <vector>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"

#include "acl/acl_rt.h"

namespace npu {
class NpuDevice;
class HdcChannel;
namespace global {
// 全局Device循环次数设置
extern std::atomic_int64_t g_npu_loop_size;
// 全局NPU自定义OP
extern std::unordered_set<std::string> g_npu_specify_ops;
// 控制Device内存释放的全局读写锁
extern tensorflow::mutex dev_memory_shared_lock;
extern bool dev_memory_released TF_GUARDED_BY(dev_memory_shared_lock);

// Rts ctx管理器
class RtsCtx {
 public:
  static tensorflow::Status CreateGlobalCtx(int32_t device_index);
  static tensorflow::Status EnsureInitialized();
  static tensorflow::Status DestroyGlobalCtx();
 private:
  static aclrtContext global_ctx_;
  static tensorflow::mutex global_ctx_mutex_;
};

class NpuCtx {
 public:
  static void SetDeviceCtx(int id, TFE_Context *ctx, NpuDevice *device);
  static tensorflow::Status GetDeviceCtx(int id, TFE_Context **ctx, NpuDevice **device);
  struct Ctx {
    TFE_Context *ctx;
    NpuDevice *device;
  };

 private:
  static std::map<int, NpuCtx::Ctx> npu_ctx_;
};

class GlobalHdcChannel {
 public:
  static GlobalHdcChannel &GetInstance() {
    static GlobalHdcChannel Instance;
    return Instance;
  }

  void Get(const std::string &name, std::vector<std::shared_ptr<npu::HdcChannel>> &channels);

  tensorflow::Status Create(const std::string &name, int64_t channel_capacity, const std::vector<int> &device_ids);

  void Destroy(const std::string &name);

 private:
  std::map<std::string, std::vector<std::shared_ptr<npu::HdcChannel>>> global_channels_;
  std::mutex global_channels_mu_;
};

}  // namespace global
}  // namespace npu

#endif  // NPU_DEVICE_CORE_NPU_GLOBAL_H
