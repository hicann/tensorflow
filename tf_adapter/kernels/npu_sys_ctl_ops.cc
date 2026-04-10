/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_NPU_SYS_CTL_OPS_H_
#define TENSORFLOW_NPU_SYS_CTL_OPS_H_

#include <fstream>
#include <sys/time.h>

#include "ge/ge_api.h"
#include "ge_common/ge_api_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/mutex.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/kernels/geop_npu.h"
#include "tf_adapter/util/ge_plugin.h"
#include "tf_adapter/util/npu_attrs.h"

namespace tensorflow {
static const int64 kSecondToMillis = 1000000;

static int64 GetCurrentTimestamp() {
  struct timeval tv;
  int ret = gettimeofday(&tv, nullptr);
  if (ret != 0) {
    ADP_LOG(ERROR) << "Func gettimeofday failed, ret:" << ret;
    LOG(ERROR) << "Func gettimeofday failed, ret:" << ret;
    return 0;
  }
  int64 timestamp = tv.tv_usec + tv.tv_sec * kSecondToMillis;
  return timestamp;
}
static mutex g_mu(LINKER_INITIALIZED);
static int g_npuInitNum = 0;

static const int64 kMicrosToMillis = 1000;

class NPUInit : public OpKernel {
 public:
  explicit NPUInit(OpKernelConstruction *ctx);
  void Compute(OpKernelContext *ctx) override;
  ~NPUInit() override;

 private:
  std::map<std::string, std::string> init_options_;
};

NPUInit::NPUInit(OpKernelConstruction *ctx) : OpKernel(ctx) {
  ADP_LOG(INFO) << "NPUInit.";
  mutex_lock lock{g_mu};
  g_npuInitNum++;
  string sess_config;
  Status s = ctx->GetAttr("_NpuOptimizer", &sess_config);
  if (s.ok()) {
    init_options_ = NpuAttrs::GetInitOptions(ctx);
  } else {
    ADP_LOG(INFO) << "[NPUInit] NPUInit can not get _NpuOptimizer attr, use default init options";
    init_options_ = NpuAttrs::GetDefaultInitOptions();
  }
}

void NPUInit::Compute(OpKernelContext *ctx) {
  (void) ctx;
  if (GePlugin::GetInstance()->IsGlobal()) {
    ADP_LOG(INFO) << "[NPUInit] GePlugin global, skip GePlugin init";
    return;
  }
  GePlugin::GetInstance()->Init(init_options_);
  ADP_LOG(INFO) << "[NPUInit] GePlugin init success.";
}

NPUInit::~NPUInit() {
  ADP_LOG(INFO) << "[~NPUInit] NPUInit destructed.";
  int64 unInitStartTime = GetCurrentTimestamp();
  {
    mutex_lock lock{g_mu};
    if (g_npuInitNum > 0) {
      g_npuInitNum--;
    }
    if (g_npuInitNum != 0) {
      int64 unInitEndTime = GetCurrentTimestamp();
      ADP_LOG(INFO) << "[~NPUInit] NPU Shutdown success. [" << ((unInitEndTime - unInitStartTime) / kMicrosToMillis)
                    << " ms]";
      return;
    }
  }
  if (!GePlugin::GetInstance()->IsGlobal()) {
    GePlugin::GetInstance()->Finalize();
    ADP_LOG(INFO) << "[~NPUInit] GePlugin Finalize success";
  } else {
    ADP_LOG(INFO) << "[~NPUInit] GePlugin global, skip GePlugin Finalize";
  }

  int64 unInitEndTime = GetCurrentTimestamp();
  ADP_LOG(INFO) << "[~NPUInit] NPU Shutdown success. [" << ((unInitEndTime - unInitStartTime) / kMicrosToMillis)
                << " ms].";
}

class NPUShutdown : public OpKernel {
 public:
  explicit NPUShutdown(OpKernelConstruction *ctx) : OpKernel(ctx){};
  void Compute(OpKernelContext *ctx) override;
  ~NPUShutdown() override = default;
};
void NPUShutdown::Compute(OpKernelContext *ctx) {
  (void) ctx;
  ADP_LOG(INFO) << "[NPUShutdown] NPUShutdown Compute.";
  {
    mutex_lock lock{g_mu};
    g_npuInitNum = 0;
  }
  if (!GePlugin::GetInstance()->IsGlobal()) {
    GePlugin::GetInstance()->Finalize();
    ADP_LOG(INFO) << "[~NPUShutdown] GePlugin Finalize success";
  } else {
    ADP_LOG(INFO) << "[~NPUShutdown] GePlugin global, skip GePlugin Finalize";
  }
}

REGISTER_KERNEL_BUILDER(Name("NPUInit").Device(DEVICE_CPU), NPUInit);
REGISTER_KERNEL_BUILDER(Name("NPUShutdown").Device(DEVICE_CPU), NPUShutdown);
}  // namespace tensorflow
#endif  // TENSORFLOW_NPU_SYS_CTL_OPS_H_
