/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_logger.h"
#include <pthread.h>
#include "npu_micros.h"

namespace npu {
tensorflow::Status NpuStdoutReceiver::Start() {
  std::unique_lock<std::mutex> lk(mu_);
  if (started_) {
    LOG(INFO) << "Npu stdout receiver has already started on device " << device_id_;
    return tensorflow::Status::OK();
  }
  const static size_t kNpuCerrChannelCapacity = 32U;
  NPU_REQUIRES_OK(npu::HdcChannel::Create(device_id_, "_npu_log", kNpuCerrChannelCapacity, &channel_));
  std::thread t([this]() {
    (void)pthread_setname_np(pthread_self(), "tfa_log_recv");
    while (true) {
      std::vector<tensorflow::Tensor> tensors;
      auto status = channel_->RecvTensors(tensors);
      if (stopping_) {
        DLOG() << "Exit npu stdout receive thread of device " << device_id_ << " as stopping";
        break;
      }
      if (!status.ok()) {
        LOG(ERROR) << "Npu stdout receiver on device " << device_id_ << " error " << status.error_message();
        break;
      }
      for (auto &tensor : tensors) {
        LOG(INFO) << "[NPU:" << device_id_ << "] " << tensor.DebugString();
      }
    }
    DLOG() << "Npu stdout receive thread of device " << device_id_ << " exited";
  });
  thread_.swap(t);
  started_ = true;
  LOG(INFO) << "Npu stdout receiver of device " << device_id_ << " started";
  return tensorflow::Status::OK();
}

tensorflow::Status NpuStdoutReceiver::Stop() {
  std::unique_lock<std::mutex> lk(mu_);
  if (!started_) {
    return tensorflow::Status::OK();
  }
  LOG(INFO) << "Stopping npu stdout receiver of device " << device_id_;
  (void)stopping_.exchange(true);
  channel_->Destroy();
  thread_.join();
  started_ = false;
  DLOG() << "Npu stdout receiver of device " << device_id_ << " stopped";
  return tensorflow::Status::OK();
}
}  // namespace npu
