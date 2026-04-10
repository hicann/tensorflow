/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/util/env_var.h"

#include "npu_global.h"
#include "npu_hdc.h"

using namespace tensorflow;
namespace {
class DeregisterCallbackGuarder {
 public:
  explicit DeregisterCallbackGuarder(std::function<void()> done) : done_(done) {}
  ~DeregisterCallbackGuarder() { done_(); }

 private:
  std::function<void()> done_;
};
}  // namespace
namespace npu {
class IteratorH2D : public OpKernel {
 public:
  explicit IteratorH2D(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("channel_name", &channel_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ids", &device_ids_));
  }

  ~IteratorH2D() override { DestroyTdtChannels(); }

  void Compute(OpKernelContext *ctx) override {
    if (!initialized_.exchange(true)) {
      std::stringstream ss;
      for (auto device_id : device_ids_) {
        ss << device_id << " ";
      }
      npu::global::GlobalHdcChannel::GetInstance().Get(channel_name_, channels_);
      LOG(INFO) << channels_.size() << " hdc channels for iterator resource " << channel_name_ << " created.";
      if (channels_.empty()) {
        use_global_channel_ = false;
        channels_.resize(device_ids_.size());
        for (size_t i = 0; i < device_ids_.size(); i++) {
          OP_REQUIRES_OK(ctx,
                         npu::HdcChannel::Create(static_cast<uint32_t>(device_ids_[i]), channel_name_, &channels_[i]));
        }
      }
      LOG(INFO) << "Hdc channel for iterator resource " << channel_name_ << " to device ["
                << ss.str().substr(0, ss.str().size() - 1) << "] created.";
    }

    CancellationManager *cm = ctx->cancellation_manager();
    CancellationToken token = cm->get_cancellation_token();
    bool cancelled = !cm->RegisterCallback(token, [this]() { DestroyTdtChannels(); });
    if (cancelled) {
      ctx->SetStatus(tensorflow::errors::Internal("Iterator resource ", channel_name_, " consume after destroyed"));
      return;
    }
    DeregisterCallbackGuarder guarder([cm, token]() { (void)cm->DeregisterCallback(token); });

    data::IteratorResource *iterator;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator));
    core::ScopedUnref unref_iterator(iterator);

    int64_t nums = ctx->input(1).flat<int64>()(0);
    OP_REQUIRES(ctx, nums >= 0, tensorflow::errors::InvalidArgument(channel_name_, " invalid consume nums ", nums));

    int64_t consumed = 0;
    bool end_of_sequence = false;
    std::vector<Tensor> components;
    while (nums == 0 || consumed++ < nums) {
      components.clear();

      Status status = iterator->GetNext(ctx, &components, &end_of_sequence);
      if (!status.ok()) {
        for (const auto &channel : channels_) {
          OP_REQUIRES_OK(ctx, channel->NotifyAbnormal());
        }
        ctx->SetStatus(status);
        return;
      } else if (end_of_sequence) {
        for (const auto &channel : channels_) {
          OP_REQUIRES_OK(ctx, channel->NotifyFinish());
        }
        ctx->SetStatus(errors::OutOfRange("Iterator resource ", channel_name_, " reach end of sequence"));
        return;
      }

      for (const auto &channel : channels_) {
        status = channel->SendTensors(components);
        if (!status.ok()) {  // suppress warning message for OP_REQUIRES_OK
          ctx->SetStatus(status);
          return;
        }
      }
    }
  }

 private:
  void DestroyTdtChannels() {
    if (use_global_channel_) {
      return;
    }
    for (const auto &channel : channels_) {
      channel->Destroy();
    }
    channels_.clear();
  }
  bool use_global_channel_ {true};
  std::atomic_bool initialized_ {false};
  std::string channel_name_;
  std::vector<int> device_ids_;
  std::vector<std::shared_ptr<npu::HdcChannel>> channels_;
};

REGISTER_KERNEL_BUILDER(Name("IteratorH2D").Device(DEVICE_CPU).Priority(3), IteratorH2D);
}  // namespace npu
