/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensorflow/core/framework/op_kernel.h"
#include "tf_adapter/common/adapter_logger.h"

namespace tensorflow {
class HcomAllReduceOpKernel : public OpKernel {
 public:
  explicit HcomAllReduceOpKernel(OpKernelConstruction *context) : OpKernel(context) {}
  ~HcomAllReduceOpKernel() {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "HcomAllReduceOp Compute."; }
};

REGISTER_KERNEL_BUILDER(Name("HcomAllReduce").Device(DEVICE_CPU), HcomAllReduceOpKernel);

class HcomAllGatherOpKernel : public OpKernel {
 public:
  explicit HcomAllGatherOpKernel(OpKernelConstruction *context) : OpKernel(context) {}
  ~HcomAllGatherOpKernel() {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "HcomAllGatherOp Compute."; }
};

REGISTER_KERNEL_BUILDER(Name("HcomAllGather").Device(DEVICE_CPU), HcomAllGatherOpKernel);

class HcomBroadcastOpKernel : public OpKernel {
 public:
  explicit HcomBroadcastOpKernel(OpKernelConstruction *context) : OpKernel(context) {}
  ~HcomBroadcastOpKernel() {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "HcomBroadcastOp Compute."; }
};

REGISTER_KERNEL_BUILDER(Name("HcomBroadcast").Device(DEVICE_CPU), HcomBroadcastOpKernel);

class HcomReduceOpKernel : public OpKernel {
 public:
  explicit HcomReduceOpKernel(OpKernelConstruction *context) : OpKernel(context) {}
  ~HcomReduceOpKernel() {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "HcomReduceOp Compute."; }
};

REGISTER_KERNEL_BUILDER(Name("HcomReduce").Device(DEVICE_CPU), HcomReduceOpKernel);

class HcomReduceScatterOpKernel : public OpKernel {
 public:
  explicit HcomReduceScatterOpKernel(OpKernelConstruction *context) : OpKernel(context) {}
  ~HcomReduceScatterOpKernel() {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "HcomReduceScatterOp Compute."; }
};

REGISTER_KERNEL_BUILDER(Name("HcomReduceScatter").Device(DEVICE_CPU), HcomReduceScatterOpKernel);

class HcomSendOpKernel : public OpKernel {
 public:
  explicit HcomSendOpKernel(OpKernelConstruction *context) : OpKernel(context) {}
  ~HcomSendOpKernel() {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "HcomSendOpKernel Compute."; }
};

REGISTER_KERNEL_BUILDER(Name("HcomSend").Device(DEVICE_CPU), HcomSendOpKernel);

class HcomReceiveOpKernel : public OpKernel {
 public:
  explicit HcomReceiveOpKernel(OpKernelConstruction *context) : OpKernel(context) {}
  ~HcomReceiveOpKernel() {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "HcomReceiveOpKernel Compute."; }
};

REGISTER_KERNEL_BUILDER(Name("HcomReceive").Device(DEVICE_CPU), HcomReceiveOpKernel);

class HcomRemoteReadOpKernel : public OpKernel {
public:
  explicit HcomRemoteReadOpKernel(OpKernelConstruction* context) : OpKernel(context) {}
  ~HcomRemoteReadOpKernel() {}
  void Compute(OpKernelContext* context) override
  {
    ADP_LOG(INFO) << "HcomRemoteReadOpKernel Compute.";
  }
};

REGISTER_KERNEL_BUILDER(Name("HcomRemoteRead").Device(DEVICE_CPU), HcomRemoteReadOpKernel);

class HcomRemoteRefReadOpKernel : public OpKernel {
public:
  explicit HcomRemoteRefReadOpKernel(OpKernelConstruction* context) : OpKernel(context) {}
  ~HcomRemoteRefReadOpKernel() {}
  void Compute(OpKernelContext* context) override
  {
    ADP_LOG(INFO) << "HcomRemoteRefRead Compute.";
  }
};

REGISTER_KERNEL_BUILDER(Name("HcomRemoteRefRead").Device(DEVICE_CPU), HcomRemoteRefReadOpKernel);

class HcomRemoteWriteKernel : public OpKernel {
public:
  explicit HcomRemoteWriteKernel(OpKernelConstruction* context) : OpKernel(context) {}
  ~HcomRemoteWriteKernel() {}
  void Compute(OpKernelContext* context) override
  {
    ADP_LOG(INFO) << "HcomRemoteWriteKernel Compute.";
  }
};

REGISTER_KERNEL_BUILDER(Name("HcomRemoteWrite").Device(DEVICE_CPU), HcomRemoteWriteKernel);

class HcomRemoteScatterWriteOpKernel : public OpKernel {
public:
  explicit HcomRemoteScatterWriteOpKernel(OpKernelConstruction* context) : OpKernel(context) {}
  ~HcomRemoteScatterWriteOpKernel() {}
  void Compute(OpKernelContext* context) override
  {
    ADP_LOG(INFO) << "HcomRemoteScatterWrite Compute.";
  }
};

REGISTER_KERNEL_BUILDER(Name("HcomRemoteScatterWrite").Device(DEVICE_CPU), HcomRemoteScatterWriteOpKernel);

class HcomGatherAllToAllVOpKernel : public OpKernel {
public:
  explicit HcomGatherAllToAllVOpKernel(OpKernelConstruction* context) : OpKernel(context) {}
  ~HcomGatherAllToAllVOpKernel() {}
  void Compute(OpKernelContext* context) override
  {
    ADP_LOG(INFO) << "HcomGatherAllToAllV Compute.";
  }
};

REGISTER_KERNEL_BUILDER(Name("HcomGatherAllToAllV").Device(DEVICE_CPU), HcomGatherAllToAllVOpKernel);

class HcomAllToAllVOpKernel : public OpKernel {
public:
  explicit HcomAllToAllVOpKernel(OpKernelConstruction* context) : OpKernel(context) {}
  ~HcomAllToAllVOpKernel() {}
  void Compute(OpKernelContext* context) override
  {
    ADP_LOG(INFO) << "HcomAllToAllV Compute.";
  }
};

REGISTER_KERNEL_BUILDER(Name("HcomAllToAllV").Device(DEVICE_CPU), HcomAllToAllVOpKernel);

class HcomAllToAllVCOpKernel : public OpKernel {
public:
  explicit HcomAllToAllVCOpKernel(OpKernelConstruction* context) : OpKernel(context) {}
  ~HcomAllToAllVCOpKernel() {}
  void Compute(OpKernelContext* context) override
  {
    ADP_LOG(INFO) << "HcomAllToAllVC Compute.";
  }
};
REGISTER_KERNEL_BUILDER(Name("HcomAllToAllVC").Device(DEVICE_CPU), HcomAllToAllVCOpKernel);
} // namespace tensorflow
