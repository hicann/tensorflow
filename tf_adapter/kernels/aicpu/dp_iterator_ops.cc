/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_adapter/kernels/aicpu/dp_iterator_ops.h"

#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/common.h"

namespace tensorflow {
namespace data {
void DpMakeIteratorOp::Compute(OpKernelContext *ctx) {
  ADP_LOG(INFO) << "===Begin Computer MakeIterator===";
  CHECK_NOT_NULL(ctx);
  DatasetBase *dataset = nullptr;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  IteratorResource *iterator_resource = nullptr;
  OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &iterator_resource));
  Status s = iterator_resource->SetIteratorFromDataset(ctx, dataset);
  iterator_resource->Unref();
  if (!s.ok()) { ctx->SetStatus(s); }
  ADP_LOG(INFO) << "===End Computer MakeIterator===";
}

namespace {

REGISTER_KERNEL_BUILDER(Name("MakeIterator").Device(DEVICE_CPU).Priority(2).Label("dp"), DpMakeIteratorOp);
REGISTER_KERNEL_BUILDER(Name("MakeIterator").Device(DEVICE_GPU).Priority(1).HostMemory("dataset").Label("dp"),
                        DpMakeIteratorOp);

}  // namespace

}  // namespace data
}  // namespace tensorflow
