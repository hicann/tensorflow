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
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/util/cache_interface.h"

namespace tensorflow {
class InitEmbeddingHashmapV2Op : public OpKernel {
public:
  explicit InitEmbeddingHashmapV2Op(OpKernelConstruction *context) : OpKernel(context) {}
  ~InitEmbeddingHashmapV2Op() override {}
  void Compute(OpKernelContext *context) override {}
};

class DeinitEmbeddingHashmapV2Op : public OpKernel {
public:
  explicit DeinitEmbeddingHashmapV2Op(OpKernelConstruction *context) : OpKernel(context) {}
  ~DeinitEmbeddingHashmapV2Op() override {}
  void Compute(OpKernelContext *context) override {}
};

class TableToResourceV2Op : public OpKernel {
public:
  explicit TableToResourceV2Op(OpKernelConstruction *context) : OpKernel(context) {}
  ~TableToResourceV2Op() override {}
  void Compute(OpKernelContext *context) override {}
};

class EmbeddingHashmapExportOp : public OpKernel {
public:
  explicit EmbeddingHashmapExportOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingHashmapExportOp() override {}
  void Compute(OpKernelContext *context) override {}
};

class EmbeddingHashmapSizeOp : public OpKernel {
public:
  explicit EmbeddingHashmapSizeOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingHashmapSizeOp() override {}
  void Compute(OpKernelContext *context) override {}
};

class EmbeddingHashmapFileSizeOp : public OpKernel {
public:
  explicit EmbeddingHashmapFileSizeOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingHashmapFileSizeOp() override {}
  void Compute(OpKernelContext *context) override {}
};

class EmbeddingHashmapImportOp : public OpKernel {
public:
  explicit EmbeddingHashmapImportOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingHashmapImportOp() override {}
  void Compute(OpKernelContext *context) override {}
};
REGISTER_KERNEL_BUILDER(Name("InitEmbeddingHashmapV2").Device(DEVICE_CPU), InitEmbeddingHashmapV2Op);
REGISTER_KERNEL_BUILDER(Name("DeinitEmbeddingHashmapV2").Device(DEVICE_CPU), DeinitEmbeddingHashmapV2Op);
REGISTER_KERNEL_BUILDER(Name("TableToResourceV2").Device(DEVICE_CPU), TableToResourceV2Op);
REGISTER_KERNEL_BUILDER(Name("EmbeddingHashmapExport").Device(DEVICE_CPU), EmbeddingHashmapExportOp);
REGISTER_KERNEL_BUILDER(Name("EmbeddingHashmapSize").Device(DEVICE_CPU), EmbeddingHashmapSizeOp);
REGISTER_KERNEL_BUILDER(Name("EmbeddingHashmapFileSize").Device(DEVICE_CPU), EmbeddingHashmapFileSizeOp);
REGISTER_KERNEL_BUILDER(Name("EmbeddingHashmapImport").Device(DEVICE_CPU), EmbeddingHashmapImportOp);
}  // namespace tensorflow
