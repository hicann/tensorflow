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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tf_adapter/common/common.h"

namespace tensorflow {
template<typename T>
class FastGeluOp : public tensorflow::OpKernel {
 public:
  explicit FastGeluOp(tensorflow::OpKernelConstruction *context)
    : OpKernel(context) {}
  ~FastGeluOp() {}
  void Compute(tensorflow::OpKernelContext *context) override {
    // Grab the input tensor
    CHECK_NOT_NULL(context);
    const Tensor &input_tensor = context->input(0);

    // Create an output tensor
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
  }
};

class EmbeddingHashTableImportOp : public tensorflow::OpKernel {
public:
  explicit EmbeddingHashTableImportOp(tensorflow::OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingHashTableImportOp() override {}
  void Compute(tensorflow::OpKernelContext *context) override {}
};

REGISTER_KERNEL_BUILDER(Name("EmbeddingHashTableImport")
.Device(tensorflow::DEVICE_CPU), EmbeddingHashTableImportOp);

class EmbeddingHashTableExportOp : public tensorflow::OpKernel {
public:
  explicit EmbeddingHashTableExportOp(tensorflow::OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingHashTableExportOp() override {}
  void Compute(tensorflow::OpKernelContext *context) override {}
};

REGISTER_KERNEL_BUILDER(Name("EmbeddingHashTableExport")
.Device(tensorflow::DEVICE_CPU), EmbeddingHashTableExportOp);

class EmbeddingHashTableLookupOrInsertOp : public tensorflow::OpKernel {
public:
  explicit EmbeddingHashTableLookupOrInsertOp(tensorflow::OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingHashTableLookupOrInsertOp() override {}
  void Compute(tensorflow::OpKernelContext *context) override {}
};

REGISTER_KERNEL_BUILDER(Name("EmbeddingHashTableLookupOrInsert")
.Device(tensorflow::DEVICE_CPU), EmbeddingHashTableLookupOrInsertOp);

class StatelessRandomChoiceWithMaskOp : public tensorflow::OpKernel {
public:
  explicit StatelessRandomChoiceWithMaskOp(tensorflow::OpKernelConstruction *context) : OpKernel(context) {}
  ~StatelessRandomChoiceWithMaskOp() override {}
  void Compute(tensorflow::OpKernelContext *context) override {}
};

REGISTER_KERNEL_BUILDER(Name("StatelessRandomChoiceWithMask")
.Device(tensorflow::DEVICE_CPU), EmbeddingHashTableLookupOrInsertOp);

REGISTER_KERNEL_BUILDER(
  Name("FastGelu")
.
Device(tensorflow::DEVICE_CPU)
.TypeConstraint<float>("T"),
FastGeluOp<float>);

REGISTER_KERNEL_BUILDER(
  Name("FastGelu")
.
Device(tensorflow::DEVICE_CPU)
.TypeConstraint<double>("T"),
FastGeluOp<double>);

REGISTER_KERNEL_BUILDER(
  Name("FastGelu")
.
Device(tensorflow::DEVICE_CPU)
.TypeConstraint<Eigen::half>("T"),
FastGeluOp<Eigen::half>);

template<typename T>
class FastGeluGradOp : public tensorflow::OpKernel {
 public:
  explicit FastGeluGradOp(tensorflow::OpKernelConstruction *context)
    : OpKernel(context) {}
  ~FastGeluGradOp() override = default;
  void Compute(tensorflow::OpKernelContext *context) override {
    // Grab the grad input tensor
    CHECK_NOT_NULL(context);
    const Tensor &grad_input_tensor = context->input(0);
    auto grad_input = grad_input_tensor.flat<T>();

    // Grab the input tensor
    const Tensor &input_tensor = context->input(1);
    auto input = input_tensor.flat<T>();

    OP_REQUIRES(
      context, grad_input.size() == input.size(),
      errors::InvalidArgument("grad_input size is not equal input size"));

    // Create an output tensor
    Tensor *grad_output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_input_tensor.shape(),
                                                     &grad_output_tensor));
  }
};

class StatelessDropoutOp : public tensorflow::OpKernel {
public:
  explicit StatelessDropoutOp(tensorflow::OpKernelConstruction *context) : OpKernel(context) {}
  ~StatelessDropoutOp() override {}
  void Compute(tensorflow::OpKernelContext *context) override {}
};

REGISTER_KERNEL_BUILDER(Name("StatelessDropout")
.Device(tensorflow::DEVICE_CPU), StatelessDropoutOp);

REGISTER_KERNEL_BUILDER(
  Name("FastGeluGrad")
.
Device(tensorflow::DEVICE_CPU)
.TypeConstraint<float>("T"),
FastGeluGradOp<float>);

REGISTER_KERNEL_BUILDER(
  Name("FastGeluGrad")
.
Device(tensorflow::DEVICE_CPU)
.TypeConstraint<double>("T"),
FastGeluGradOp<double>);

REGISTER_KERNEL_BUILDER(
  Name("FastGeluGrad")
.
Device(tensorflow::DEVICE_CPU)
.TypeConstraint<Eigen::half>("T"),
FastGeluGradOp<Eigen::half>);

class InitEmbeddingHashTableOp : public tensorflow::OpKernel {
public:
  explicit InitEmbeddingHashTableOp(tensorflow::OpKernelConstruction *context) : OpKernel(context) {}
  ~InitEmbeddingHashTableOp() override {}
  void Compute(tensorflow::OpKernelContext *context) override {}
};

REGISTER_KERNEL_BUILDER(Name("InitEmbeddingHashTable").Device(tensorflow::DEVICE_CPU), InitEmbeddingHashTableOp);

class EmbeddingHashTableApplyAdamWOp : public tensorflow::OpKernel {
public:
  explicit EmbeddingHashTableApplyAdamWOp(tensorflow::OpKernelConstruction *context)
    : OpKernel(context) {}
  ~EmbeddingHashTableApplyAdamWOp() override {}
  void Compute(tensorflow::OpKernelContext *context) override {}
};

REGISTER_KERNEL_BUILDER(Name("EmbeddingHashTableApplyAdamW").Device(tensorflow::DEVICE_CPU),
  EmbeddingHashTableApplyAdamWOp);

class EmbeddingHashTableEvictOp : public tensorflow::OpKernel {
public:
  explicit EmbeddingHashTableEvictOp(tensorflow::OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingHashTableEvictOp() override {}
  void Compute(tensorflow::OpKernelContext *context) override {}
};

REGISTER_KERNEL_BUILDER(Name("EmbeddingHashTableEvict")
.Device(tensorflow::DEVICE_CPU), EmbeddingHashTableEvictOp);
}  // namespace tensorflow
