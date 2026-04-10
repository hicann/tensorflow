/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/common/compat_tf1_tf2.h"

namespace tensorflow {
namespace data {
namespace {
class DPGroupDatasetOp : public DatasetOpKernel {
public:
  explicit DPGroupDatasetOp(OpKernelConstruction *ctx) : DatasetOpKernel(ctx) {
    CHECK_NOT_NULL(ctx);
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }
  ~DPGroupDatasetOp() override = default;
  void MakeDataset(OpKernelContext *ctx, DatasetBase **output) override {
    CHECK_NOT_NULL(ctx);
    CHECK_NOT_NULL(output);
    std::vector<DatasetBase *> inputs;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      DatasetBase *input = nullptr;
      OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(i), &input));
      inputs.push_back(input);
    }
    *output = new (std::nothrow) Dataset(ctx, inputs, output_types_, output_shapes_);
    OP_REQUIRES(ctx, *output != nullptr, errors::Internal("Failed new dataset of DPGroupDatasetOp"));
  }

private:
  class Dataset : public DatasetBase {
  public:
    explicit Dataset(OpKernelContext *ctx, const std::vector<DatasetBase *> &inputs, const DataTypeVector &output_types,
                     const std::vector<PartialTensorShape> &output_shapes)
      : DatasetBase(DatasetContext(ctx)), inputs_(inputs) {
      for (const auto &input : inputs_) { input->Ref(); }
      output_types_.insert(output_types_.end(), output_types.begin(), output_types.end());
      output_shapes_.insert(output_shapes_.end(), output_shapes.begin(), output_shapes.end());
    }

    ~Dataset() override {
      for (const auto &input : inputs_) { input->Unref(); }
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(const string &prefix) const override {
      return absl::make_unique<Iterator>(Iterator::Params({this, prefix + "::GEOP"}));
    }

    const DataTypeVector &output_dtypes() const override { return output_types_; }

    const std::vector<PartialTensorShape> &output_shapes() const override { return output_shapes_; }

    string DebugString() const override { return "DPGroupDatasetOp::Dataset"; }

#ifdef TF_VERSION_TF2
    Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
      for (const auto &input : inputs_) { inputs->push_back(input); }
      return Status::OK();
    }
#endif

    STATUS_FUNCTION_ONLY_TF2(CheckExternalState() const override);

  protected:
    Status AsGraphDefInternal(SerializationContext *ctx, DatasetGraphDefBuilder *b, Node **output) const override {
#ifdef TF_VERSION_TF2
      return errors::Unimplemented(DebugString(), " does not support serialization");
#else
      return Status::OK();
#endif
    }

  private:
    class Iterator : public DatasetIterator<Dataset> {
    public:
      explicit Iterator(const Params &para) : DatasetIterator<Dataset>(para) {}
      ~Iterator() override = default;
      Status Initialize(IteratorContext *ctx) override {
        REQUIRES_NOT_NULL(ctx);
        ADP_LOG(INFO) << "Start to initialize iterator of DPGroupDatasetOp";
        mutex_lock l(mu_);
        try {
          input_impls_.resize(dataset()->inputs_.size());
        } catch (...) { return errors::InvalidArgument("input impls resize failed."); }
        for (size_t i = 0; i < input_impls_.size(); ++i) {
#ifdef TF_VERSION_TF2
          TF_RETURN_IF_ERROR(
            dataset()->inputs_[i]->MakeIterator(ctx, this, prefix() + "[" + std::to_string(i) + "]", &input_impls_[i])
          );
#else
          TF_RETURN_IF_ERROR(
            dataset()->inputs_[i]->MakeIterator(ctx, prefix() + "[" + std::to_string(i) + "]", &input_impls_[i])
          );
#endif
        }
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext *ctx, std::vector<Tensor> *out_tensors, bool *end_of_sequence) override {
        *end_of_sequence = true;
        return Status::OK();
      }

    protected:
      STATUS_FUNCTION_ONLY_TF2(SaveInternal(SerializationContext *ctx, IteratorStateWriter *writer) override);
      STATUS_FUNCTION_ONLY_TF1(SaveInternal(IteratorStateWriter *writer) override);

      Status RestoreInternal(IteratorContext *ctx, IteratorStateReader *reader) override { return Status::OK(); }

    private:
      mutex mu_;
      std::vector<std::unique_ptr<IteratorBase>> input_impls_ GUARDED_BY(mu_);
    };
    const std::vector<DatasetBase *> inputs_;
    DataTypeVector output_types_;
    std::vector<PartialTensorShape> output_shapes_;
  };
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("DPGroupDataset").Device(DEVICE_CPU), DPGroupDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
