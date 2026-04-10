/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_TESTUTIL_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_TESTUTIL_H_

#include <numeric>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace test {

// Constructs a flat tensor with 'tensor_vals'.
template <typename T>
Tensor AsTensor(gtl::ArraySlice<T> tensor_vals) {
  Tensor tensor(DataTypeToEnum<T>::value, {static_cast<int64>(tensor_vals.size())});
  std::copy_n(tensor_vals.data(), tensor_vals.size(), tensor.flat<T>().data());
  return tensor;
}

// Constructs a tensor of "shape" with values "tensor_vals".
template <typename T>
Tensor AsTensor(gtl::ArraySlice<T> tensor_vals, const TensorShape& shape) {
  Tensor tensor;
  CHECK(tensor.CopyFrom(AsTensor(tensor_vals), shape));
  return tensor;
}

// Constructs a scalar tensor with 'val'.
template <typename T>
Tensor AsScalar(const T& val) {
  Tensor tensor(DataTypeToEnum<T>::value, {});
  tensor.scalar<T>()() = val;
  return tensor;
}

// Fills in '*tensor' with a sequence of value of func(0), func(1), ...
//   Tensor x(&alloc, DT_FLOAT, TensorShape({2, 2}));
//   test::FillFn<float>(&x, [](int i)->float { return i*i; });
template <typename T>
void FillFn(Tensor* tensor, std::function<T(int)> func) {
  auto flat = tensor->flat<T>();
  for (int i = 0; i < flat.size(); ++i) flat(i) = func(i);
}

// Fills in '*tensor' with 'vals'. E.g.,
//   Tensor x(&alloc, DT_FLOAT, TensorShape({2, 2}));
//   test::FillValues<float>(&x, {11, 21, 21, 22});
template <typename T>
void FillValues(Tensor* tensor, gtl::ArraySlice<T> vals) {
  auto flat_data = tensor->flat<T>();
  CHECK_EQ(flat_data.size(), vals.size());
  if (flat_data.size() > 0) {
    std::copy_n(vals.data(), vals.size(), flat_data.data());
  }
}

// Fills in '*tensor' with a sequence of value of val, val+1, val+2, ...
//   Tensor x(&alloc, DT_FLOAT, TensorShape({2, 2}));
//   test::FillIota<float>(&x, 1.0);
template <typename T>
void FillIota(Tensor* tensor, const T& val) {
  auto flat_data = tensor->flat<T>();
  std::iota(flat_data.data(), flat_data.data() + flat_data.size(), val);
}

// Fills in '*tensor' with 'vals', converting the types as needed.
template <typename T, typename SrcType>
void FillValues(Tensor* tensor, std::initializer_list<SrcType> vals) {
  auto flat_data = tensor->flat<T>();
  CHECK_EQ(flat_data.size(), vals.size());
  if (flat_data.size() > 0) {
    size_t i = 0;
    for (auto it = vals.begin(); it != vals.end(); ++it, ++i) {
      flat_data(i) = T(*it);
    }
  }
}

}  // namespace test
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_TESTUTIL_H_
