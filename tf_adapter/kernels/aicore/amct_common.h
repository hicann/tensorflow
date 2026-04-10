/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANTIZE_COMMON_H
#define QUANTIZE_COMMON_H
#include <map>
#include <float.h>
#include <vector>
#include <semaphore.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <pthread.h>
#include <mutex>
#include <iostream>
#include <fcntl.h>
#include <stdio.h>

#include "tensorflow/core/platform/default/integral_types.h"

// Define common constants in quantization
const int BASE = 2;
const float EPSILON = 1e-6f;
const int SHIFT_POW = 15;
const int DEQ_SCALE_BINS = 32;
const int N_LFET_BINS = 24;
const int N_RIGHT_BINS = 56;
const int CIN_DIM = 2;
const int COUT_DIM = 3;
const int NCHW_H_DIM = 2;
const int NCHW_W_DIM = 3;
const int NHWC_H_DIM = 1;
const int NHWC_W_DIM = 2;


// Define the structure of data quantification
template <typename T>
struct QuantInputParam {
  int size;
  const T* in;
  T* out;
  float scale;
  float offset;
  int quant_bits;
};

// Define the structure of weight quantification
template <typename T>
struct WeightQuantInputParam {
  int size;
  const signed char* weight;
  const signed char* offset;
  T* out;
  int channel_in_num;
  int channel_out_num;
  bool channel_wise;
  bool transpose;
};

// Define the structure of data anti quantification
template <typename T>
struct AntiQuantInputParam {
  int size;
  const T* in;
  T* out;
  float scale;
  float offset;
};

// Define the structure of data dequantification
template <typename T>
struct DequantInputParam {
  int area_factor;
  int size;
  const T* input;
  T* out;
  const tensorflow::uint64 *deqscale;
  int channel_num;
  int hw_size;
  bool channel_wise;
  bool transpose;
  std::string data_format;
};

const int SUCCESS = 0;
const int NULL_PTR_ERROR = 1;

#define NULLPTR_CHECK(ptr)                                                                          \
  do {                                                                                              \
    if (ptr == nullptr) {                                                                           \
      return NULL_PTR_ERROR;                                                                        \
    }                                                                                               \
  } while (0)

#define ERROR_CHECK(errorCode)                                                                      \
  do {                                                                                              \
    switch (errorCode) {                                                                            \
      case 1:                                                                                       \
        OP_REQUIRES(context, false, errors::InvalidArgument("Null Ptr ERROR!"));                    \
        break;                                                                                      \
      default:                                                                                      \
        break;                                                                                      \
    }                                                                                               \
  } while (0)

#endif
