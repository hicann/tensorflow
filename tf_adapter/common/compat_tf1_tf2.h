/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_COMMON_COMPAT_TF1_TF2_H_
#define TENSORFLOW_COMMON_COMPAT_TF1_TF2_H_

#include "tensorflow/core/platform/tstring.h"

namespace npu {
namespace compat_tf1_tf2 {
#ifdef TF_VERSION_TF2
using string = tensorflow::tstring;
#else
using string = tensorflow::string;
#endif
}  // namespace compat_tf1_tf2
}  // namespace npu

#if defined(TF_VERSION_TF2)
#define STATUS_FUNCTION_ONLY_TF2(F)                                                                                    \
  tensorflow::Status F {                                                                                               \
    return tensorflow::Status::OK();                                                                                   \
  }
#else
#define STATUS_FUNCTION_ONLY_TF2(F)
#endif

#if !defined(TF_VERSION_TF2)
#define STATUS_FUNCTION_ONLY_TF1(F)                                                                                    \
  tensorflow::Status F {                                                                                               \
    return tensorflow::Status::OK();                                                                                   \
  }
#else
#define STATUS_FUNCTION_ONLY_TF1(F)
#endif

#if defined(TF_VERSION_TF2)
#define VOID_FUNCTION_ONLY_TF2(F)                                                                                      \
  void F {}
#else
#define VOID_FUNCTION_ONLY_TF2(F)
#endif

#if !defined(TF_VERSION_TF2)
#define VOID_FUNCTION_ONLY_TF1(F)                                                                                      \
  void F {}
#else
#define VOID_FUNCTION_ONLY_TF1(F)
#endif

#endif  // TENSORFLOW_COMMON_COMPAT_TF1_TF2_H_
