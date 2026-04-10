/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_COMMON_COMMON_H_
#define TENSORFLOW_COMMON_COMMON_H_

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tf_adapter/common/adapter_logger.h"
#include "acl/acl_base.h"

#define CHECK_NOT_NULL(v)                                                                                              \
  do { \
    if ((v) == nullptr) {                                                                                              \
      ADP_LOG(ERROR) << #v " is nullptr.";                                                                             \
      LOG(ERROR) << #v " is nullptr.";                                                                                 \
      return;                                                                                                          \
    }                                                                                                                  \
  } while (false)

#define REQUIRES_NOT_NULL(v)                                                                                           \
  if ((v) == nullptr) {                                                                                                \
    ADP_LOG(ERROR) << #v " is nullptr.";                                                                               \
    LOG(ERROR) << #v " is nullptr.";                                                                                   \
    return errors::Internal(#v " is nullptr.");                                                                        \
  }

#define REQUIRES_STATUS_OK(s)                                                                                          \
  do { \
    if (!(s).ok()) {                                                                                                   \
      return (s);                                                                                                      \
    }                                                                                                                  \
  } while (false)

#define REQUIRES_ACL_STATUS_OK(expr, interface) \
  do { \
    const auto __ret = (expr); \
    if (__ret != ACL_SUCCESS) { \
      LOG(ERROR) << #interface " is failed, ret code is " <<  __ret; \
      return errors::Internal(#interface " is failed."); \
    } \
  } \
  while (false)

namespace npu {
constexpr int ADAPTER_ENV_MAX_LENTH = 1024 * 1024;
}  // namespace npu

#define ADAPTER_LOG_IF_ERROR(...)                                                                                      \
  do {                                                                                                                 \
    const ::tensorflow::Status status = (__VA_ARGS__);                                                                 \
    if (TF_PREDICT_FALSE(!status.ok()))                                                                                \
      LOG(INFO) << status.ToString();                                                                                  \
  } while (0)

#endif  // TENSORFLOW_COMMON_COMMON_H_
