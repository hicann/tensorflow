/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_DEVICE_CORE_NPU_MICROS_H
#define NPU_DEVICE_CORE_NPU_MICROS_H

#define NPU_CTX_REQUIRES_OK(CTX, ...)          \
  do {                                         \
    (CTX)->status = (__VA_ARGS__);             \
    if (TF_PREDICT_FALSE(!CTX->status.ok())) { \
      LOG(ERROR) << (CTX)->status.ToString();  \
      return;                                  \
    }                                          \
  } while (0)

#define NPU_CTX_REQUIRES(CTX, EXP, STATUS)    \
  do {                                        \
    if (!TF_PREDICT_TRUE(EXP)) {              \
      (CTX)->status = (STATUS);               \
      LOG(ERROR) << (CTX)->status.ToString(); \
      return;                                 \
    }                                         \
  } while (0)

#define NPU_CTX_REQUIRES_OK_RETURN(CTX, EXP, RET) \
  do {                                            \
    (CTX)->status = (EXP);                        \
    if (TF_PREDICT_FALSE(!(CTX)->status.ok())) {  \
      LOG(ERROR) << (CTX)->status.ToString();     \
      return RET;                                 \
    }                                             \
  } while (0)

#define NPU_CTX_REQUIRES_RETURN(CTX, EXP, STATUS, RET) \
  do {                                                 \
    if (TF_PREDICT_FALSE(!(EXP))) {                    \
      (CTX)->status = (STATUS);                        \
      LOG(ERROR) << (CTX)->status.ToString();          \
      return RET;                                      \
    }                                                  \
  } while (0)

#define NPU_REQUIRES_OK(...)                    \
  do {                                          \
    tensorflow::Status _status = (__VA_ARGS__); \
    if (TF_PREDICT_FALSE(!_status.ok())) {      \
      LOG(ERROR) << _status.ToString();         \
      return _status;                           \
    }                                           \
  } while (0)

#define NPU_REQUIRES(EXP, STATUS)            \
  do {                                       \
    if (!TF_PREDICT_TRUE((EXP))) {           \
      tensorflow::Status _status = (STATUS); \
      LOG(ERROR) << _status.ToString();      \
      return _status;                        \
    }                                        \
  } while (0)

#define NPU_CTX_REQUIRES_GE_OK(CTX, PREFIX, ...)                           \
  do {                                                                     \
    ge::Status _status = (__VA_ARGS__);                                    \
    if (TF_PREDICT_FALSE(_status != ge::SUCCESS)) {                        \
      std::string err_msg(ge::GEGetErrorMsgV2().GetString());              \
      if (err_msg.empty()) {                                               \
        err_msg = "<unknown error> code:" + std::to_string(_status);       \
      }                                                                    \
      CTX->status = tensorflow::errors::Internal(PREFIX, ":\n", err_msg);  \
      LOG(ERROR) << CTX->status.ToString();                                \
      return;                                                              \
    }                                                                      \
  } while (0)

#define NPU_CTX_REQUIRES_GE_OK_RETURN(CTX, PREFIX, EXP, RET)                \
  do {                                                                      \
    ge::Status _status = (EXP);                                             \
    if (TF_PREDICT_FALSE(_status != ge::SUCCESS)) {                         \
      std::string err_msg(ge::GEGetErrorMsgV2().GetString());               \
      if (err_msg.empty()) {                                                \
        err_msg = "<unknown error> code:" + std::to_string(_status);        \
      }                                                                     \
      (CTX)->status = tensorflow::errors::Internal(PREFIX, ":\n", err_msg); \
      LOG(ERROR) << (CTX)->status.ToString();                               \
      return RET;                                                           \
    }                                                                       \
  } while (0)

#define NPU_REQUIRES_ACL_OK(PREFIX, ...)                                              \
  do {                                                                                \
    auto _status = (__VA_ARGS__);                                                     \
    if (TF_PREDICT_FALSE(_status != ACL_ERROR_NONE)) {                                \
      return tensorflow::errors::Internal(PREFIX, ":<unknown error> code:", _status); \
    }                                                                                 \
  } while (0)

#define NPU_LOG_IF_ERROR(...)                                              \
  do {                                                                     \
    const ::tensorflow::Status _status = (__VA_ARGS__);                    \
    if (TF_PREDICT_FALSE(!_status.ok())) LOG(ERROR) << _status.ToString(); \
  } while (0)

#define NPU_REQUIRES_TFE_OK(STATUS)    \
  do {                                 \
    if (TF_GetCode(STATUS) != TF_OK) { \
      return;                          \
    }                                  \
  } while (0)

#define HANDLE_ALL_FORMAT() \
  HANDLE_FORMAT(Nd)         \
  HANDLE_FORMAT(Nchw)       \
  HANDLE_FORMAT(Nc1hwc0)    \
  HANDLE_FORMAT(Fz)         \
  HANDLE_FORMAT(Hz)

#endif  // NPU_DEVICE_CORE_NPU_MICROS_H
