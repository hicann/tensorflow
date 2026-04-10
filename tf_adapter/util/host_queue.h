/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_HOST_QUEUE_H_
#define TENSORFLOW_HOST_QUEUE_H_

#include "acl/acl_tdt.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
Status HostQueueInit(const std::string &name, const uint32_t &depth, uint32_t &queue_id);

Status MappingTensor2Buff(const acltdtTensorType &acl_type, const std::vector<tensorflow::Tensor> &tensors,
                          void *&buff);

Status HostQueueSendData(uint32_t queue_id, void *buff, bool &need_resend);

void HostQueueFreeBuff(void *buff);

void HostQueueDestroy(const uint32_t &queue_id);

Status HostQueueSetTransId(const uint32_t queue_id, void *&buff);

}  // namespace tensorflow
#endif  // TENSORFLOW_HOST_QUEUE_H_
