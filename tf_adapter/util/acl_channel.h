/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_ACL_CHANNEL_H_
#define TENSORFLOW_ACL_CHANNEL_H_

#include "acl/acl_tdt.h"
#include "tensorflow/core/framework/tensor.h"
namespace tensorflow {

Status MappingTfDtypeToAcl(const tensorflow::DataType tf_type, aclDataType &acl_type);

Status MappingAclDtypeToTf(const aclDataType &acl_type, tensorflow::DataType &tf_type);

Status AssembleAclTensor2Tensor(const acltdtDataItem *item, std::vector<Tensor> &tensors, bool call_by_channel_receive);

Status AssembleAclDataset2Tensors(const acltdtDataset *acl_dataset, std::vector<Tensor> &out_tensors,
                                  bool call_by_channel_receive);

Status AssembleTensors2AclDataset(acltdtTensorType acl_type, const std::vector<Tensor> &tensors,
                                  acltdtDataset **output_acl_dataset,
                                  std::vector<std::unique_ptr<uint8_t[]>> &buff_list);

Status AssembleTensors2AclDataset(acltdtTensorType acl_type, const std::vector<Tensor> &tensors,
                                  acltdtDataset *acl_dataset, std::vector<std::unique_ptr<uint8_t[]>> &buff_list);

Status DestroyAclDataset(acltdtDataset *acl_dataset, bool include_data_item = true);

Status RecvTensorByAcl(const acltdtChannelHandle *acl_handle, std::vector<Tensor> &tensors);

Status SendTensorsByAcl(const acltdtChannelHandle *acl_handle, acltdtTensorType acl_type,
                        const std::vector<Tensor> &tensors, bool &need_resend);

Status StopRecvTensorByAcl(acltdtChannelHandle **handle, const std::string &channel_name);

acltdtChannelHandle *CreateAclTdtRecvChannel(uint32_t device_id, const std::string &channel_name,
                                             const size_t capacity);
}  // namespace tensorflow

#endif  // TENSORFLOW_ACL_CHANNEL_H_
