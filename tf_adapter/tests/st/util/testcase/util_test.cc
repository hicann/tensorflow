/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_adapter/util/util.h"
#include <vector>
#include "gtest/gtest.h"
#include "graph/def_types.h"
#include "graph/types.h"

namespace tensorflow {
namespace {
class UtilTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(UtilTest , MappingDTStringTensor2DataItemTest) {
  Tensor t0(DT_STRING, TensorShape({}));
  t0.scalar<tstring>()() = "123";

  Tensor t1(DT_STRING, TensorShape({2, 2}));
  for (int i = 0; i < 4; ++i) t1.flat<tstring>()(i) = std::to_string(i);

  std::vector<std::unique_ptr<uint8_t[]>> buff_list;
  tdt::DataItem item0;
  TF_CHECK_OK(MappingDTStringTensor2DataItem(t0, item0, buff_list));
  std::string res = tstring(reinterpret_cast<const char *>(item0.dataPtr_.get()), item0.dataLen_);
  EXPECT_EQ(res, "123");

  tdt::DataItem item1;
  TF_CHECK_OK(MappingDTStringTensor2DataItem(t1, item1, buff_list));
  void *base_ptr = item1.dataPtr_.get();
  for (int i = 0; i < 4; ++i) {
    ge::StringHead *head = reinterpret_cast<ge::StringHead *>(base_ptr + i * sizeof(ge::StringHead));
    std::string tmp = tstring(reinterpret_cast<const char *>(base_ptr + head->addr), head->len);
    EXPECT_EQ(tmp, std::to_string(i));
  }
}

TEST_F(UtilTest , MappingDtStringTensor2AclDataItemTest) {
  Tensor t0(DT_STRING, TensorShape({}));
  t0.scalar<tstring>()() = "123";

  Tensor t1(DT_STRING, TensorShape({2, 2}));
  for (int i = 0; i < 4; ++i) t1.flat<tstring>()(i) = std::to_string(i);

  std::vector<std::unique_ptr<uint8_t[]>> buff_list;
  acltdtDataItem *acl_data0 = nullptr;
  TF_CHECK_OK(MappingDtStringTensor2AclDataItem(t0, acl_data0, buff_list));
  std::string res = tstring(reinterpret_cast<const char *>(acltdtGetDataAddrFromItem(acl_data0)), acltdtGetDataSizeFromItem(acl_data0));
  EXPECT_EQ(res, "123");

  acltdtDataItem *acl_data1 = nullptr;
  TF_CHECK_OK(MappingDtStringTensor2AclDataItem(t1, acl_data1, buff_list));
  void *base_ptr = acltdtGetDataAddrFromItem(acl_data1);
  size_t offset = 4 * sizeof(ge::StringHead);
  for (int i = 0; i < 4; ++i) {
    ge::StringHead *head = reinterpret_cast<ge::StringHead *>(base_ptr + i * sizeof(ge::StringHead));
    std::string tmp = tstring(reinterpret_cast<const char *>(base_ptr + head->addr), head->len);
    offset += head->len;
    EXPECT_EQ(tmp, std::to_string(i));
  }
  EXPECT_EQ(offset, acltdtGetDataSizeFromItem(acl_data1));

  acltdtDestroyDataItem(acl_data0);
  acltdtDestroyDataItem(acl_data1);
}
}
} // end tensorflow
