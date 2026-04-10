/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <memory>
#include "npu_device.h"
#include "register/register_base.h"

namespace {
const char *kNpuDeviceName = "/job:localhost/replica:0/task:0/device:NPU:0";
const int kNpuDeviceIndex = 0;
}

class ST_CustomOp : public ::testing::Test {
};
TEST_F(ST_CustomOp, load_custom_op) {
  std::string cust_path = __FILE__;
  cust_path = cust_path.substr(0, cust_path.rfind("/") + 1) + "cust_path/";
  std::string path_tf = cust_path + "framework/tensorflow";
  std::string path_file = path_tf + "/npu_supported_ops.json";
  system(("mkdir -p " + path_tf).c_str());
  system(("echo '{\"AddCustom\": {\"isGray\": false,\"isHeavy\": false}}' > " + path_file).c_str());

  setenv("ASCEND_CUSTOM_OPP_PATH", cust_path.c_str(), 0);
  std::map<std::string, std::string> global_options;
  global_options["ge.jobType"] = "1";
  global_options["ge.tuningPath"] = "./";
  global_options["ge.graph_compiler_cache_dir"] = "./";
  std::map<std::string, std::string> session_options;
  npu::NpuDevice *device = nullptr;
  auto create_status = npu::NpuDevice::CreateDevice(kNpuDeviceName, kNpuDeviceIndex, global_options, session_options, &device);
  auto isSupported = device->Supported("AddCustom");
  CHECK_EQ(isSupported, true);
}
