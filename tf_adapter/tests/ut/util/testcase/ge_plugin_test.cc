/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_adapter/util/npu_plugin.h"
#include "tf_adapter/util/npu_attrs.h"
#include "gtest/gtest.h"
#include <stdlib.h>

namespace tensorflow {
namespace {
class GePluginTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(GePluginTest, PluginInitTest_1) {
  std::map<std::string, std::string> init_options;
  setenv("JOB_ID", "1000", true);
  setenv("RANK_SIZE", "1", true);
  setenv("RANK_ID", "0", true);
  setenv("RANK_TABLE_FILE", "rank_table", true);
  setenv("FUSION_TENSOR_SIZE", "524288000", true);
  std::string tf_config = "{'task':{'type':'a'}, 'cluster':{'chief':['1']}}";
  setenv("TF_CONFIG", tf_config.c_str(), true);
  init_options["ge.exec.profilingMode"] = "1";
  init_options["ge.exec.profilingOptions"] = "trace";
  init_options["ge.exec.precision_mode"] = "allow_fp32_to_fp16";
  init_options["ge.autoTuneMode"] = "GA";
  init_options["ge.opDebugLevel"] = "1";
  init_options["ge.jobType"] = "2";
  PluginInit(init_options);
  ASSERT_FALSE(GePlugin::GetInstance()->GetInitOptions().empty());
}

TEST_F(GePluginTest, PluginInitTest_Success) {
  std::map<std::string, std::string> init_options;
  setenv("JOB_ID", "1000", true);
  setenv("RANK_SIZE", "1", true);
  setenv("RANK_ID", "0", true);
  setenv("RANK_TABLE_FILE", "rank_table", true);
  setenv("FUSION_TENSOR_SIZE", "524288000", true);
  std::string tf_config = "{'task':{'type':'a'}, 'cluster':{'chief':['1']}}";
  setenv("TF_CONFIG", tf_config.c_str(), true);
  init_options["ge.exec.profilingMode"] = "1";
  init_options["ge.exec.profilingOptions"] = "trace";
  init_options["ge.exec.precision_mode_v2"] = "fp16";
  init_options["ge.autoTuneMode"] = "GA";
  init_options["ge.opDebugLevel"] = "1";
  init_options["ge.jobType"] = "2";
  PluginInit(init_options);
  ASSERT_FALSE(GePlugin::GetInstance()->GetInitOptions().empty());
}

TEST_F(GePluginTest, PluginInitTest) {
  PluginFinalize();
  std::map<std::string, std::string> init_options;
  setenv("JOB_ID", "1000", true);
  setenv("RANK_SIZE", "1", true);
  setenv("RANK_ID", "0", true);
  setenv("POD_NAME", "0", true);
  setenv("RANK_TABLE_FILE", "rank_table", true);
  setenv("FUSION_TENSOR_SIZE", "524288000", true);
  setenv("ENABLE_HF32_EXECUTION", "1", true);
  std::string tf_config = "{'task':{'type':'a'}, 'cluster':{'chief':['1']}}";
  setenv("TF_CONFIG", tf_config.c_str(), true);
  init_options["ge.exec.profilingMode"] = "1";
  init_options["ge.exec.profilingOptions"] = "trace";
  init_options["ge.exec.precision_mode"] = "allow_fp32_to_fp16";
  init_options["ge.autoTuneMode"] = "GA";
  init_options["ge.opDebugLevel"] = "1";
  init_options["ge.jobType"] = "2";
  PluginInit(init_options);
  ASSERT_FALSE(GePlugin::GetInstance()->GetInitOptions().empty());
}

TEST_F(GePluginTest, PluginInitTest_hccl) {
  std::map<std::string, std::string> init_options;
  unsetenv("RANK_SIZE");
  unsetenv("RANK_TABLE_FILE");
  setenv("JOB_ID", "1000", true);
  setenv("CM_WORKER_SIZE", "1", true);
  setenv("RANK_ID", "0", true);
  setenv("CM_CHIEF_IP", "11", true);
  setenv("CM_CHIEF_PORT", "22", true);
  setenv("CM_CHIEF_DEVICE", "8", true);
  setenv("CM_WORKER_IP", "127.0.0.1", true);
  setenv("FUSION_TENSOR_SIZE", "524288000", true);
  std::string tf_config = "{'task':{'type':'a'}, 'cluster':{'chief':['1']}}";
  setenv("TF_CONFIG", tf_config.c_str(), true);
  init_options["ge.exec.profilingMode"] = "1";
  init_options["ge.exec.profilingOptions"] = "trace";
  init_options["ge.exec.precision_mode"] = "allow_fp32_to_fp16";
  init_options["ge.exec.precision_mode_v2"] = "fp16";
  init_options["ge.autoTuneMode"] = "GA";
  init_options["ge.opDebugLevel"] = "1";
  init_options["ge.jobType"] = "2";
  PluginInit(init_options);
  ASSERT_FALSE(GePlugin::GetInstance()->GetInitOptions().empty());
  PluginFinalize();
}

TEST_F(GePluginTest, InitRdmaPoolOKTest) {
  int32_t ret = InitRdmaPool(1);
  EXPECT_EQ(ret, 0);
}
TEST_F(GePluginTest, InitRdmaPoolFaliedTest) {
  int32_t ret = InitRdmaPool(0);
  EXPECT_EQ(ret, -1);
}
TEST_F(GePluginTest, RegistRdmaRemoteAddrFailedTest) {
  std::vector<std::pair<uint64_t, uint64_t>> var_info;
  int32_t ret = RegistRdmaRemoteAddr(var_info);
  EXPECT_EQ(ret, -1);
}
TEST_F(GePluginTest, RegistRdmaRemoteAddrOKTest) {
  std::vector<std::pair<uint64_t, uint64_t>> var_info;
  std::pair<uint64_t, uint64_t> host_var_info;
  host_var_info.first = 0;
  var_info.push_back(host_var_info);
  int32_t ret = RegistRdmaRemoteAddr(var_info);
  EXPECT_EQ(ret, 0);
}
TEST_F(GePluginTest, GetVarAddrAndSizeOKTest) {
  uint64_t base_addr = 0;
  uint64_t var_size = 0;
  int32_t ret = GetVarAddrAndSize("var", base_addr, var_size);
  EXPECT_EQ(ret, 0);
}
TEST_F(GePluginTest, GetVarAddrAndSizeFailedTest) {
  uint64_t base_addr = 0;
  uint64_t var_size = 0;
  int32_t ret = GetVarAddrAndSize("", base_addr, var_size);
  EXPECT_EQ(ret, -1);
}
TEST_F(GePluginTest, MallocSharedMemFailedTest) {
  std::string var_name;
  std::vector<int64_t> dims;
  ge::DataType data_type = ge::DataType::DT_UNDEFINED;
  uint64_t dev_addr = 0;
  uint64_t memory_size = 0;
  int32_t ret = MallocSharedMem(var_name, dims, data_type, dev_addr, memory_size);
  EXPECT_EQ(ret, -1);
}
TEST_F(GePluginTest, MallocSharedMemOKTest) {
  std::string var_name = "ge";
  std::vector<int64_t> dims;
  ge::DataType data_type = ge::DataType::DT_UNDEFINED;
  uint64_t dev_addr = 0;
  uint64_t memory_size = 0;
  int32_t ret = MallocSharedMem(var_name, dims, data_type, dev_addr, memory_size);
  EXPECT_EQ(ret, 0);
}
TEST_F(GePluginTest, SetDeviceSatModeTest) {
  uint64_t mode = 1U;
  int32_t ret = SetDeviceSatMode(mode);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(GetDeviceSatMode(), 1);
  mode = 2U;
  ret = SetDeviceSatMode(mode);
  EXPECT_EQ(ret, -1);
  EXPECT_EQ(GetDeviceSatMode(), -1);
}
TEST_F(GePluginTest, NpuCloseTest) {
  std::map<std::string, std::string> init_options;
  init_options["ge.jobType"] = "1";
  init_options["ge.tuningPath"] = "./";
  PluginInit(init_options);
  ASSERT_FALSE(GePlugin::GetInstance()->GetInitOptions().empty());
  NpuClose();
}
TEST_F(GePluginTest, RdmaInitAndRegisterFail1Test) {
  std::vector<std::pair<uint64_t, uint64_t>> var_info;
  std::pair<uint64_t, uint64_t> host_var_info;
  host_var_info.first = 0;
  var_info.push_back(host_var_info);
  size_t size = 0;
  int32_t ret = RdmaInitAndRegister(var_info, size);
  EXPECT_EQ(ret, -1);
}
TEST_F(GePluginTest, RdmaInitAndRegisterFail2Test) {
  std::vector<std::pair<uint64_t, uint64_t>> var_info;
  size_t size = 1;
  int32_t ret = RdmaInitAndRegister(var_info, size);
  EXPECT_EQ(ret, -1);
}
TEST_F(GePluginTest, RdmaInitAndRegisterOKTest) {
  std::vector<std::pair<uint64_t, uint64_t>> var_info;
  std::pair<uint64_t, uint64_t> host_var_info;
  host_var_info.first = 0;
  var_info.push_back(host_var_info);
  size_t size = 1;
  int32_t ret = RdmaInitAndRegister(var_info, size);
  EXPECT_EQ(ret, 0);
}

TEST_F(GePluginTest, PluginInitTest_export_compile_stat) {
  std::map<std::string, std::string> init_options;
  init_options["ge.exportCompileStat"] = "1";
  PluginInit(init_options);
  ASSERT_FALSE(GePlugin::GetInstance()->GetInitOptions().empty());
  NpuClose();
}

TEST_F(GePluginTest, PluginInitTest_aicore_num) {
  std::map<std::string, std::string> init_options;
  init_options["ge.aicoreNum"] = "2|2";
  PluginInit(init_options);
  ASSERT_FALSE(GePlugin::GetInstance()->GetInitOptions().empty());
  NpuClose();
}

TEST_F(GePluginTest, PluginInitTest_oo_constant_folding) {
  std::map<std::string, std::string> init_options;
  init_options["ge.oo.constantFolding"] = "true";
  PluginInit(init_options);
  ASSERT_FALSE(GePlugin::GetInstance()->GetInitOptions().empty());
  NpuClose();
}

TEST_F(GePluginTest, PluginInitTest_input_batch_cpy) {
  std::map<std::string, std::string> init_options;
  init_options["ge.inputBatchCpy"] = "true";
  PluginInit(init_options);
  ASSERT_FALSE(GePlugin::GetInstance()->GetInitOptions().empty());
  NpuClose();
}

TEST_F(GePluginTest, PluginInitTest_oo_level) {
  std::map<std::string, std::string> init_options;
  init_options["ge.oo.level"] = "O3";
  PluginInit(init_options);
  ASSERT_FALSE(GePlugin::GetInstance()->GetInitOptions().empty());
  NpuClose();
}

TEST_F(GePluginTest, PluginInitTest_oo_level2) {
  std::map<std::string, std::string> init_options;
  init_options["ge.optimizationSwitch"] = "pass1:on";
  PluginInit(init_options);
  ASSERT_FALSE(GePlugin::GetInstance()->GetInitOptions().empty());
  NpuClose();
}

}
} // end tensorflow
