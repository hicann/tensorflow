/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_NPU_OPS_IDENTIFIER_H_
#define TENSORFLOW_NPU_OPS_IDENTIFIER_H_

#include <string>
#include "nlohmann/json_fwd.hpp"

// Sigleton class for check weather or not a tensorflow op is supported by NPU,
// and, weather or not a tensorflow op is performance sensitive on NPU.
class NpuOpsIdentifier {
 public:
  // This method returns different instance Pointers in mixed mode and in the full sink model
  static NpuOpsIdentifier *GetInstance(bool is_mix = false);
  // Determine if the node is supported by NPU. Note that it will behave
  // differently in mixed mode and full sink mode
  bool IsNpuSupported(const char *op_name, const std::string &node_name);
  bool IsNpuSupported(const std::string &op_name, const std::string &node_name);
  // Determine if the node is performance-sensitive on NPU, this should
  // normally be done after calling IsNpuSupported to confirm that the node
  // is supported by NPU. To be on the safe side, it internally performs a
  // check on whether it is supported by NPU, if not, prints an error log,
  // and returns `false`
  bool IsPerformanceSensitive(const char *op);
  bool IsPerformanceSensitive(const std::string &op);

 private:
  NpuOpsIdentifier(bool is_mix, nlohmann::json &ops_info);
  ~NpuOpsIdentifier() = default;
  // Parse and store the ops configuration json file, return num of parsed ops
  int32_t ParseOps(const std::string &f, nlohmann::json &root) const;
  static bool GetOppPluginVendors(const std::string &vendors_config, std::vector<std::string> &vendors);
  static bool IsNewOppPathStruct(const std::string &opp_path);
  static void GetCustomOpPathFromCustomOppPath(std::vector<std::string> &custom_ops_json_path_vec);
  static bool GetCustomOpPath(const std::string &ops_path, std::string &ops_json_path,
                              std::vector<std::string> &custom_ops_json_path_vec);
  const bool is_mix_;
  nlohmann::json &ops_info_;
};

#endif
