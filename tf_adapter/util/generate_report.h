/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSORFLOW_GENERATE_REPORT_H_
#define TENSORFLOW_GENERATE_REPORT_H_

#include "tensorflow/core/graph/graph.h"
// Op will be written to json if it can not sink to device during one excute.
namespace tensorflow {
class GenerateReport {
 public:
  struct Details {
    int code;

    std::string message;
  };
  enum class ReasonCode { TypeNoDefine = 1, TypeGray = 2, ScenarioProblems = 3, NotSupport = 4 };

  static GenerateReport *GetInstance();

  Status AddUnSupportedInfo(const std::string &name, const std::string &type, const Details &infos);

  Status AddUnSupportedInfo(const Node &node, const Details &infos);

  Status SaveUnsupportedInfo();

  ~GenerateReport();

 private:
  GenerateReport();
  struct UnSupportedInfo {
    std::string name;
    std::string type;
    bool is_support = 0;
    Details info_details;
  };
  std::map<std::string, UnSupportedInfo> check_info_map_;
};
}  // namespace tensorflow

#endif
