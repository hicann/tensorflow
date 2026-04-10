/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_OMG_PARSER_PARSER_FACTORY_H_
#define INC_FRAMEWORK_OMG_PARSER_PARSER_FACTORY_H_

#include "stub/defines.h"

#include "model_parser.h"

namespace domi {
class GE_FUNC_VISIBILITY ModelParserFactory {
 public:
  static ModelParserFactory *Instance() {
    static ModelParserFactory instance;
    return &instance;
  }
  std::shared_ptr<ModelParser> CreateModelParser(const domi::FrameworkType type) {
    return std::make_shared<ModelParser>();
  }

 private:
  ModelParserFactory() = default;
};
}  // namespace domi

#endif
