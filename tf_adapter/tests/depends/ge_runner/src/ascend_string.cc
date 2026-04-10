/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ascend_string.h"

namespace ge {
AscendString::AscendString(const char* name) {
  if (name != nullptr) {
    name_ = std::shared_ptr<std::string>(new (std::nothrow) std::string(name)); //lint !e1524
    if (name_ == nullptr) {
      fprintf(stderr, "[New][String]AscendString[%s] make shared failed.", name);
    }
  }
}
AscendString::AscendString(const char_t *const name, size_t length) {
  if (name != nullptr) {
    name_ = std::shared_ptr<std::string>(new (std::nothrow) std::string(name, length)); //lint !e1524
    if (name_ == nullptr) {
      fprintf(stderr, "[New][String]AscendString[%s] make shared failed.", name);
    }
  }
}

const char* AscendString::GetString() const {
  if (name_ == nullptr) {
    return nullptr;
  }

  return (*name_).c_str();
}

bool AscendString::operator<(const AscendString& d) const {
  if (name_ == nullptr && d.name_ == nullptr) {
    return false;
  } else if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return false;
  }
  return (*name_ < *(d.name_));
}

bool AscendString::operator>(const AscendString& d) const {
  if (name_ == nullptr && d.name_ == nullptr) {
    return false;
  } else if (name_ == nullptr) {
    return false;
  } else if (d.name_ == nullptr) {
    return true;
  }
  return(*name_ > *(d.name_));
}

bool AscendString::operator==(const AscendString& d) const {
  if (name_ == nullptr && d.name_ == nullptr) {
    return true;
  } else if (name_ == nullptr) {
    return false;
  } else if (d.name_ == nullptr) {
    return false;
  }
  return (*name_ == *(d.name_));
}

bool AscendString::operator<=(const AscendString& d) const {
  if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return false;
  }
  return (*name_ <= *(d.name_));
}

bool AscendString::operator>=(const AscendString& d) const {
  if (d.name_ == nullptr) {
    return true;
  } else if (name_ == nullptr) {
    return false;
  }
  return (*name_ >= *(d.name_));
}

bool AscendString::operator!=(const AscendString& d) const {
  if (name_ == nullptr && d.name_ == nullptr) {
    return false;
  } else if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return true;
  }
  return (*name_ != *(d.name_));
}
}  // namespace ge
