/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GRAPH_ASCEND_STRING_H_
#define INC_EXTERNAL_GRAPH_ASCEND_STRING_H_

#include <string>
#include <memory>
#include <functional>

namespace ge {
class AscendString {
public:
    AscendString() = default;
    ~AscendString() = default;
    inline explicit AscendString(const char *name);
    inline explicit AscendString(const char *name, size_t length);
    inline const char *GetString() const;
    inline size_t GetLength() const;
    inline bool operator<(const AscendString &d) const;
    inline bool operator>(const AscendString &d) const;
    inline bool operator<=(const AscendString &d) const;
    inline bool operator>=(const AscendString &d) const;
    inline bool operator==(const AscendString &d) const;
    inline bool operator!=(const AscendString &d) const;

private:
    std::shared_ptr<std::string> name_;
};

inline AscendString::AscendString(const char *name) {
  if (name != nullptr) {
    try {
      name_ = std::make_shared<std::string>(name);
    } catch (...) {
      name_ = nullptr;
    }
  }
}

inline AscendString::AscendString(const char *name, size_t length) {
  if (name != nullptr) {
    try {
      name_ = std::make_shared<std::string>(name, length);
    } catch (...) {
      name_ = nullptr;
    }
  }
}

inline size_t AscendString::GetLength() const {
  if (name_ == nullptr) {
    return 0;
  }
  return name_->length();
}

inline const char *AscendString::GetString() const {
  if (name_ == nullptr) {
    return "";
  }
  return (*name_).c_str();
}

inline bool AscendString::operator<(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return false;
  }
  return (*name_ < *(d.name_));
}

inline bool AscendString::operator>(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return false;
  } else if (d.name_ == nullptr) {
    return true;
  }
  return (*name_ > *(d.name_));
}

inline bool AscendString::operator==(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return true;
  } else if (name_ == nullptr) {
    return false;
  } else if (d.name_ == nullptr) {
    return false;
  }
  return (*name_ == *(d.name_));
}

inline bool AscendString::operator<=(const AscendString &d) const {
  if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return false;
  }
  return (*name_ <= *(d.name_));
}

inline bool AscendString::operator>=(const AscendString &d) const {
  if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return false;
  }
  return (*name_ >= *(d.name_));
}

inline bool AscendString::operator!=(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return true;
  }
  return (*name_ != *(d.name_));
}
}
#endif  // INC_EXTERNAL_GRAPH_ASCEND_STRING_H_
