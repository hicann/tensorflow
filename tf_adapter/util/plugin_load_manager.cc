/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_adapter/util/plugin_load_manager.h"
#include <climits>
#include <dlfcn.h>
#include "tf_adapter/common/adapter_logger.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
void *PluginLoadManager::DlOpen(const std::string &path) {
  void *handle = dlopen(path.c_str(), RTLD_NOW);
  if (handle == nullptr) {
    ADP_LOG(WARNING) << "dlopen failed, reason:" << dlerror();
  }
  return handle;
}

void *PluginLoadManager::DlSym(void *handle, const std::string &func_name) {
  if (handle == nullptr) {
    ADP_LOG(WARNING) << "handle is null, not valid!";
    LOG(WARNING) << "handle is null, not valid!";
    return nullptr;
  }
  void *func = dlsym(handle, func_name.c_str());
  if (func == nullptr) {
    ADP_LOG(WARNING) << "get func[" << func_name << "] failed, reason:" << dlerror();
  }
  return func;
}

std::string PluginLoadManager::GetTFPluginRealPath() {
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(&PluginLoadManager::GetTFPluginRealPath), &dl_info) == 0) {
    ADP_LOG(WARNING) << "can not get tf-adapter base path!";
    LOG(WARNING) << "can not get tf-adapter base path!";
    return string();
  } else {
    std::string so_path = dl_info.dli_fname;
    char path[PATH_MAX] = {0};
    if (so_path.length() >= PATH_MAX) {
      ADP_LOG(WARNING) << "The shared library file path is too long!";
      LOG(WARNING) << "The shared library file path is too long!";
      return string();
    }
    if (realpath(so_path.c_str(), path) == nullptr) {
      ADP_LOG(WARNING) << "Failed to get realpath of " << so_path;
      LOG(WARNING) << "Failed to get realpath of " << so_path;
      return string();
    }
    so_path = path;
    so_path = so_path.substr(0, so_path.rfind('/') + 1);
    ADP_LOG(INFO) << "tf-plugin base path is: " << so_path;
    return so_path;
  }
}
}  // namespace tensorflow
