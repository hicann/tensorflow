/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mmpa/mmpa_api.h"

INT32 mmAccess2(const CHAR *path_name, INT32 mode) {
  if (path_name == NULL) { return EN_INVALID_PARAM; }

  INT32 ret = access(path_name, mode);
  if (ret != EN_OK) { return EN_ERROR; }
  return EN_OK;
}

INT32 mmGetPid() {
  return (INT32)getpid();
}

INT32 mmMkdir(const CHAR *path_name, mmMode_t mode) {
  if (path_name == NULL) { return EN_INVALID_PARAM; }

  INT32 ret = mkdir(path_name, mode);
  if (ret != EN_OK) { return EN_ERROR; }
  return EN_OK;
}

INT32 mmIsDir(const CHAR *file_name) {
  if (file_name == NULL) { return EN_INVALID_PARAM; }
  struct stat file_stat;
  (void)memset_s(&file_stat, sizeof(file_stat), 0, sizeof(file_stat));
  INT32 ret = lstat(file_name, &file_stat);
  if (ret < MMPA_ZERO) { return EN_ERROR; }

  if (!S_ISDIR(file_stat.st_mode)) { return EN_ERROR; }
  return EN_OK;
}

INT32 mmRealPath(const CHAR *path, CHAR *real_path, INT32 real_path_len) {
  INT32 ret = EN_OK;
  if ((real_path == NULL) || (path == NULL) || (real_path_len < MMPA_MAX_PATH)) {
    return EN_INVALID_PARAM;
  }

  CHAR *ptr = realpath(path, real_path);
  if (ptr == NULL) { ret = EN_ERROR; }
  return ret;
}

VOID *mmDlopen(const CHAR *file_name, INT32 mode) {
  if ((file_name == NULL) || (mode < MMPA_ZERO)) { return NULL; }
  return dlopen(file_name, mode);
}

CHAR *mmDlerror(void) {
  return dlerror();
}

VOID *mmDlsym(VOID *handle, const CHAR *func_name) {
  if ((handle == NULL) || (func_name == NULL)) { return NULL; }
  return dlsym(handle, func_name);
}

INT32 mmDlclose(VOID *handle) {
  if (handle == NULL) { return EN_INVALID_PARAM; }

  INT32 ret = dlclose(handle);
  if (ret != EN_OK) { return EN_ERROR; }
  return EN_OK;
}

INT32 mmSleep(UINT32 milliSecond) {
  return EN_OK;
}

INT32 mmGetTid() {
  INT32 ret = (INT32)syscall(SYS_gettid);

  if (ret < MMPA_ZERO) {
    return EN_ERROR;
  }

  return ret;
}
