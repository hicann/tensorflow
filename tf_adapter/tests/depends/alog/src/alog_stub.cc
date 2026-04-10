/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "toolchain/slog.h"
#include "base/plog.h"
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <fstream>
#include <string.h>
#include <stdarg.h>

#define MSG_LENGTH_STUB 1024
int g_logLevel = 0xffffffff;

void SetLogLevelForC(int logLevel) {
  g_logLevel = logLevel;
}

void ClearLogLevelForC() {
  g_logLevel = 0xffffffff;
}

int CheckLogLevel(int moduleId, int logLevel) {
  if (logLevel >= g_logLevel) {
    return 1;
  } else {
    return 0;
  }
}

void DlogRecord(int moduleId, int level, const char *fmt, ...) {
  int len;
  char msg[MSG_LENGTH_STUB] = {0};
  snprintf(msg, MSG_LENGTH_STUB, "[moduleId:%d] [level:%d] ", moduleId, level);
  va_list ap;

  va_start(ap, fmt);
  len = strlen(msg);
  vsnprintf(msg + len, MSG_LENGTH_STUB - len, fmt, ap);
  va_end(ap);

  printf("\r\n%s", msg);
  fflush(stdout);
  return;
}

int DlogReportFinalize() {
  return 0;
}
