/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime_stub.h"
#include <cstring>
#include <chrono>
#include <memory>
#include <atomic>
#include <unordered_set>
#include <securec.h>
#include "runtime/config.h"
#include "runtime/dev.h"
#include "runtime/mem.h"
#include "runtime/rt_mem_queue.h"

#include "tf_adapter/common/adapter_logger.h"

bool g_runtime_stub_mock = true;

void setMockStub(bool g_stub){
  g_runtime_stub_mock = g_stub;
  ADP_LOG(INFO) << "g_runtime_stub_mock = " << g_runtime_stub_mock;
}

struct rtMbuf {
  void *data;
  uint64_t size;
};

void del_fun(rtMbuf *buf_ptr) {
  if (buf_ptr != nullptr) {
    rtMbufFree((rtMbufPtr_t)buf_ptr);
  }
}

static std::unordered_set<std::unique_ptr<rtMbuf, void(*)(rtMbuf *)>> buff_set;

rtError_t rtSetDevice(int32_t device) {
  ADP_LOG(INFO) << "rtSetDevice, g_runtime_stub_mock = " << g_runtime_stub_mock;
  if (!g_runtime_stub_mock) {
    return -1;
  }
  return RT_ERROR_NONE;
}

rtError_t rtDeviceReset(int32_t device) {
  return RT_ERROR_NONE;
}

rtError_t rtMemQueueCreate(int32_t devId, const rtMemQueueAttr_t *queAttr, uint32_t *qid) {
  *qid = 0;
  return RT_ERROR_NONE;
}

rtError_t rtMemQueueDestroy(int32_t devId, uint32_t qid) { return RT_ERROR_NONE; }

rtError_t rtMemQueueInit(int32_t devId) { return RT_ERROR_NONE; }

rtError_t rtMemQueueEnQueue(int32_t devId, uint32_t qid, void *mbuf) { return RT_ERROR_NONE; }

rtError_t rtMemQueueDeQueue(int32_t devId, uint32_t qid, void **mbuf) { return RT_ERROR_NONE; }

rtError_t rtMemQueueQueryInfo(int32_t device, uint32_t qid, rtMemQueueInfo_t *queueInfo) { return RT_ERROR_NONE; }

rtError_t rtMemQueueGrant(int32_t devId, uint32_t qid, int32_t pid, rtMemQueueShareAttr_t *attr) { return RT_ERROR_NONE; }

rtError_t rtMemQueueAttach(int32_t devId, uint32_t qid, int32_t timeout) { return RT_ERROR_NONE; }

rtError_t rtMbufInit(rtMemBuffCfg_t *cfg) { return RT_ERROR_NONE; }

rtError_t rtMemGrpCreate(const char *name, const rtMemGrpConfig_t *cfg) { return RT_ERROR_NONE; }

rtError_t rtMemGrpAddProc(const char *name, int32_t pid, const rtMemGrpShareAttr_t *attr) { return RT_ERROR_NONE; }

rtError_t rtMemGrpAttach(const char *name, int32_t timeout) { return RT_ERROR_NONE; }

rtError_t rtBuffConfirm(void *buff, const uint64_t size) { return RT_ERROR_NONE; }

rtError_t rtBuffAlloc(const uint64_t size, void **const buff) {
  if (size > 0) {
    *buff = new uint8_t[size];
    memset_s(*buff, size, 0, size);
    return RT_ERROR_NONE;
  }
  return -1;
}

rtError_t rtBuffFree(void *buff) {
  auto *buffer = reinterpret_cast<uint8_t*>(buff);
  delete []buffer;
  return RT_ERROR_NONE;
}

rtError_t rtBuffGetInfo(rtBuffGetCmdType type, const void * const inBuff, uint32_t inLen,
                        void * const outBuff, uint32_t * const outLen) {
  return RT_ERROR_NONE;
}
rtError_t rtMbufBuild(void *buff, const uint64_t size, rtMbufPtr_t *mbuf) { return RT_ERROR_NONE; }

rtError_t rtMbufUnBuild(rtMbufPtr_t mbuf, void **buff, uint64_t *const size) { return RT_ERROR_NONE; }

rtError_t rtMbufAlloc(rtMbufPtr_t *mbuf, uint64_t size) {
  rtMbuf *buf = new rtMbuf();
  void *data = malloc(size+256);
  buf->data = data;
  buf->size = size;
  *mbuf = buf;
//  std::cout << "mbuf: " << buf << ", data: " << data << std::endl;
  buff_set.insert(std::unique_ptr<rtMbuf, void(*)(rtMbuf *)>(buf, del_fun));
  return RT_ERROR_NONE;
}

rtError_t rtMbufFree(rtMbufPtr_t mbuf) {
  free(((rtMbuf *)mbuf)->data);
  delete (rtMbuf *)mbuf;
  return RT_ERROR_NONE;
}

rtError_t rtMbufGetBuffAddr(rtMbufPtr_t mbuf, void **databuf) {
  *databuf = ((rtMbuf *)mbuf)->data;
  return RT_ERROR_NONE;
}

rtError_t rtMbufGetBuffSize(rtMbufPtr_t mbuf, uint64_t *size) {
  *size = ((rtMbuf *)mbuf)->size;
  return RT_ERROR_NONE;
}

rtError_t rtGetIsHeterogenous(int32_t *heterogeneous) {
  return RT_ERROR_NONE;
}

rtError_t rtMbufGetPrivInfo(rtMbufPtr_t mbuf, void **priv, uint64_t *size) {
  *priv = ((rtMbuf *)mbuf)->data;
  *size = 256;
  return RT_ERROR_NONE;
}

rtError_t rtFree(void *devPtr) {
  free(devPtr);
  return RT_ERROR_NONE;
}

rtError_t rtMemcpy(void *dst, size_t destMax, const void *src, size_t count, rtMemcpyKind_t kind) {
  (void)std::memcpy(dst, src, count);
  return RT_ERROR_NONE;
}
