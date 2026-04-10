/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <securec.h>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tf_adapter/common/adapter_logger.h"
#include "acl/error_codes/rt_error_codes.h"
#include "runtime/rt_mem_queue.h"
#include "infershape_util.h"
#include "runtime/dev.h"

namespace tensorflow {
namespace {
enum class BuffType {
  kBuffTypeNormal,
  kBuffTypeMbufData
};
struct BuffTypeInfo {
  BuffType type;
};
} // namespace
static mutex mtx;
#ifdef TF_VERSION_TF2
static bool isMbufInit TF_GUARDED_BY(mtx) = false;
#else
static bool isMbufInit GUARDED_BY(mtx) = false;
#endif
static const int32 kTimeOut = 3000;
const int64_t kRuntimeTensorDescSize = 1024UL;

bool IsMbufAllocatorEnabled() {
  tensorflow::int64 enable_mbuf_allocator = 0;
  (void)tensorflow::ReadInt64FromEnvVar("ENABLE_MBUF_ALLOCATOR", 0, &enable_mbuf_allocator);
  ADP_LOG(INFO) << "enable_mbuf_allocator:" << enable_mbuf_allocator;
  return enable_mbuf_allocator == 1;
}

Status MemGroupInit(string &group_name) {
  rtMemGrpConfig_t cfg {};
  auto ret = rtMemGrpCreate(group_name.c_str(), &cfg);
  if (ret != RT_ERROR_NONE) {
    return errors::Internal("Call rtMemGrpCreate failed, ret:", ret, ", group_name:", group_name.c_str());
  }
  ADP_LOG(INFO) << "rtMemGrpCreate success.";

  rtMemGrpShareAttr_t attr {};
  memset_s(&attr, sizeof(rtMemGrpShareAttr_t), 0, sizeof(rtMemGrpShareAttr_t));
  attr.admin = 1;
  attr.alloc = 1;
  attr.read = 1;
  attr.write = 1;
  int32_t pid = getpid();
  ret = rtMemGrpAddProc(group_name.c_str(), pid, &attr);
  if (ret != RT_ERROR_NONE && ret != ACL_ERROR_RT_REPEATED_INIT) {
    return errors::Internal("Call rtMemGrpAddProc failed, ret:", ret, ", group_name:", group_name.c_str());
  }
  ADP_LOG(INFO) << "rtMemGrpAddProc success.";

  ret = rtMemGrpAttach(group_name.c_str(), kTimeOut);
  if (ret != RT_ERROR_NONE) {
    return errors::Internal("Call rtMemGrpAttach failed, ret:", ret, ", group_name:", group_name.c_str());
  }

  ADP_LOG(INFO) << "MemGroupInit success, group_name:" << group_name.c_str();
  return Status::OK();
}

Status MbufInit() {
  mutex_lock lock(mtx);
  if (isMbufInit) {
    ADP_LOG(INFO) << "MbufInit function is already executed.";
    return Status::OK();
  }
  ADP_LOG(INFO) << "MemGroupInit begin.";
  std::string group_name = std::string("DM_QS_GROUP_") + std::to_string(getpid());
  auto ret = MemGroupInit(group_name);
  if (ret != Status::OK()) {
    return errors::Internal("Call MemGroupInit failed, ret:", ret, ", group_name:", group_name.c_str());
  }
  ADP_LOG(INFO) << "MemGroupInit success.";

  rtMemBuffCfg_t buff_cfg = {};
  auto rt_error = rtMbufInit(&buff_cfg);
  if ((rt_error != ACL_RT_SUCCESS) && (rt_error != ACL_ERROR_RT_REPEATED_INIT)) {
    return errors::Internal("Call rtMbufInit failed, ret:", ret, ", group_name:", group_name.c_str());
  }
  isMbufInit = true;
  ADP_LOG(INFO) << "MbufInit success.";
  return Status::OK();
}

namespace {
class MbufAllocator : public Allocator {
public:
    MbufAllocator() = default;
    ~MbufAllocator() override = default;
    string Name() override { return "MbufAllocator"; }

    void *AllocateRaw(size_t alignment, size_t num_bytes) override {
      ADP_LOG(INFO) << "MbufAllocator AllocateRaw begin, size:" << num_bytes;
      void *mbuf_data = nullptr;
      auto rt_error = rtBuffAlloc(num_bytes + kRuntimeTensorDescSize, &mbuf_data);
      if (rt_error != RT_ERROR_NONE) {
        ADP_LOG(ERROR) << "Call rtBuffAlloc with size:" << num_bytes << " failed, ret:" << rt_error;
        return nullptr;
      }

      ADP_LOG(INFO) << "MbufAllocator AllocateRaw success, size:" << num_bytes
                    << ", mbuf_data:" << mbuf_data
                    << ", return ptr: " << PtrToDataAddr(mbuf_data, kRuntimeTensorDescSize);
      return PtrToDataAddr(mbuf_data, kRuntimeTensorDescSize);
    }

    void *AllocateRaw(size_t alignment, size_t num_bytes,
                      const AllocationAttributes &allocation_attr) override {
      return AllocateRaw(alignment, num_bytes);
    }

    void DeallocateRaw(void *ptr) override {
      if ((ptr == nullptr) || !IsBuffTypeNomal(DataAddrToPtr(ptr, kRuntimeTensorDescSize))) {
        return;
      }
      ADP_LOG(INFO) << "MbufAllocator DeallocateRaw begin, ptr:" << ptr;
      auto free_ret = rtBuffFree(DataAddrToPtr(ptr, kRuntimeTensorDescSize));
      if (free_ret != RT_ERROR_NONE) {
        ADP_LOG(ERROR) << "rtBuffFree failed, ret:" << free_ret << ", ptr:" << ptr;
        return;
      }
      ADP_LOG(INFO) << "MbufAllocator DeallocateRaw success, ptr:" << ptr
                    << ", ptr-1024:" << DataAddrToPtr(ptr, kRuntimeTensorDescSize);
    }

    static void *PtrToDataAddr(void *ptr, int64_t offset) {
      return reinterpret_cast<void *>(reinterpret_cast<uint8_t*>(ptr) + offset);
    }

    static void *DataAddrToPtr(void *ptr, int64_t offset) {
      return reinterpret_cast<void *>(reinterpret_cast<uint8_t*>(ptr) - offset);
    }

private:
    bool IsBuffTypeNomal(const void *ptr) const {
      BuffTypeInfo buff_type_info{};
      uint32_t len = static_cast<uint32_t>(sizeof(BuffTypeInfo));
      (void) rtBuffGetInfo(RT_BUFF_GET_MBUF_BUILD_INFO, &ptr, sizeof(void *), &buff_type_info, &len);
      (void) len;
      return buff_type_info.type == BuffType::kBuffTypeNormal;
    }
    TF_DISALLOW_COPY_AND_ASSIGN(MbufAllocator);
};

class MbufAllocatorFactory : public AllocatorFactory {
public:
    Allocator *CreateAllocator() override {
      if (MbufInit() != Status::OK()) {
        return nullptr;
      }
      return new MbufAllocator;
    }
    SubAllocator *CreateSubAllocator(int numa_node) override {
      if (MbufInit() != Status::OK()) {
        return nullptr;
      }
      return new CPUSubAllocator(new MbufAllocator);
    }

private:
    class CPUSubAllocator : public SubAllocator {
    public:
        explicit CPUSubAllocator(MbufAllocator *cpu_allocator)
            : SubAllocator({}, {}), cpu_allocator_(cpu_allocator) {
        }
#ifdef TF_VERSION_TF2
        void* Alloc(size_t alignment, size_t num_bytes,
                    size_t* bytes_received) override {
          *bytes_received = num_bytes;
          return cpu_allocator_->AllocateRaw(alignment, num_bytes);
        }

        bool SupportsCoalescing() const override { return false; }
#else
        void *Alloc(size_t alignment, size_t num_bytes) override {
          return cpu_allocator_->AllocateRaw(alignment, num_bytes);
        }
#endif
        void Free(void *ptr, size_t num_bytes) override {
          cpu_allocator_->DeallocateRaw(ptr);
        }

    private:
        MbufAllocator *cpu_allocator_;
    };
};

class MbufAllocatorFactoryRegistration {
public:
    MbufAllocatorFactoryRegistration(const char *file, int line, const string &name, int priority) {
      if (IsMbufAllocatorEnabled()) {
        AllocatorFactory *factory = new MbufAllocatorFactory;
        AllocatorFactoryRegistry::singleton()->Register(file, line, name, priority, factory);
      }
    }
};

#define REGISTER_MEM_MBUF_ALLOCATOR(name, priority)                                                                    \
  REGISTER_MEM_MBUF_ALLOCATOR_UNIQ_HELPER(__COUNTER__, __FILE__, __LINE__, name, priority)

#define REGISTER_MEM_MBUF_ALLOCATOR_UNIQ_HELPER(ctr, file, line, name, priority)                                       \
  REGISTER_MEM_MBUF_ALLOCATOR_UNIQ(ctr, file, line, name, priority)

#define REGISTER_MEM_MBUF_ALLOCATOR_UNIQ(ctr, file, line, name, priority)                                              \
  static MbufAllocatorFactoryRegistration allocator_factory_reg_##ctr(file, line, name, priority)

// if env ENABLE_MBUF_ALLOCATOR is set
REGISTER_MEM_MBUF_ALLOCATOR("DefaultMBUFAllocator", 110);
}  // namespace
} // tensorflow
