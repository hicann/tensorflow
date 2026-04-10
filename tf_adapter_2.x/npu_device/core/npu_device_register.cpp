/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_device_register.h"

#include "tensorflow/core/platform/logging.h"

#include "npu_device.h"
#include "npu_global.h"

namespace {
TFE_TensorHandle *CopyTensorToNpuDevice(TFE_Context *context, TFE_TensorHandle *tensor, TF_Status *status,
                                        void *device_info) {
  auto *dev = static_cast<npu::NpuDevice *>(device_info);
  tensorflow::Status tf_status;
  if (npu::IsNpuTensorHandle(tensor)) {
    DLOG() << "[CopyTensorToNpuDevice] Ref tensor from " << tensorflow::unwrap(tensor)->DeviceName(&tf_status) << " to "
           << dev->device_name;
    tensorflow::unwrap(tensor)->Ref();
    return tensor;
  }
  LOG(INFO) << "[CopyTensorToNpuDevice] Copy tensor from " << tensorflow::unwrap(tensor)->DeviceName(&tf_status)
            << " to " << dev->device_name;
  TFE_TensorHandle *npu_handle = dev->CopyTensorH2D(context, tensor, status);
  if (TF_GetCode(status) != TF_OK) {
    return nullptr;
  }
  return npu_handle;
}

TFE_TensorHandle *CopyTensorFromNpuDevice(TFE_Context *context, TFE_TensorHandle *tensor,
                                          const char *target_device_name, TF_Status *status, void *device_info) {
  auto *dev = static_cast<npu::NpuDevice *>(device_info);
  DLOG() << "[CopyTensorFromNpuDevice] Copy tensor from " << dev->device_name << " to " << target_device_name;
  // 输入的TensorHandle是NPU的，应当先进行NPU->CPU的传输，再调用TFE_TensorHandleCopyToDevice防止可能的NPU->GPU传输
  // 一旦Copy动作发生，需要进行stream同步。如果是NPU->NPU的拷贝（理论上不应该发生），可以不同步。
  TFE_TensorHandle *local_tensor = dev->CopyTensorD2H(context, tensor, *status);
  if (TF_GetCode(status) != TF_OK) {
    return nullptr;
  }
  TFE_TensorHandle *target_tensor = TFE_TensorHandleCopyToDevice(local_tensor, context, target_device_name, status);
  if (TF_GetCode(status) != TF_OK) {
    return nullptr;
  }

  TFE_DeleteTensorHandle(local_tensor);
  return target_tensor;
}

void NpuDeviceExecute(const TFE_Op *op, int *num_outputs, TFE_TensorHandle **outputs, TF_Status *s, void *device_info) {
  auto *dev = static_cast<npu::NpuDevice *>(device_info);
  dev->Execute(op, *num_outputs, outputs, s);
}

void DeleteNpuDevice(void *device_info) { npu::NpuDevice::DeleteDevice(device_info); }

void RegisterNpuDevice(TFE_Context *context, const char *name, void *device_info, TF_Status *status) {
  TFE_CustomDevice custom_device;
  custom_device.copy_tensor_to_device = &CopyTensorToNpuDevice;
  custom_device.copy_tensor_from_device = &CopyTensorFromNpuDevice;
  custom_device.delete_device = &DeleteNpuDevice;
  custom_device.execute = &NpuDeviceExecute;
  TFE_RegisterCustomDevice(context, custom_device, name, device_info, status);
}

std::vector<npu::NpuDevice *> devices_instances;
}  // namespace

namespace npu {
/**
 * @breif: create device
 * @param context: context
 * @param name: device name
 * @param device_index: device index
 * @param device_options: device options
 */
std::string CreateDevice(TFE_Context *context, const char *name, int device_index,
                         const std::map<std::string, std::string> &global_options,
                         const std::map<std::string, std::string> &session_options) {
  const static std::string kSucceed;

  NpuDevice *device = nullptr;
  auto create_status = NpuDevice::CreateDevice(name, device_index, global_options, session_options, &device);
  if (create_status != kSucceed) {
    return create_status;
  }

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  RegisterNpuDevice(context, name, device, status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    NpuDevice::DeleteDevice(device);
    return std::string("Register Npu device ") + name + " failed:" + TF_Message(status.get());
  }
  LOG(INFO) << "Npu device instance " << name << " created";
  devices_instances.push_back(device);
  global::NpuCtx::SetDeviceCtx(device_index, context, device);

  return kSucceed;
}

/**
 * @breif: release device resource
 */
void ReleaseDeviceResource() {
  for (auto &device : devices_instances) {
    device->ReleaseResource();
  }
  devices_instances.clear();
}
}  // namespace npu
