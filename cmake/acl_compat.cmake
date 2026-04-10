# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

include(ExternalProject)
set(REQ_URL "https://mirrors.huaweicloud.com/artifactory/cann-run/8.5.0/inner/${CMAKE_SYSTEM_PROCESSOR}/acl-compat_8.5.0_linux-${CMAKE_SYSTEM_PROCESSOR}.tar.gz")

ExternalProject_Add(download-acl_compat
    URL ${REQ_URL}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(download-acl_compat SOURCE_DIR)
set(acl_compat_dir "${SOURCE_DIR}")
message(STATUS "acl_compat_dir=${acl_compat_dir}")
