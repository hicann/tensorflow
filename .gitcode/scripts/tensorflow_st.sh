#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e
set -x

export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
export ASCEND_INSTALL_PATH=/home/jenkins/Ascend/cann
export ASCEND_HOME_PATH=/home/jenkins/Ascend/cann
export ASCEND_CUSTOM_PATH=/home/jenkins/Ascend/cann

function LOG_DO() {
    local date_time
    local BPurple='\e[1;35m'
    local Purple='\e[0;35m'
    local Color_Off='\e[0m'
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BPurple}[Command]${Color_Off} ${date_time} ${Purple}$*${Color_Off}"
    "$@"
}
if sudo update-alternatives --set gcc /usr/bin/gcc-16 2>/dev/null; then
    echo "Switched to gcc-16"
elif sudo update-alternatives --set gcc /usr/bin/gcc-15 2>/dev/null; then
    echo "Switched to gcc-15"
elif sudo update-alternatives --set gcc /usr/bin/gcc-14 2>/dev/null; then
    echo "gcc-16/15 not available, fell back to gcc-14"
fi
if gcc --version | head -n1 | grep -q "15\."; then
    rm -rf /home/jenkins/opensource/lib_cache
    if [ -d /home/jenkins/opensource/gcc15 ]; then
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15/lib_cache/ /home/jenkins/opensource/lib_cache
    elif [ -d /home/jenkins/opensource/gcc15x86 ]; then
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15x86/lib_cache/ /home/jenkins/opensource/lib_cache
    fi
elif gcc --version | head -n1 | grep -q "14\."; then
    gcc --version
else
    gcc --version
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi

gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash || { echo "setenv failed"; exit 1; }

set +e
echo "Start run c++ st testcase"
cd "${WORKSPACE}" || exit 1

sudo localedef -i en_US -f UTF-8 en_US.UTF-8
LOG_DO sh scripts/build.sh -s
ret=$?
set -e

if [ "$ret" -ne 200 ] && [ "$ret" -ne 0 ]; then
    echo "run st fail, exit code: $ret"
    exit 1
fi
exit 0
