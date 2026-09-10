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
function LOG_DO() {
    local date_time
    local BPurple='\e[1;35m'
    local Purple='\e[0;35m'
    local Color_Off='\e[0m'
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BPurple}[Command]${Color_Off} ${date_time} ${Purple}$*${Color_Off}"
    "$@"
}
if [[ "${task_name}" == *ubuntu24* ]]; then
    if sudo update-alternatives --set gcc /usr/bin/gcc-16 2>/dev/null; then
        echo "Switched to gcc-16"
    elif sudo update-alternatives --set gcc /usr/bin/gcc-15 2>/dev/null; then
        echo "Switched to gcc-15"
    elif sudo update-alternatives --set gcc /usr/bin/gcc-14 2>/dev/null; then
        echo "gcc-16/15 not available, fell back to gcc-14"
    fi
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
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
else
    gcc --version
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi
gcc --version
THREAD_NUMBER=${THREAD_NUMBER:-$(nproc)}

source /home/jenkins/Ascend/cann/bin/setenv.bash

echo "Build ${REPOSITORY_NAME}."
cd ${WORKSPACE}/ || exit
old_branches=("r1.5.0")
if [[ "${old_branches[@]}" == "${GIT_TARGET_BRANCH}" ]];then
  echo "The TF branch is ${GIT_TARGET_BRANCH}"
  LOG_DO sh scripts/build.sh -j${THREAD_NUMBER}
elif [[ "${task_name}" =~ ^Compile_Ascend_X86_TF(_ubuntu24)?$ ]]; then
  sh ${WORKSPACE}/install_tf_abi.sh
  LOG_DO sh scripts/build.sh -x -a
elif [ "${OS_TYPE}" = "ubuntu_aarch64" ];then
  echo "/home/jenkins/.local/bin/swig" | sh scripts/build.sh -c -j20
else
  echo "The TF branch is ${GIT_TARGET_BRANCH}"
  LOG_DO sh scripts/build.sh -c -j${THREAD_NUMBER}
fi
ret=$?
set -e

if [ $ret -ne 0 ]; then
    echo "Build failed with exit code $ret"
    exit $ret
fi
if [[ "${task_name}" == *ubuntu24* ]];then
    if [[ ! "${task_name}" =~ ^Compile_Ascend_X86_TF(_ubuntu24)?$ ]]; then
        compile_package_name=$(ls "${WORKSPACE}/output" | grep -E '\.tar$' | head -n1)
        if [[ -z "${compile_package_name}" ]]; then
            echo "No .tar package found in output!"
            exit 1
        fi
        target_name="${compile_package_name%.tar}_ubuntu24.tar"
        echo "Renaming package: ${compile_package_name} -> ${target_name}"
        mv "${WORKSPACE}/output/${compile_package_name}" "${WORKSPACE}/output/${target_name}"
    elif [[ "${task_name}" =~ ^Compile_Ascend_X86_TF(_ubuntu24)?$ ]]; then
        compile_package_name=$(ls "${WORKSPACE}/tf_adapter_2.x/build/dist/python/dist" | grep -E '\.whl$' | head -n1)
        if [[ -z "${compile_package_name}" ]]; then
            echo "No .whl package found in tf_adapter_2.x/build/dist/python/dist!"
            exit 1
        fi
        target_name="${compile_package_name%.whl}_ubuntu24.whl"
        echo "Renaming package: ${compile_package_name} -> ${target_name}"
        mv "${WORKSPACE}/tf_adapter_2.x/build/dist/python/dist/${compile_package_name}" "${WORKSPACE}/tf_adapter_2.x/build/dist/python/dist/${target_name}"
    fi

fi
exit 0