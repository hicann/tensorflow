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
tensorflow_path26="/home/jenkins/tensorflow26"

# The TF wheel deps (grpcio 1.62.3 sdist) are compiled with the system gcc,
# and its bundled boringssl fails on gcc-14+ (the __builtin_addc/_Generic
# path is invalid C++). Switch to gcc-13 for the install and restore after.
saved_gcc=$(update-alternatives --query gcc 2>/dev/null | grep "^Value:" | awk '{print $2}')
if [ -x /usr/bin/gcc-13 ];then
  sudo update-alternatives --set gcc /usr/bin/gcc-13
fi

cd /home/jenkins/
sudo chown -R jenkins:jenkins tensorflow26
python3.7 -m venv  ${tensorflow_path26}/.env
cp /home/jenkins/.pip/pip.conf /home/jenkins/tensorflow26/.env/
source ${tensorflow_path26}/.env/bin/activate
pip3.7 --version
pip3.7 install -U numpy==1.19.3
pip3.7 install -U protobuf==3.13.0
pip3.7 install -U swig
wget -nv "https://opencann-obs.obs.cn-north-4.myhuaweicloud.com/ci/tensorflow-2.6.5-cp37-cp37m-linux_x86_64.whl"
pip3.7 install tensorflow-2.6.5-cp37-cp37m-linux_x86_64.whl
deactivate

if [ -n "${saved_gcc}" ] && [ -x "${saved_gcc}" ];then
  sudo update-alternatives --set gcc ${saved_gcc}
fi
