#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e
CUR_PATH=$(cd "$(dirname $0)"; pwd)
BASE_PATH=$(cd "$(dirname $0)/.."; pwd)
RELEASE_PATH="${BASE_PATH}/output"
BUILD_PATH="${BASE_PATH}/build"
CMAKE_PATH="${BUILD_PATH}/tfadapter"
RELEASE_TARGET="tfadapter.tar"

# print usage message
usage() {
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-v] [-g] [-u] [-s] [-c] [-x] [-a]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -j[n] Set the number of threads used to build CANN, default is 8"
  echo "    -v Verbose"
  echo "    -g GCC compiler prefix, used to specify the compiler toolchain"
  echo "    -u TF_adapter utest"
  echo "    -s TF_adapter stest"
  echo "    -c TF_adapter ci build"
  echo "    -x TF_adapter2.x ci build"
  echo "    -a TF_adapter2.x use python3.7"
  echo "to be continued ..."
}

logging() {
  echo "[INFO] $@"
}

# parse and set optionss
checkopts() {
  VERBOSE=""
  THREAD_NUM=8
  GCC_PREFIX=""
  ENABLE_TFADAPTER_UT="off"
  ENABLE_TFADAPTER_ST="off"
  ENABLE_CI_BUILD="off"
  ENABLE_2X_CI_BUILD="off"
  TFADAPTER_2X_PY37="off"
  # Process the options
  while getopts 'hj:vuscg:xa' opt
  do
    case "${opt}" in
      h) usage
         exit 0 ;;
      j) THREAD_NUM=$OPTARG ;;
      v) VERBOSE="VERBOSE=1" ;;
      g) GCC_PREFIX=$OPTARG ;;
      u) ENABLE_TFADAPTER_UT="on" ;;
      s) ENABLE_TFADAPTER_ST="on" ;;
      c) ENABLE_CI_BUILD="on" ;;
      x) ENABLE_2X_CI_BUILD="on" ;;
      a) TFADAPTER_2X_PY37="on" ;;
      *) logging "Undefined option: ${opt}"
         usage
         exit 1 ;;
    esac
  done
}

# mkdir directory
mk_dir() {
  local dir_name="$1"
  mkdir -pv "${dir_name}"
  logging "Created dir ${dir_name}"
}

# create build path
build_tfadapter() {
  logging "Create build directory and build tfadapter"
  cd "${BASE_PATH}" && \
  if ! ./configure; then
    exit 1
  fi
  CMAKE_ARGS="-DENABLE_OPEN_SRC=True -DBUILD_PATH=$BUILD_PATH -DCMAKE_INSTALL_PREFIX=${RELEASE_PATH}"
  if [[ "$GCC_PREFIX" != "" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DGCC_PREFIX=$GCC_PREFIX"
  fi
  CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_TFADAPTER_UT=$ENABLE_TFADAPTER_UT -DENABLE_TFADAPTER_ST=$ENABLE_TFADAPTER_ST"
  logging "CMake Args: ${CMAKE_ARGS}"

  mk_dir "${CMAKE_PATH}"
  cd "${CMAKE_PATH}" && cmake ${CMAKE_ARGS} ../..
  if [ 0 -ne $? ]
  then
    echo "execute command: cmake ${CMAKE_ARGS} .. failed."
    return 1
  fi
  if [[ "X$ENABLE_TFADAPTER_UT" = "Xon" ]]; then
    make tfadapter_utest ${VERBOSE} -j${THREAD_NUM}
    logging "Build tfadapter utest success!"
  elif [[ "X$ENABLE_TFADAPTER_ST" = "Xon" ]]; then
    make tfadapter_stest ${VERBOSE} -j${THREAD_NUM}
    logging "Build tfadapter stest success!"
  else
    make ${VERBOSE} -j${THREAD_NUM}
    logging "tfadapter build success!"
    chmod +x "${CUR_PATH}/tf_adapter_2.x/CI_Build"
    sh "${CUR_PATH}/tf_adapter_2.x/CI_Build"
  fi
}

release_tfadapter() {
  logging "Create output directory"
  cd ${CMAKE_PATH}/dist/python/dist && mkdir -p fwkplugin/bin && mv npu_bridge-*.whl fwkplugin/bin && mv ${BASE_PATH}/tf_adapter_2.x/build/dist/python/dist/npu_device-*.whl fwkplugin/bin && tar cfz "${RELEASE_TARGET}" * && mv "${RELEASE_TARGET}" "${RELEASE_PATH}"
}

main() {
  checkopts "$@"
  # tfadapter build start
  logging "---------------- tfadapter build start ----------------"
  ${GCC_PREFIX}g++ -v
  mk_dir "${RELEASE_PATH}"
  if [[ "X$ENABLE_2X_CI_BUILD" = "Xon" ]]; then
    chmod +x "${CUR_PATH}/tf_adapter_2.x/CI_Build"
    if [[ "X$TFADAPTER_2X_PY37" = "Xon" ]]; then
      sh "${CUR_PATH}/tf_adapter_2.x/CI_Build" "python3.7"
    else
      sh "${CUR_PATH}/tf_adapter_2.x/CI_Build" "python3.9"
    fi
  else
    build_tfadapter
  fi

  if [[ "X$ENABLE_TFADAPTER_UT" = "Xoff" ]] && [[ "X$ENABLE_TFADAPTER_ST" = "Xoff" ]] && [[ "X$ENABLE_CI_BUILD" = "Xon" ]]; then
    release_tfadapter
  fi
  if [[ "X$ENABLE_TFADAPTER_UT" = "Xon" ]]; then
    cd ${BASE_PATH}
    export ASCEND_OPP_PATH=${BASE_PATH}/tf_adapter/tests/depends/support_json
    export PRINT_MODEL=1
    export LD_LIBRARY_PATH=${CMAKE_PATH}/tf_adapter/tests/depends/aoe/:$LD_LIBRARY_PATH
    RUN_TEST_CASE=${CMAKE_PATH}/tf_adapter/tests/ut/tfadapter_utest && unset ENABLE_MBUF_ALLOCATOR \
    && ${RUN_TEST_CASE} && export ENABLE_MBUF_ALLOCATOR=1 && ${RUN_TEST_CASE} "--gtest_filter=MbufAllocatorTest.EnableMbufAllocatorTest"
    if [[ "$?" -ne 0 ]]; then
      echo "!!! UT FAILED, PLEASE CHECK YOUR CHANGES !!!"
      echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
      exit 1;
    fi
    logging "Generating coverage statistics, please wait..."
    rm -rf ${BASE_PATH}/coverage
    mkdir ${BASE_PATH}/coverage
    lcov -c -d ${CMAKE_PATH}/tf_adapter/tests/ut/ -o coverage/tmp.info
    lcov -r coverage/tmp.info '*/tests/*' '*/nlohmann_json-src/*' '*/tensorflow-src/*' \
      '*/inc/*' '*/output/*' '*/usr/*' '*/Eigen/*' '*/absl/*' '*/google/*' '*/tensorflow/core/*' \
      -o adapter1_coverage.info
    export LD_LIBRARY_PATH=${BASE_PATH}/tf_adapter_2.x/tests/build/:$LD_LIBRARY_PATH
    bash ${CUR_PATH}/tf_adapter_2.x/tests/CI_Build adapter2_ut
    lcov -o coverage/coverage.info -a ${BASE_PATH}/tf_adapter_2.x/tests/build/ut/ut.coverage -a adapter1_coverage.info
  fi
  if [[ "X$ENABLE_TFADAPTER_ST" = "Xon" ]]; then
    cd ${BASE_PATH}
    export ASCEND_OPP_PATH=${BASE_PATH}/tf_adapter/tests/depends/support_json
    export PRINT_MODEL=1
    export LD_LIBRARY_PATH=${CMAKE_PATH}/tf_adapter/tests/depends/aoe/:$LD_LIBRARY_PATH
    RUN_TEST_CASE=${CMAKE_PATH}/tf_adapter/tests/st/tfadapter_stest && unset ENABLE_MBUF_ALLOCATOR \
    && ${RUN_TEST_CASE} && export ENABLE_MBUF_ALLOCATOR=1 && ${RUN_TEST_CASE} "--gtest_filter=MbufAllocatorTest.EnableMbufAllocatorTest"
    if [[ "$?" -ne 0 ]]; then
      echo "!!! ST FAILED, PLEASE CHECK YOUR CHANGES !!!"
      echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
      exit 1;
    fi
    logging "Generating coverage statistics, please wait..."
    rm -rf ${BASE_PATH}/coverage
    mkdir ${BASE_PATH}/coverage
    lcov -c -d ${CMAKE_PATH}/tf_adapter/tests/st/ -o coverage/tmp.info
    lcov -r coverage/tmp.info '*/tests/*' '*/nlohmann_json-src/*' '*/tensorflow-src/*' \
      '*/inc/*' '*/output/*' '*/usr/*' '*/Eigen/*' '*/absl/*' '*/google/*' '*/tensorflow/core/*' \
      -o adapter1_coverage.info
    export LD_LIBRARY_PATH=${BASE_PATH}/tf_adapter_2.x/tests/build/:$LD_LIBRARY_PATH
    bash ${CUR_PATH}/tf_adapter_2.x/tests/CI_Build adapter2_st
    lcov -o coverage/coverage.info -a ${BASE_PATH}/tf_adapter_2.x/tests/build/st/st.coverage -a adapter1_coverage.info
  fi
  logging "---------------- tfadapter build finished ----------------"
}

main "$@"
