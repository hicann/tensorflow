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
WORK_PATH="$CUR_PATH"
BUILD_PATH="${WORK_PATH}/build"

# print usage message
usage() {
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-v] [-g] [-c] [-u] [-s]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -j[n] Set the number of threads used to build CANN, default is 8"
  echo "    -v Verbose"
  echo "    -g GCC compiler prefix, used to specify the compiler toolchain"
  echo "    -c TF_adapter compile"
  echo "    -u TF_adapter utest"
  echo "    -s TF_adapter stest"
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
  ENABLE_COMPILE="off"
  ENABLE_TFADAPTER_UT="off"
  ENABLE_TFADAPTER_ST="off"

  # Process the options
  while getopts 'hj:vuscg:xa' opt
  do
    case "${opt}" in
      h) usage
         exit 0 ;;
      j) THREAD_NUM=$OPTARG ;;
      v) VERBOSE="VERBOSE=1" ;;
      g) GCC_PREFIX=$OPTARG ;;
      c) ENABLE_COMPILE="on" ;;
      u) ENABLE_TFADAPTER_UT="on" ;;
      s) ENABLE_TFADAPTER_ST="on" ;;
      *) logging "Undefined option: ${opt}"
         usage
         exit 1 ;;
    esac
  done

  if [[ "X$ENABLE_COMPILE" = "Xoff" ]] && \
     [[ "X$ENABLE_TFADAPTER_UT" = "Xoff" ]] && \
     [[ "X$ENABLE_TFADAPTER_ST" = "Xoff" ]]; then
    ENABLE_COMPILE="on"
  fi
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

  PYTHON_BIN_PATH="$(which python3)"
  if ! "$PYTHON_BIN_PATH" "configure.py"; then
    echo "ERROR: Failed to run configure.py!"
    exit 1
  fi
  echo "Configuration finished"

  CMAKE_ARGS="-DENABLE_OPEN_SRC=True -DCMAKE_INSTALL_PREFIX=${RELEASE_PATH}"
  if [[ "X$ENABLE_COMPILE" = "Xon" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DASCEND_CI_LIMITED_PY37_ENABLE=ON"
  fi
  if [[ "$GCC_PREFIX" != "" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DGCC_PREFIX=$GCC_PREFIX"
  fi
  CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_TFADAPTER_UT=$ENABLE_TFADAPTER_UT -DENABLE_TFADAPTER_ST=$ENABLE_TFADAPTER_ST"
  logging "CMake Args: ${CMAKE_ARGS}"

  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  if [ 0 -ne $? ]; then
    echo "execute command: cmake ${CMAKE_ARGS} .. failed."
    return 1
  fi
  if [[ "X$ENABLE_COMPILE" = "Xon" ]]; then
    make ${VERBOSE} -j${THREAD_NUM}
    logging "Build tfadapter finished!"
  elif [[ "X$ENABLE_TFADAPTER_UT" = "Xon" ]]; then
    export LD_LIBRARY_PATH=${BUILD_PATH}:$LD_LIBRARY_PATH
    make adapter2_ut ${VERBOSE} -j${THREAD_NUM}
    logging "Build tfadapter utest finished!"
  elif [[ "X$ENABLE_TFADAPTER_ST" = "Xon" ]]; then
    export LD_LIBRARY_PATH=${BUILD_PATH}:$LD_LIBRARY_PATH
    make adapter2_st ${VERBOSE} -j${THREAD_NUM}
    logging "Build tfadapter stest finished!"
  fi
}

main() {
  checkopts "$@"

  # tfadapter build start
  logging "---------------- tfadapter build start ----------------"
  ${GCC_PREFIX}g++ -v
  mk_dir "${RELEASE_PATH}"

  if [[ "X$ENABLE_COMPILE" = "Xoff" ]]; then
    WORK_PATH="$CUR_PATH/tests"
    BUILD_PATH="${WORK_PATH}/build"
  fi
  cd ${WORK_PATH}
  build_tfadapter

  export PRINT_MODEL=1
  rm -rf ${BASE_PATH}/coverage
  mkdir ${BASE_PATH}/coverage
  if [[ "X$ENABLE_TFADAPTER_UT" = "Xon" ]]; then
    lcov -o ${BASE_PATH}/coverage/coverage.info -a ${BUILD_PATH}/ut/ut.coverage
  fi
  if [[ "X$ENABLE_TFADAPTER_ST" = "Xon" ]]; then
    lcov -o ${BASE_PATH}/coverage/coverage.info -a ${BUILD_PATH}/st/st.coverage
  fi
  logging "---------------- tfadapter build finished ----------------"
}

main "$@"
