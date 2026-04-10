# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (HAVE_C_SEC)
    return()
endif()

include(ExternalProject)

if ((${CMAKE_INSTALL_PREFIX} STREQUAL /usr/local) OR
    (${CMAKE_INSTALL_PREFIX} STREQUAL "C:/Program Files (x86)/ascend"))
    set(CMAKE_INSTALL_PREFIX ${TFADAPTER_DIR}/output CACHE STRING "path for install()" FORCE)
    message(STATUS "No install prefix selected, default to ${CMAKE_INSTALL_PREFIX}.")
endif()

set(REQ_URL "https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.16.tar.gz")

ExternalProject_Add(c_sec_build
                    URL ${REQ_URL}
                    PATCH_COMMAND patch -p1 < ${TFADAPTER_DIR}/tf_adapter/tests/patch/securec/0001-add-securec-cmake-script.patch
                    CONFIGURE_COMMAND ${CMAKE_COMMAND}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DCMAKE_LINKER=${CMAKE_LINKER}
                        -DCMAKE_AR=${CMAKE_AR}
                        -DCMAKE_RANLIB=${CMAKE_RANLIB}
                        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/c_sec
                        <SOURCE_DIR>
                    BUILD_COMMAND $(MAKE)
                    INSTALL_COMMAND $(MAKE) install
                    EXCLUDE_FROM_ALL TRUE
)

set(C_SEC_PKG_DIR ${CMAKE_INSTALL_PREFIX}/c_sec)

add_library(c_sec SHARED IMPORTED)

file(MAKE_DIRECTORY ${C_SEC_PKG_DIR}/include)

set_target_properties(c_sec PROPERTIES
    IMPORTED_LOCATION ${C_SEC_PKG_DIR}/lib/libc_sec.so
)

target_include_directories(c_sec INTERFACE ${C_SEC_PKG_DIR}/include)

add_dependencies(c_sec c_sec_build)

set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(FILES ${C_SEC_PKG_DIR}/lib/libc_sec.so OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR})

add_library(c_sec_static_lib STATIC IMPORTED)
set_target_properties(c_sec_static_lib PROPERTIES
    IMPORTED_LOCATION ${C_SEC_PKG_DIR}/lib/libc_sec.a
)

add_library(c_sec_static INTERFACE)
target_include_directories(c_sec_static INTERFACE ${C_SEC_PKG_DIR}/include)
target_link_libraries(c_sec_static INTERFACE c_sec_static_lib)

add_dependencies(c_sec_static c_sec_build)

set(HAVE_C_SEC TRUE)
