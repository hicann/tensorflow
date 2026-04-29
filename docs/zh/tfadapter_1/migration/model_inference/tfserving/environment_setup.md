# 环境准备

- 已安装CANN软件包与框架包，详细请参见[环境准备](../../../installation/README.md)。
- 安装如下表所示相关依赖。

    | 依赖包 | 版本限制 |
    |--------|----------|
    | gcc | 8.4及以上版本<br>用户可使用 `cc --version` 命令确定当前系统使用的gcc版本。<br>由于使用9.x版本的gcc会导致bazel编译失败，因此不建议使用9.x版本的gcc。若必须使用，请自行参见[链接](https://github.com/grpc/grpc/pull/19647)修复相关问题。 |
    | g++ | 8.4及以上版本<br>用户可使用 `c++ --version` 命令确定当前系统使用的gcc版本。<br>由于使用9.x版本的gcc会导致bazel编译失败，因此不建议使用9.x版本的gcc。若必须使用，请自行参见[链接](https://github.com/grpc/grpc/pull/19647)修复相关问题。 |
    | zip | 无特定版本要求。 |
    | unzip | 无特定版本要求。 |
    | libtool | 无特定版本要求。 |
    | automake | 无特定版本要求。 |
    | Python | 3.7.5 |
    | TensorFlow | 1.15.0 |
    | tensorflow-serving-api | 1.15.0 |
    | future | 无特定版本要求。 |
    | bazel | 0.24.1及以上版本 |
    | CMake | 3.14.0及以上版本 |
    | swig | 若操作系统架构为 `aarch64`，软件安装版本需大于或等于3.0.12。<br>若操作系统架构为 `x86_64`，软件安装版本需大于或等于4.0.1 。 |

    > [!NOTE]说明
    >
    > - gcc和g++的版本需要保持一致，否则TF Serving源码编译时可能会报错。
    > - bazel编译安装可参考[源码安装0.24.1版本bazel](common_operation.md#源码安装0241版本bazel)。
    > - CMake编译安装可参考[源码安装3.14.0版本CMake](common_operation.md#源码安装3140版本cmake)。
    > - 如果在安装swig软件包时出现无法安装软件包问题，参考[无法安装swig软件包](FAQ.md#无法安装swig软件包)解决。
    > - gcc、g++、zip、unzip、libtool和automake软件包使用apt或yum进行安装，TensorFlow、tensorflow-serving-api和future软件包使用pip3方式安装。
