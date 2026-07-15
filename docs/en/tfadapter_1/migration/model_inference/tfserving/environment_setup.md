# Environment Setup

- Download and install the CANN software package and framework package by referring to  [Environment Setup](../../../installation/README.md).
- Install the dependencies listed in the following table.

  | Dependency | Version |
  | --- | --- |
  | gcc | 8.4 or later<br>You can run the cc  --version command to check the GCC version in use.<br>If GCC 9.x is used, Bazel compilation will fail. Therefore, GCC 9.x is not recommended. If you have to use it, click [here](https://github.com/grpc/grpc/pull/19647) to rectify related faults. |
  | g++ | 8.4 or later<br>You can run the c++ --version command to check the GCC version in use.<br>If GCC 9.x is used, Bazel compilation will fail. Therefore, GCC 9.x is not recommended. If you have to use it, click [here](https://github.com/grpc/grpc/pull/19647) to rectify related faults. |
  | zip | Any version |
  | unzip | Any version |
  | libtool | Any version |
  | automake | Any version |
  | Python | 3.7.5 |
  | TensorFlow | 1.15.0 |
  | tensorflow-serving-api | 1.15.0 |
  | future | Any version |
  | bazel | 0.24.1 or later |
  | CMake | 3.14.0 or later |
  | swig | If the operating system architecture is aarch64, the software version must be 3.0.12 or later.<br>If the operating system architecture is x86_64, the software version must be 4.0.1 or later. |

    > [!NOTE]NOTE
    >
    > - The GCC and G++ versions must match. Otherwise, errors may occur during the compilation of TF Serving.
    > - For details about how to use bazel for building, see  [Installing bazel 0.24.1 from Source Code](common_operation.md#installing-bazel-0241-from-source-code).
    > - For details about how to use CMake for building, see  [Installing CMake 3.14.0 from Source Code](common_operation.md#installing-cmake-3140-from-source-code).
    > - If the swig software package fails to be installed, rectify the fault by referring to  [What Should I Do If the swig Dependency Package Cannot Be Installed?](FAQ.md#what-should-i-do-if-the-swig-dependency-package-cannot-be-installed).
    > - The gcc, g++, zip, unzip, libtool, and automake software packages must be installed using apt or yum. The TensorFlow, tensorflow-serving-api, and future software packages must be installed using pip3.
