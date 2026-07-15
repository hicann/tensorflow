# Installing Protobuf of the Python Version

If the training script depends on the Python version of Protobuf to store data in the serialized structure \(for example, the serialization APIs of TensorFlow\), you need to install Protobuf Python.

1. Check that the system contains the dynamic library  **/usr/local/python3.7.5/lib/python3.7/site-packages/google/protobuf/pyext/_message.cpython-37m-_<arch\>_-linux-gnu.so**. If not, perform the following steps to install it.  _<arch\>_  indicates the system architecture type.

    > [!NOTE]NOTE
    > **/usr/local/python3.7.5/lib/python3.7/site-packages**  is the path for installing third-party libraries using pip. You can run the  **pip3 -V**  command to query the path.
    > If the system displays  **/usr/local/python3.7.5/lib/python3.7/site-packages/pip**, the path for installing third-party libraries using pip is  **/usr/local/python3.7.5/lib/python3.7/site-packages**.

2. Run the following command to uninstall Protobuf:

    ```bash
    pip3 uninstall protobuf
    ```

3. Download the Protobuf software package.

    Download  **protobuf-python-3.11.3.tar.gz**  \(or an alternative version that is compatible with TensorFlow installed in the current environment\) from  [https://github.com/protocolbuffers/protobuf/releases/download/v3.11.3/protobuf-python-3.11.3.tar.gz](https://github.com/protocolbuffers/protobuf/releases/download/v3.11.3/protobuf-python-3.11.3.tar.gz), upload the package to any directory on the Linux server as the  **root**  user, and decompress the package.

    ```bash
    tar zxvf protobuf-python-3.11.3.tar.gz
    ```

4. Install Protobuf as the  **root**  user.

    Go to the Protobuf software package directory.

    1. Install the dependencies of Protobuf.

        If the OS is Ubuntu, run the following command:

        ```bash
        apt-get install autoconf automake libtool curl make g++ unzip libffi-dev -y
        ```

        If the OS is CentOS or BC-Linux, run the following command:

        ```bash
        yum install autoconf automake libtool curl make gcc-c++ unzip libffi-devel -y
        ```

    2. Grant the execute permission on the  **autogen.sh**  script and execute the script.

        ```bash
        chmod +x autogen.sh
        ./autogen.sh
        ```

    3. Configure the installation path. The default installation path is  **/usr/local**.

        ```bash
        ./configure
        ```

        To specify the installation path, run the following command:

        ```bash
        ./configure --prefix=/protobuf
        ```

        **/protobuf**  indicates the installation path specified by the user.

    4. Run the following commands to install Protobuf:

        ```bash
        make -j15        # Check the number of CPUs by running grep -w processor /proc/cpuinfo|wc -l. In this example, the number is 15. You can set the parameters as required.
        make install
        ```

    5. Refresh the shared libraries.

        ```bash
        ldconfig
        ```

       After protobuf is installed, the `google/protobuf` folder will be generated in the include directory under the path specified by `--prefix`, where protobuf-related header files are stored. Additionally, the `protoc` executable file will be generated in the bin directory under the path specified by `--prefix`, which is used to compile `*.proto` files and generate the C++ header files and implementation files for protobuf.

    6. Check whether the installation is complete.

        ```bash
        ln -s /protobuf/bin/protoc /usr/bin/protoc
        protoc --version
        ```

        **/protobuf**  is the installation path configured in `--prefix`. If the installation path is not configured, run the  **protoc --version**  command to check whether the installation is successful.

5. Install the runtime library of Protobuf Python.
    1. Go to the Python subdirectory in the Protobuf software package directory and compile the Python runtime library.

        ```bash
        python3 setup.py build --cpp_implementation
        ```

        > [!NOTE]NOTE
        > If you do not use this command to generate the runtime library of the binary version, the processing performance for serialized structures will be diminished.

    2. Install the dynamic library.

        ```bash
        cd .. && make install
        ```

        Go to the Python subdirectory and install the Python runtime library.

        ```bash
        python3 setup.py install --cpp_implementation
        ```

    3. Check whether the installation is successful.

        Check whether the system contains the dynamic library  **/usr/local/python3.7.5/lib/python3.7/site-packages/protobuf-3.11.3-py3.7-linux-aarch64.egg/google/protobuf/pyext/_message.cpython-37m-_<arch\>_-linux-gnu.so**.  _<arch\>_  indicates the system architecture type.

        > [!NOTE]NOTE
        >**/usr/local/python3.7.5/lib/python3.7/site-packages**  is the path for installing third-party libraries using pip. You can run the  **pip3 -V**  command to query the path.
        >If the system displays  **/usr/local/python3.7.5/lib/python3.7/site-packages/pip**, the path for installing third-party libraries using pip is  **/usr/local/python3.7.5/lib/python3.7/site-packages**.

    4. If you have specified the installation path in `--prefix`, add the following environment variable settings to the run script:

        ```bash
        export LD_LIBRARY_PATH=/protobuf/lib:${LD_LIBRARY_PATH}
        ```

        **/protobuf**  is the installation path configured in  `--prefix`.

    5. Create a soft link.

        If you have specified the installation path, you need to establish a soft link. Otherwise, an error will be reported when TensorFlow is imported. The command is as follows:

        ```bash
        ln -s /protobuf/lib/libprotobuf.so.22.0.3 /usr/lib/libprotobuf.so.22
        ```

        **/protobuf**  is the installation path configured in `--prefix`.
