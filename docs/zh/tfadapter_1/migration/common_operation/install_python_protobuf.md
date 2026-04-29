# 安装Python版本的proto

如果训练脚本依赖protobuf的Python版本进行序列化结构的数据存储（例如TensorFlow的序列化相关接口），则需要安装Python版本的proto。

1. 检查系统中是否存在“/usr/local/python3.7.5/lib/python3.7/site-packages/google/protobuf/pyext/_message.cpython-37m-_<arch\>_-linux-gnu.so”动态库，如果没有，需要按照如下步骤安装。其中_<arch\>_为系统架构类型。

    > [!NOTE]说明
    > “/usr/local/python3.7.5/lib/python3.7/site-packages”是pip安装第三方库的路径，可以使用**pip3 -V**检查。
    >
    > 如果系统显示：/usr/local/python3.7.5/lib/python3.7/site-packages/pip，则pip安装第三方库的路径为/usr/local/python3.7.5/lib/python3.7/site-packages。

2. 执行如下命令卸载protobuf。

    ```bash
    pip3 uninstall protobuf
    ```

3. 下载protobuf软件包。

    从[https://github.com/protocolbuffers/protobuf/releases/download/v3.11.3/protobuf-python-3.11.3.tar.gz](https://github.com/protocolbuffers/protobuf/releases/download/v3.11.3/protobuf-python-3.11.3.tar.gz)路径中下载3.11.3版本protobuf-python-3.11.3.tar.gz软件包（或者其他版本，保证和当前环境上安装的TensorFlow兼容），并以root用户上传到所在Linux服务器任意目录并执行命令解压。

    ```bash
    tar zxvf protobuf-python-3.11.3.tar.gz
    ```

4. 以root用户安装protobuf。

    进入protobuf软件包目录。

    1. 安装protobuf的依赖。

        当操作系统为Ubuntu时，安装命令如下：

        ```bash
        apt-get install autoconf automake libtool curl make g++ unzip libffi-dev -y
        ```

        当操作系统为CentOS/BClinux时，安装命令如下：

        ```bash
        yum install autoconf automake libtool curl make gcc-c++ unzip libffi-devel -y
        ```

    2. 为autogen.sh脚本添加可执行权限并执行此脚本。

        ```bash
        chmod +x autogen.sh
        ./autogen.sh
        ```

    3. 配置安装路径（默认安装路径为"**/usr/local**"）。

         ```bash
         ./configure
         ```

         如果想指定安装路径，可参考以下命令。

         ```bash
         ./configure --prefix=/protobuf
         ```

         “/protobuf”为用户指定的安装路径。

    4. 执行protobuf安装命令。

        ```bash
        make -j15        # 通过grep -w processor /proc/cpuinfo|wc -l查看cpu数，示例为15，用户可自行设置相应参数。
        make install
        ```

    5. 刷新共享库。

        ```bash
        ldconfig
        ```

        protobuf安装完成后，会在`--prefix`配置的路径下面的include目录中生成google/protobuf文件夹，存放protobuf相关头文件；在`--prefix`配置路径下面的bin目录中生成protoc可执行文件，用于进行\*.proto文件的编译，生成protobuf的C++头文件及实现文件。

    6. 检查是否安装完成。

        ```bash
        ln -s /protobuf/bin/protoc /usr/bin/protoc
        protoc --version
        ```

        其中`/protobuf`为`--prefix`中用户配置的安装路径。如果用户未配置安装路径，则直接执行`protoc --version`检查是否安装成功。

5. 安装protobuf的Python版本运行库。
    1. 进入protobuf软件包目录的Python子目录，编译Python版本的运行库。

        ```bash
        python3 setup.py build --cpp_implementation
        ```

        > [!NOTE]说明
        > 如果不执行此命令生成二进制版本的运行库，序列化结构的处理性能会较低。

    2. 安装动态库。

        ```bash
        cd .. && make install
        ```

        进入Python子目录，安装Python版本的运行库。

        ```bash
        python3 setup.py install --cpp_implementation
        ```

    3. 检查是否安装成功。

        检查系统中是否存在“/usr/local/python3.7.5/lib/python3.7/site-packages/protobuf-3.11.3-py3.7-linux-aarch64.egg/google/protobuf/pyext/_message.cpython-37m-_<arch\>_-linux-gnu.so”这个动态库。其中_<arch\>_为系统架构类型。

        > [!NOTE]说明
        > “/usr/local/python3.7.5/lib/python3.7/site-packages”是pip安装第三方库的路径，可以使用**pip3 -V**检查。
        > 如果系统显示：/usr/local/python3.7.5/lib/python3.7/site-packages/pip，则pip安装第三方库的路径为/usr/local/python3.7.5/lib/python3.7/site-packages。

    4. 若用户在`--prefix`中指定了安装路径，则需在运行脚本中增加环境变量的设置：

        ```bash
        export LD_LIBRARY_PATH=/protobuf/lib:${LD_LIBRARY_PATH}
        ```

        "/protobuf"为`--prefix`中用户配置的安装路径。

    5. 建立软链接。

        当用户自行配置安装路径时，需要建立软链接，否则导入TensorFlow会报错。命令如下：

        ```bash
        ln -s /protobuf/lib/libprotobuf.so.22.0.3 /usr/lib/libprotobuf.so.22
        ```

        其中"/protobuf"为`--prefix`中用户配置的安装路径。
