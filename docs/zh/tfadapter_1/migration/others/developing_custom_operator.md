# 自定义算子开发

当TensorFlow网络中存在CANN不支持的算子时，开发者可通过自定义实现Ascend C算子并在TensorFlow框架适配完成，具体的流程如下图所示：

![](../figures/customop.png)

1. 自定义算子开发。

    基于Ascend C完成自定义算子的开发，包括如下几个步骤：

    1. 创建算子工程。
    2. 实现算子，进行算子的原型定义、Kernel侧算子实现与Host侧Tiling实现。
    3. 算子入图开发，主要进行shape推导等算子入图适配函数的实现。

2. TensorFlow框架适配插件开发。

    通过REGISTER_CUSTOM_OP注册自定义算子并完成TensorFlow算子到CANN算子的映射。对于TensorFlow自定义算子映射到CANN算子的场景，还需要完成TensorFlow自定义算子的开发。

3. 算子工程编译部署，编译生成自定义算子安装包并进行算子包的安装，将自定义算子部署到算子加速库。
4. TensorFlow框架算子调用。

上述流程的详细描述可参见《[Ascend C算子开发指南](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)》。
