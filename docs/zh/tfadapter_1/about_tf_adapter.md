# TF Adapter简介

TF Adapter为支持TensorFlow图在NPU上执行的TensorFlow插件，主要目的是将TensorFlow图转换为NPU上可执行的图。

TF Adapter在昇腾AI软件栈中的位置如下图所示。

**图 1**  昇腾AI软件栈架构图
![](./migration/figures/ascend_architecture.png "昇腾AI软件栈架构图")

TF Adapter架构如下所示。

**图 2**  TF Adapter架构
![](./migration/figures/tfadapter_architecture.png "TF-Adapter架构")

上图左侧是TensorFlow 1.15框架架构示例，右侧为TF Adapter架构示例，可以看出TensorFlow框架的每一层在TF Adapter中都有对应的实现。

- Python API

  TF Adapter提供了适配TensorFlow框架的用户Python接口，提供如下功能：

  - 提供session策略，包括功能调试、精度调优、性能调优等配置项。
  - 提供便于在NPU上执行模型训练的NPUEstimator高阶API。
  - 提供资源初始化、分布式训练等API。

- 图优化器

    图优化器的作用是接收TensorFlow下发的子图，识别可下沉到Device执行的算子，并将包含这部分算子的子图下沉到Device执行。

- GEOP

    GEOP为TF Adapter扩展的TensorFlow算子，作用是将标识的子图下沉到Device执行。

- GE Model

    经过TF Adapter适配后的GE可执行图。
