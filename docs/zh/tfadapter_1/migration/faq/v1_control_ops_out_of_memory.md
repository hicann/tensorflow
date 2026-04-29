# V1版本控制流算子导致内存不足

## 问题现象

模型执行时报错，内存超过31G，导致内存分配不足。

![](../figures/tf_v1_faq.png)

## 原因分析

发现该网络的图结构中有switch-\>merge的V1控制流结构。当网络中的分支结构较多且采用V1版本的控制流算子时，可能影响内存复用效果，导致内存不足。

## 解决方案

将网络中的TensorFlow V1版本的控制流算子转换为V2版本。

- 方式一（推荐）：修改网络脚本，在import tensorflow as tf后增加如下两条指令，将TensorFlow V1版本的控制流算子转换为V2版本。

    ```python
    tf.enable_control_flow_v2()
    tf.enable_resource_variables()
    ```

- 方式二：配置环境变量，将V1版本的控制流算子转换为V2版本。

    ```bash
    export ENABLE_FORCE_V2_CONTROL=1
    ```

    注意：使用该环境变量，**可能会存在V1版本控制流算子到V2版本控制流算子转换失败的场景**，例如网络脚本中带ref控制算子的场景。
