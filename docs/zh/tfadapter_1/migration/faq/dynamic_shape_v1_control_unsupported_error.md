# 动态shape网络执行时报V1控制流算子不支持的错误

## 问题现象

模型执行时报如下错误：

```text
node node_name(node_type) is v1 control operator, which is not supported, please convert to v2 control operator
```

## 原因分析

这是由于当前网络为动态shape网络，且存在TensorFlow V1版本的控制流算子。动态shape网络的执行当前不支持V1版本的控制流算子，所以会造成网络执行失败。

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
