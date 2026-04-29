# 关闭memory_optimization后发生core dump

## 问题现象

迁移时关闭了TensorFlow的memory_optimization功能**，**发生了core dump：

```text
tensorflow/core/grappler/optimizers/memory_optimizer.cc xxx (core dump)
```

## 原因分析

多p场景下，memory_optimization在NPU上执行可能会出现问题，因此迁移时要求关闭，使用NPU的内存优化逻辑，关闭配置如下：

```python
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
```

但是无论memory优化器开启或关闭，TensorFlow原生代码都应该保证网络正常进行。关闭后出现TensorFlow core dump，说明是TensorFlow自身问题导致。

## 解决方案

如果出现上述错误，建议用户注释如下代码，开启memory_optimization：

```python
# config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
```
