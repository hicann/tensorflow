# 基础功能

## graph_run_mode

图执行模式。

- 0：在线推理场景下，请配置为0。
- 1（默认值）：训练场景下，请配置为1。

配置示例：

```python
custom_op.parameter_map["graph_run_mode"].i = 1
```

## session_device_id

当用户需要将不同的模型通过同一个脚本在不同的Device上执行，可以通过该参数指定Device的逻辑ID。

通常可以为不同的图创建不同的Session，并且传入不同的session_device_id。

配置示例：

```python
config_0 = tf.ConfigProto()
custom_op = config_0.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["session_device_id"].i = 0
config_0.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config_0.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
with tf.Session(config=config_0) as sess_0:
    sess_0.run(...)

config_1 = tf.ConfigProto()
custom_op = config_1.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["session_device_id"].i = 1
config_1.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config_1.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
with tf.Session(config=config_1) as sess_1:
    sess_1.run(...)

config_7 = tf.ConfigProto()
custom_op = config_7.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["session_device_id"].i = 7
config_7.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config_7.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
with tf.Session(config=config_7) as sess_7:
    sess_7.run(...)
```

## deterministic

是否开启确定性计算，开启确定性开关后，算子在相同的硬件和输入下，多次执行将产生相同的输出。

此配置项有以下两种取值：

- 0（默认值）：不开启确定性计算。
- 1：开启确定性计算。

默认情况下，无需开启确定性计算。因为开启确定性计算后，算子执行时间会变慢，导致性能下降。在不开启确定性计算的场景下，多次执行的结果可能不同。这个差异的来源，一般是因为在算子实现中，存在异步的多线程执行，会导致浮点数累加的顺序变化。

但当发现模型执行多次结果不同，或者精度调试时，可以通过此配置开启确定性计算辅助进行调试调优。需要注意，如果希望有完全确定的结果，在训练脚本中需要设置确定的随机数种子，保证程序中产生的随机数也都是确定的。

配置示例：

```python
custom_op.parameter_map["deterministic"].i = 1
```
