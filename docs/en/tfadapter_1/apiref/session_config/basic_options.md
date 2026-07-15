# Basic Options

## graph_run_mode

Graph run mode.

- 0: online inference.
- 1 (default): training

Example:

```python
custom_op.parameter_map["graph_run_mode"].i = 1
```

## session_device_id

Logical ID of a device. Setting this parameter allows you to run different models on multiple devices by executing a single training script.

You can create different sessions for different graphs and pass different session_device_id values.

Example:

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

Whether to enable deterministic computing. If enabled, the same output is generated if an operator is executed for multiple times with the same hardware and input.

The values are as follows:

- 0 (default): disables deterministic computing.
- 1: enables deterministic computing.

By default, deterministic computing does not need to be enabled, because it slows down operator execution and affects performance. If it is disabled, the results of multiple executions may be different. This is generally caused by asynchronous multi-thread executions during operator implementation, which changes the accumulation sequence of floating point numbers.

However, if a model produces inconsistent execution results across multiple runs or requires accuracy optimization, you can enable deterministic computing to assist with model debugging and tuning. Note that if you want a completely definite result, you need to set a definite random seed in the training script to ensure that the random numbers generated in the program are also definite.

Example:

```python
custom_op.parameter_map["deterministic"].i = 1
```
