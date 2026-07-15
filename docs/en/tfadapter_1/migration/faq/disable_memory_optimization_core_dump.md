# How Do I Resolve Core Dumps Caused by Disabled memory_optimization?

## Symptom

The memory_optimization function of TensorFlow is disabled during porting. As a result, a core dump occurs.

```text
tensorflow/core/grappler/optimizers/memory_optimizer.cc xxx (core dump)
```

## Possible Cause

In the multi-device scenario, it is advised to use the memory optimization logic of the NPU for porting instead of memory_optimization, as it may incur errors when executed on the NPU. Make the following configuration to disable the memory_optimization function:

```python
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
```

However, the native TensorFlow code should ensure that the network runs properly regardless of whether memory_optimization is enabled or disabled. The occurrence of a TensorFlow core dump after memory_optimization is disabled indicates that the issue originates from within TensorFlow itself.

## Solution

You are advised to comment out the following line to enable memory_optimization:

```python
# config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
```
