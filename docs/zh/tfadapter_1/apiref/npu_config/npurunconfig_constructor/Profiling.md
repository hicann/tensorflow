# Profiling

## profiling_config

Profiling开关，用户在创建NPURunConfig之前，可以实例化一个ProfilingConfig类进行Profiling的配置。ProfilingConfig类的构造函数，请参见[ProfilingConfig构造函数](../profilingconfig_constructor.md)。

配置示例：

```python
config = NPURunConfig(profiling_config=profiling_config)
```
