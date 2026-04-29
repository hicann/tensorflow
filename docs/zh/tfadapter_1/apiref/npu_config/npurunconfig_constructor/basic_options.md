# 基础功能

## graph_run_mode

图执行模式。

- 0：在线推理场景下，请配置为0。
- 1（默认值）：训练场景下，请配置为1。

配置示例：

```python
config = NPURunConfig(graph_run_mode=1)
```

## session_device_id

当用户需要将不同的模型通过同一个训练脚本在不同的Device上执行，可以通过该参数指定Device的逻辑ID。

通常可以为不同的图创建不同的Session，并且传入不同的session_device_id，该参数优先级高于环境变量ASCEND_DEVICE_ID**。**

配置示例：

```python
config0 = NPURunConfig(..., session_device_id=0, ...)
estimator0 = NPUEstimator(..., config=config0, ...)
# ...
config1 = NPURunConfig(..., session_device_id=1, ...)
estimator1 = NPUEstimator(..., config=config1, ...)
# ...
config7 = NPURunConfig(..., session_device_id=7, ...)
estimator7 = NPUEstimator(..., config=config7, ...)
# ...
```

## distribute

通过PS-Worker架构进行分布式训练时，用于传入ParameterServerStrategy对象。

配置示例：

```python
config = NPURunConfig(distribute=strategy)
```

## deterministic

是否开启确定性计算，开启确定性开关后，算子在相同的硬件和输入下，多次执行将产生相同的输出。

- 0（默认值）：不开启确定性计算。
- 1：开启确定性计算

默认情况下，无需开启确定性计算。因为开启确定性计算后，算子执行时间会变慢，导致性能下降。在不开启确定性计算的场景下，多次执行的结果可能不同。这个差异的来源，一般是因为在算子实现中，存在异步的多线程执行，会导致浮点数累加的顺序变化。

但当发现模型执行多次结果不同，或者精度调优时，可以通过此配置开启确定性计算辅助进行调试调优。需要注意，如果希望有完全确定的结果，在训练脚本中需要设置确定的随机数种子，保证程序中产生的随机数也都是确定的。

配置示例：

```python
config = NPURunConfig(deterministic=1)
```
