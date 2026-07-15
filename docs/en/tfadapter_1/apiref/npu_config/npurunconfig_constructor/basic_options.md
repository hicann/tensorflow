# Basic Options

## graph_run_mode

Graph run mode. Values are as follows:

- 0: online inference.
- 1 (default): training

Example:

```python
config = NPURunConfig(graph_run_mode=1)
```

## session_device_id

Logical ID of a device. Setting this parameter allows you to run different models on multiple devices by executing a single training script.

Generally, you can create sessions for multiple graphs and pass the corresponding argument of session_device_id to the session. This parameter takes precedence over the environment variable ASCEND_DEVICE_ID.

Example:

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

ParameterServerStrategy object for distributed training in the PS-Worker architecture.

Example:

```python
config = NPURunConfig(distribute=strategy)
```

## deterministic

Whether to enable deterministic computing. If enabled, the same output is generated if an operator is executed for multiple times with the same hardware and input.

- 0 (default): disables deterministic computing.
- 1: enables deterministic computing.

By default, deterministic computing does not need to be enabled, because it slows down operator execution and affects performance. If it is disabled, the results of multiple executions may be different. This is generally caused by asynchronous multi-thread executions during operator implementation, which changes the accumulation sequence of floating point numbers.

However, if the execution results of a model are different for multiple times or the precision needs to be tuned, you can enable deterministic computing to assist model debugging and tuning. Note that if you want a completely definite result, you need to set a definite random seed in the training script to ensure that the random numbers generated in the program are also definite.

Example:

```python
config = NPURunConfig(deterministic=1)
```
