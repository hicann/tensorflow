# Basic Options

## graph_run_mode

Graph run mode.

- 0: online inference.
- 1 (default): training

Example:

```python
npu.global_options().graph_run_mode=1
```

## deterministic

Whether to enable deterministic computing. If enabled, the same output is generated when an operator is executed for multiple times with the same hardware and input.

The values are as follows:

- 0 (default): disables deterministic computing.
- 1: enables deterministic computing.

By default, deterministic computing does not need to be enabled, because it slows down operator execution and affects performance. If it is disabled, the results of multiple executions may be different. This is generally caused by asynchronous multi-thread executions during operator implementation, which changes the accumulation sequence of floating point numbers.

However, if the execution results of a model are different for multiple times or the precision needs to be tuned, you can enable deterministic computing to assist model debugging and tuning. Note that if you want a completely definite result, you need to set a definite random seed in the training script to ensure that the random numbers generated in the program are also definite.

Example:

```python
npu.global_options().deterministic=1
```
