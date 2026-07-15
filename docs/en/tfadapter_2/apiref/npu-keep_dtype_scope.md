# npu.keep_dtype_scope

## Description

Specifies the operators that preserve the original precision. If the operator precision in an original network model is not supported by the  AI processor, the system automatically uses the high precision supported by the operators for compute.

## Prototype

```python
npu.keep_dtype_scope()
```

## Parameters

None

## Returns

Python context manager. Operators in the context have special attributes identified by the NPU.

## Example

```python
import npu_device as npu
with npu.keep_dtype_scope():
    v = tf.add(1, 1)
```
