# keep_dtype_scope

## Description

Specifies the operators that preserve the original precision. If the operator precision in an original network model is not supported by the  AI processor, the system automatically uses the high precision supported by the operators for compute.

## Prototype

```python
def keep_dtype_scope()
```

## Parameters

None

## Restrictions

This API does not take effect if the original precision is preserved \(that is,  **precision_mode**  is set to  **must_keep_origin_dtype**\).

## Returns

None

## Example

```python
with npu_scope.keep_dtype_scope(): 
    X = tf.conv2d(a)
```
