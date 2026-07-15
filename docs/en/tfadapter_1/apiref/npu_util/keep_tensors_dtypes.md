# keep_tensors_dtypes

## Description

Specifies the operators that preserve the original precision.

## Prototype

```python
def keep_tensors_dtypes(graph, input_tensors)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| graph | Input | Graph imported from the PB model |
| input_tensors | Input | Name of the operator whose original precision needs to be preserved. |

## Returns

None

## Restrictions

- This API works only in online inference scenarios.
- This API does not take effect if the original precision is preserved \(that is,  **precision_mode**  is set to  **must_keep_origin_dtype**\).

## Example

```python
from npu_bridge.estimator.npu import util
g=tf.Graph()
util.keep_tensors_dtypes(g,("random_uniform_1/sub:0",))
```
