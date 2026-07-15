# set_op_input_tensor_multi_dims

## Description

Applies to subgraph-wide dynamic shape profiles. Specifies the input shape of the operator and shape profiles.

**Note: This API is a trial API and should not be directly used by developers.**

## Prototype

```python
def set_op_input_tensor_multi_dims(tensor, input_shape, input_dims)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| tensor | Input | Output tensor of the target operator. |
| input_shape | Input | Shapes of all input tensors of the target operator. For example:<br>Assume the target operator has two input tensors:<br>"0: -1, -1, 3; 1: 2, 2, -1, 224" |
| input_dims | Input | Dynamic dimension size profiles. Separate profiles with semicolons (;). The number and sequence of dimension sizes in each profile must be the same as those of -1 in input_shape. For example:<br>Assume there are three profiles:<br>"480, 480, 112; 960, 960, 224; 1920, 1920, 448" |

## Returns

None

## Restrictions

This API must work with the  **subgraph_multi_dims_scope**  API, and the target operator must be the input node of the target scope.

## Example

For details, see  [Example](../npu_scope/subgraph_multi_dims_scope.md#example).
