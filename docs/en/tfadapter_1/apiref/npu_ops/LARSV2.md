# LARSV2

## Description

This operator scales gradients based on the norm of weight and the norm of gradient at different levels using different learning rates. It is used to improve the training precision in large batch size scenarios and is used for large-scale cluster training to reduce the training time.

## Prototype

```python
def LARSV2(input_weight,
           input_grad,
           weight_decay,
           learning_rate,
           hyperpara=0.001,
           epsilon=0.00001,
           use_clip=False,
           name=None)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| input_weight | Input | Weight tensor of type float. |
| input_grad | Input | Weight gradient tensor of type float. |
| weight_decay | Input | Scalar tensor of type float. |
| learning_rate | Input | Scalar tensor of type float, indicating the learning rate. |
| hyperpara | Input | Scalar of type float, for the hyperparameter of the operator. Generally it is set to 0.001. |
| epsilon | Input | A scalar, added to avoid dividing by zero. Generally it is set to 1e-5. |
| use_clip | Input | A bool. Defaults to False.<br>If this parameter is set to True, the scaling coefficient must be within a specified range. |
| name | Input | Name of the network layer. |

## Returns

Result gradient tensor

## Example

```python
from npu_bridge.npu_init import *
layers = npu_ops.LARSV2(input_weight , input_grad, weight_decay, learning_rate)
```
