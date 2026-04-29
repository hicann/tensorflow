# set_op_input_tensor_multi_dims

## 功能说明

在线推理场景下，使用子图分档功能，用于指定当前算子的输入shape以及所有分档档位的shape信息。

**注意：此接口是试验接口，不建议开发者直接使用。**

## 函数原型

```python
def set_op_input_tensor_multi_dims(tensor, input_shape, input_dims)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tensor | 输入 | 目标算子的任意输出tensor。 |
| input_shape | 输入 | 目标算子所有输入tensor的shape，例如：<br>若目标算子有2个输入tensor：<br>"0: -1, -1, 3; 1: 2, 2, -1, 224" |
| input_dims | 输入 | 动态dim的分档档位信息，每个档位间用";"分隔，档位内dim数量与input_shape中"-1"的数量和顺序一致，例如：<br>若共有3个档位：<br>"480, 480, 112; 960, 960, 224; 1920, 1920, 448" |

## 返回值

无

## 使用约束

该接口需要配合分档范围划分接口（即subgraph_multi_dims_scope）使用，且目标算子必须为分档范围的输入节点，否则不生效。

## 调用示例

请参见[调用示例](../npu_scope/subgraph_multi_dims_scope.md#调用示例)。
