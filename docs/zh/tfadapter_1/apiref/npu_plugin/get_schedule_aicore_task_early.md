# get_schedule_aicore_task_early

## 功能说明

查询是否已开启AI Core任务提前下发模式。

## 函数原型

```python
def get_schedule_aicore_task_early()
```

## 参数说明

无

## 返回值

True：已开启AI Core任务提前下发模式。False：未开启该模式。

## 约束说明

无

## 调用示例

```python
import tensorflow as tf
from npu_bridge.npu_init import *

......
# 开启AI Core任务提前下发模式
npu_plugin.set_schedule_aicore_task_early(True)
# 查询是否已开启该模式
print(npu_plugin.get_schedule_aicore_task_early())  # True
sess.run(xxx)
```
