# set_schedule_aicore_task_early

## 功能说明

设置是否开启AI Core任务提前下发模式。开启该模式后，AI Core任务可以提前调度，以降低调度延迟，提升执行效率。

## 函数原型

```python
def set_schedule_aicore_task_early(value)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| value | 输入 | 是否开启AI Core任务提前下发模式。<br><br>  - True：开启该模式。<br>  - False：关闭该模式。 |

## 返回值

0：成功。其他值：失败。

## 约束说明

该接口需要在运行时配置，网络脚本执行前调用。

## 调用示例

```python
import tensorflow as tf
from npu_bridge.npu_init import *

......
# 网络执行时调用如下接口开启AI Core任务提前下发模式
npu_plugin.set_schedule_aicore_task_early(True)
sess.run(xxx)
```
