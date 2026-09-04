# get_schedule_aicore_task_early

## Description

Queries whether the AI Core task early start mode is enabled.

## Prototype

```python
def get_schedule_aicore_task_early()
```

## Parameters

None

## Returns

True: the AI Core task early start mode is enabled. False: the mode is disabled.

## Restrictions

None

## Example

```python
import tensorflow as tf
from npu_bridge.npu_init import *

......
# Enable the AI Core task early start mode.
npu_plugin.set_schedule_aicore_task_early(True)
# Query whether the mode is enabled.
print(npu_plugin.get_schedule_aicore_task_early())  # True
sess.run(xxx)
```
