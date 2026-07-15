# ScopedGraphManager

## Description

Unloads the variable initialization graph in one go and releases the memory held by constant nodes in the graph.

## Prototype

```python
class ScopedGraphManager(object):
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exe_type, exe_val, exc_tb):
        self.stop()

    def start(self):
        tf_adapter.EnableControl()

    def stop(self):
        tf_adapter.Clear()
```

## Parameters

None

## Returns

None

## Restrictions

- The  **ScopedGraphManager**  class needs to be called using the  **with**  statement. Once the scope is exceeded, the variable initialization graph executed in the scope is automatically unloaded, and the occupied constant memory is released.
- Only a variable initialization graph can be placed in the scope of  **ScopedGraphManager**. If a non-variable initialization graph is placed in this scope, it may interfere with the normal operation of training functions.
- This API is applicable only to the scenario where variables are initialized at a time in the main process. It does not support scenarios where variables are initialized in multiple threads, sessions, or times.

## Example

```python
import tensorflow as tf
from npu_bridge.npu_init import *

with scoped_graph_manager.ScopedGraphManager():
    # Initialize variables.
    sess.run(tf.global_variables_initializer())
```
