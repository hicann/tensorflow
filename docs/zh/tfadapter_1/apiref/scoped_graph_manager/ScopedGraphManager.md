# ScopedGraphManager

## 功能说明

可通过该接口一次性卸载变量初始化图，并释放其中常量节点占用的内存。

## 函数原型

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

## 参数说明

无

## 返回值

无

## 约束说明

- ScopedGraphManager类需要通过with语句调用，且一旦超出该作用域，作用域内执行的变量初始化图将被自动卸载，所占用的常量内存也会被释放。
- 仅变量初始化图可以放置在ScopedGraphManager所在作用域中，如果非变量初始化图放到了此作用域，可能会影响训练功能的正常运行。
- 该接口仅适用于在主进程中一次性进行变量初始化的场景，不支持多线程、多session或多次执行变量初始化的场景。

## 调用示例

```python
import tensorflow as tf
from npu_bridge.npu_init import *
with scoped_graph_manager.ScopedGraphManager():
    # 执行变量初始化
    sess.run(tf.global_variables_initializer())
```
