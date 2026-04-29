# shutdown_system

## 功能说明

关闭所有Device，和[initialize_system](initialize_system.md)配合使用。

## 函数原型

```python
def shutdown_system(name = None)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| name | 输入 | 算子名称。 |

## 返回值

返回一个op，供用户通过sess.run\(op\)完成设备关闭。
