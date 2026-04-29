# compat.v1模式下使用工具迁移Horovod脚本后，执行失败

工具在compat.v1模式下迁移Horovod脚本时，会自动替换相关Horovod接口，同时删除原始脚本中的Horovod相关包引用，此时对于原始脚本中的某些特殊写法，可能会导致关联代码报错，建议手工修改适配。例如：

迁移前：

```python
from horovod.common.util import env

def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  with env(HOROVOD_STALL_CHECK_TIME_SECONDS="300"):
      hvd.init()
```

迁移后：

```python
None

def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  with env(HOROVOD_STALL_CHECK_TIME_SECONDS="300"):
      hvd.init()
```

包引用删除后导致找不到env模块，因此可以通过重新添加包引用解决问题：

```python
from horovod.common.util import env

def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  with env(HOROVOD_STALL_CHECK_TIME_SECONDS="300"):
      hvd.init()
```

同时也建议根据实际代码逻辑，判断是否保留这些代码。
