# Script Execution Failure After the Horovod Script Is Migrated Using a Tool in compat.v1 Mode

When porting the Horovod script in compat.v1 mode, the tool automatically replaces the related Horovod APIs and deletes the Horovod package reference from the original script. In this case, some special writing methods in the original script may cause errors in the associated code. You must manually modify the configuration. The following provides an example.

Before migration

```python
from horovod.common.util import env

def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  with env(HOROVOD_STALL_CHECK_TIME_SECONDS="300"):
      hvd.init()
```

After migration

```python
None

def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  with env(HOROVOD_STALL_CHECK_TIME_SECONDS="300"):
      hvd.init()
```

After the package reference is removed, the env module cannot be found. You can add the package reference back to solve the problem.

```python
from horovod.common.util import env

def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  with env(HOROVOD_STALL_CHECK_TIME_SECONDS="300"):
      hvd.init()
```

You are advised to determine whether to retain the code based on your code logic.
