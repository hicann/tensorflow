# 数据预处理中存在tf.Variable导致训练异常

## 问题现象

TensorFlow网络执行时，报如下错误：

```text
tensorflow.python.framework.errors_impl.FailedPreconditionError: Error while reading resource variable inference/embed_continuous from Container: localhost.  This could mean that the variable was uninitialized. Not found: Resource localhost/inference/embed_continuous/N10tensorflow3VarE does not exist.
```

## 原因分析

此问题是由于数据预处理脚本中存在tf.Variable变量。训练脚本在昇腾平台运行时，tf.Variable变量在Host侧执行，而tf.Variable变量的初始化在Device侧执行，变量执行和变量初始化不在同一设备，导致训练异常。

使用了tf.Variable的训练脚本代码示例如下：

```python
batch_size = tf.Variable(
    tf.placeholder(tf.int64, [], 'batch_size'),
    trainable= False, collections=[]
)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
```

## 解决方案

修改训练脚本，将tf.Variable修改成常量，修改示例如下：

```python
batch_size = 64
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
```
