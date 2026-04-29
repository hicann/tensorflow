# NPULossScaleOptimizer优化器使用常见问题

## 问题现象

使用NPULossScaleOptimizer优化器时，如果训练脚本中存在如下类似情况：

```python
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
  train_op = tf.train.***Optimizer(0.01).minimize(loss)
```

会出现Placeholder not support报错。

## 原因分析

这是由于动态loss_scale_manager中存在变量，如果这样创建loss_scale_manager会导致Variable初始化时，执行出错。

## 解决方案

在创建动态loss_scale_manager时，需要将这个动作放在with的作用域之外。

```python
loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32,incr_every_n_steps=1000,decr_every_n_nan_or_inf=2,decr_ratio=0.5)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
  train_op = NPULossScaleOptimizer(tf.train.***Optimizer(0.01), loss_scale_manager).minimize(loss)
```
