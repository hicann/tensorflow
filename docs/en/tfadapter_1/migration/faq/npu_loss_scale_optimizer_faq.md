# What Do I Do If an NPULossScaleOptimizer Error Occurs?

## Symptom

If your training script contains code similar to the following:

```python
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
  train_op = tf.train.***Optimizer(0.01).minimize(loss)
```

The error message "Placeholder not support" is displayed.

## Possible Cause

This is because variables exist in the dynamic  **loss_scale_manager**, creating which in this way may cause errors during variable initialization.

## Solution

When creating a dynamic  **loss_scale_manager**, place this action outside the scope of  **with**.

```python
loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32,incr_every_n_steps=1000,decr_every_n_nan_or_inf=2,decr_ratio=0.5)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
  train_op = NPULossScaleOptimizer(tf.train.***Optimizer(0.01), loss_scale_manager).minimize(loss)
```
