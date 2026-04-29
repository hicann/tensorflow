# 数据预处理使用tf.keras.backend.zeros生成变量，训练报错变量未初始化

## 问题现象

训练报错变量算子未初始化：

![](../figures/keras-backend-zeros-faq.png)

## 原因分析

数据预处理使用tf.keras.backend.zeros生成变量，变量算子无法下沉到Device侧执行，从而导致变量算子初始化失败。

## 解决方案

修改训练脚本，不使用tf.keras.backend.zeros生成变量，而直接使用TensorFlow原生接口tf.zeros以tensor形式在Host侧生成变量。

原始脚本：

```python
y = {
   'mlm_loss': tf.keras.backend.zeros([1]),
   'mlm_acc': tf.keras.backend.zeros([1]),
}
```

修改后脚本：

```python
y = {
   'mlm_loss': tf.zeros([1]),
   'mlm_acc': tf.zeros([1]),
}
```
