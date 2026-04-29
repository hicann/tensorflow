# 迁移背景

TensorFlow 2.6.5中的compat.v1模块用于兼容TensorFlow 1.x中的API，使开发者可以在TensorFlow 2.6.5中使用1.x版本的API。

在实际场景中，存在部分TensorFlow 2.6.5脚本以tf.compat.v1形式调用TensorFlow 1.x API的情况，用于以TensorFlow 1.x原有的Session方式（Estimator/Session/Keras）控制脚本的执行行为。对于这类情况的训练脚本，支持迁移到AI处理器上，且迁移点与TensorFlow 1.x的迁移基本一致。
