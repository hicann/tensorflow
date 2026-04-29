# 替换GELU激活函数

GELU是神经网络中一种常见的激活函数，全称为“Gaussian Error Linear Unit”，是ReLU的一种平滑版本，具体可以参考[论文解释](https://arxiv.org/abs/1606.08415)。TensorFlow中GELU对应的实现接口为tf.nn.gelu和tf.keras.activations.gelu，部分网络如BERT中也会使用自定义的GELU实现.。使用近似的实现（如将tf.nn.gelu的approximate参数设置为True，或者BERT中的自定义GELU实现）在训练时可以得到更好的性能。NPU上也提供了高性能的GELU近似实现接口[npu.ops.gelu](../../apiref/npu-ops-gelu.md)，将GELU在下沉执行时获得更好的性能。

如果你要使用NPU上GELU接口，请注意以下两点：

1. 只有在function模式下，下沉到NPU执行时调用npu.ops.gelu才能提升性能；在Eager模式下，npu.ops.gelu不会得到比TensorFlow原生接口更好的性能。
2. NPU提供的GELU接口是近似实现，并不保证在所有场景下都能替换标准实现的GELU接口并达到收敛，需要针对具体的网络实现进行尝试。

鉴于以上两点，目前自动迁移工具不会自动迁移所有GELU接口，如果你仍希望通过替换GELU激活函数获取性能提升，请参考以下步骤：

1. 引入npu_device模块

    ```python
    import npu_device as npu
    ```

2. （function模式下）找到网络中定义或者使用GELU接口的地方，替换为npu_device.ops.gelu：

    ```python
    def gelu(x):
      """Gaussian Error Linear Unit
      This is a smoother version of the ReLU.
      Original paper: https://arxiv.org/abs/1606.08415
      Args:
      x: float Tensor to perform activation
      Returns:
      `x` with the GELU activation applied
      """
      return npu.ops.gelu(x)
      # return tf.keras.activations.gelu(x, approximate=True)
    ```
