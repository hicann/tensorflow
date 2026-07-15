# Replacing the GELU Activation Function

Gaussian Error Linear Unit \(GELU\) is a common activation function in neural networks. It is a smooth version of ReLU. For details, see  [Explanation of the thesis](https://arxiv.org/abs/1606.08415). The implementation interfaces corresponding to the GELU in TensorFlow are  **tf.nn.gelu**  and  **tf.keras.activations.gelu**. Some networks such as BERT also use the customized GELU. Better performance can be achieved during training using an approximate implementation \(for example, setting the  **approximate**  parameter of  **tf.nn.gelu**  to  **True**, or using a custom GELU implementation in BERT\). The NPU also provides the high-performance GELU approximate implementation interface  [npu.ops.gelu](../../apiref/npu-ops-gelu.md)  to achieve better performance during offload execution.

If you want to use the GELU interface on the NPU, pay attention to the following points:

1. The performance is improved only when  **npu.ops.gelu**  is called during offload to NPU execution in function mode. In eager mode,  **npu.ops.gelu**  does not offer better performance than the native TensorFlow API.
2. The GELU interface provided by the NPU is an approximate implementation. It is not guaranteed that the GELU interface can replace the standard GELU interface and achieve convergence in all scenarios. Therefore, you need to try the GELU interface based on the specific network implementation.

In view of the preceding two points, not all GELU interfaces are automatically ported. If you still want to obtain performance improvement by replacing the GELU activation function, perform the following steps:

1. Import the npu_device module.

    ```python
    import npu_device as npu
    ```

2. In function mode, find the place where the GELU interface is defined or used on the network and replace it with  **npu_device.ops.gelu**.

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
