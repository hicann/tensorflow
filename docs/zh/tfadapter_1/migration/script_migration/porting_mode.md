# 迁移方式比较

将基于TensorFlow的Python API开发的训练脚本迁移到AI处理器上执行训练，目前有两种迁移方式，用户可任选其一：

- [自动迁移](automated_porting.md)

    算法工程师通过迁移工具，可自动分析出原生的TensorFlow Python API和Horovod Python API在AI处理器上的支持度情况，同时将原生的TensorFlow训练脚本自动迁移成AI处理器支持的脚本，对于无法自动迁移的API，可以参考工具输出的迁移报告，对训练脚本进行相应的适配修改。

- [手工迁移](manual_porting.md)

    算法工程师通过自行修改TensorFlow训练脚本，以支持在AI处理器上执行训练，该种方式较为复杂，建议优先使用自动迁移方式。
