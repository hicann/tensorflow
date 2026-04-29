# 脚本实现流程与关键文件介绍

## 训练脚本实现流程

ResNet50原始网络脚本是Estimator模型的API，属于TensorFlow的高阶API，此训练脚本的实现流程为：

| 序号 | 过程 | 描述 |
|------|------|------|
| 1 | 数据预处理 | 创建输入函数input_fn。 |
| 2 | 模型构建 | 构建模型函数model_fn。 |
| 3 | 运行配置 | 实例化Estimator，并传入RunConfig类对象作为运行参数。 |
| 4 | 执行训练 | 在Estimator上调用训练方法Estimator.train()，利用指定输入对模型进行固定步数的训练。 |

## 关键文件介绍

关键文件目录结构如下所示（只列出部分需要修改文件，更多文件请查看获取的ResNet原始网络脚本）：

```text
├── r1
│   ├── resnet       // resnet主目录
│        ├── imagenet_main.py      // 基于ImageNet数据集训练网络模型
│        ├── imagenet_preprocessing.py     // ImageNet数据集数据预处理模块
│        ├── resnet_model.py    // resnet模型文件
│        ├── resnet_run_loop.py    // 数据输入处理与运行循环（训练、验证、测试）
├── utils
│   ├── flags
│   │   ├── _base.py     //定义模型的通用参数并设置默认值
```

| 文件名称 | 简介 |
|----------|------|
| imagenet_main.py | 包含ImageNet数据集数据预处理、模型构建定义、模型运行的相关函数接口。其中数据预处理部分包含get_filenames()、parse_record()、input_fn()、get_synth_input_fn()，_parse_example_proto()函数，模型部分包含ImagenetModel类、imagenet_model_fn()、run_cifar()、define_cifar_flags()函数。 |
| imagenet_preprocessing.py | ImageNet图像数据预处理接口，训练过程中包括使用提供的边界框对训练图像进行采样、将图像裁剪到采样边界框、随机翻转图像，然后调整到目标输出大小（不保留纵横比）。评估过程中使用图像大小调整（保留纵横比）和中央裁剪。 |
| resnet_model.py | ResNet模型的实现，包括辅助构建ResNet模型的函数以及ResNet block定义函数。 |
| resnet_run_loop.py | 模型运行文件，包括输入处理和运行循环两部分，输入处理包括对输入数据进行解码和格式转换，输出image和label，还根据是否是训练过程对数据的随机化、批次、预读取等细节做出了设定；运行循环部分包括构建Estimator，然后进行训练和验证过程。总体来看，是将模型放置在具体的环境中，实现数据与误差在模型中的流动，进而利用梯度下降法更新模型参数。 |
