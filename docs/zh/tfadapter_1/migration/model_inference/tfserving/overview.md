# 概述

TensorFlow Serving是一款针对机器学习模型的灵活，高性能的服务系统，专为生产环境设计。使用SavedModel格式模型，提供RESTful API + gRPC对外接口，依赖TensorFlow源码，官网链接：[www.tensorflow.org/tfx/guide/serving?hl=zh-cn](https://www.tensorflow.org/tfx/guide/serving?hl=zh-cn)。

使用TensorFlow Serving可以方便地部署新算法和实验，同时保持相同的服务器体系结构和API。TensorFlow Serving实际上是封装了TensorFlow，提供服务化的能力，并且从性能考虑核心代码都采用C++，依赖的TensorFlow也是C++版本。

![](../../figures/tfserving_overview.png)

本文介绍如何基于TF Adapter和TF Serving源码进行编译，以便TF Serving通过TensorFlow可加载TF Adapter插件，最终使用昇腾AI处理器进行在线推理。
