# 执行分布式训练时，出现get rank id error错误

## 问题现象

屏显信息出现“get rank id error”的错误，如下所示：

![](../figures/get_rank_id_error.png)

查看Host日志，出现“Call hcom_bind_model failed”的错误信息，如下所示：

![](../figures/hcom_bind_failed.png)

## 原因分析

集合通信的管理类Python接口需要在完成集合通信初始化之后调用，才能正常执行。

## 解决方法

训练脚本中，在完成集合通信初始化后再调用集合通信的管理类Python接口，集合通信初始化详细描述可参见[集合通信初始化](../others/init_collective_communication.md)。
