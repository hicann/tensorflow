# GetNext算子超时，返回错误码E30008

## 问题现象

执行训练脚本时出现GetNext算子超时。

```text
Error Message is :
E30008: AI CPU operator execution time out.
Possible Cause: 1. For the GetNext operator, its preprocessing duration may be too long. 2. For a custom operator, its logic may be improper.
Solution: 1. For the GetNext operator, check its preprocessing or set OpExecuteTimeOut to a larger value. 2. For a custom operator, make sure its logic is proper.
TraceBack (most recent call last):
Aicpu kernel execute failed, device_id=0, stream_id=2, task_id=2, fault op_name=aicpu_getnext_IteratorGetNext[FUNC:GetError][FILE:stream.cc][LINE:1133]
rtStreamSynchronizeWithTimeout execute failed, reason=[aicpu timeout][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:49]

[[{{node GeOp2_0}}]]
```

## 原因分析

可通过查询Device日志定界，日志中记录了GetNext算子执行超时后Device侧驱动队列中的出队/入队相关信息：

```text
debug/device-0/device-1927_20230406202333033.log:147:[ERROR] AICPU(3480,aicpu_scheduler):2023-04-06-20:24:34.559.758 [kernel_util.cc:101][AICPU][operator():101][tid:3543]:device_id:0, queue_name:12658048656348665736, queue_id:1, size:0, depth:2, status:1, workMode:1, type:2, enqueCnt:0, dequeCnt=0, enqueFailCnt=0, dequeFailCnt=1, enqueEventOk=0, enqueEventFail=0, FullToNotFullEventOkCnt=0, FullToNotFullEventFailCnt = 0, lastEnqueTime.tv_sec:0, lastEnqueTime.tv_usec:0, lastDequeTime.tv_sec:0, lastDequeTime.tv_usec:0
```

可通过入队/出队信息确认数据集是否正常发送到了Device侧，如果入队数量很少，则可能是数据集生成不稳定、数据集传输网络不稳定或者预处理阶段耗时较大导致。

## 解决方法

针对以上可能原因，可参考以下步骤处理：

1. 检查训练模型的输入数据集是否正常生成以及数据传输是否稳定。
2. 检查Host侧预处理过程处理逻辑是否存在耗时较大情况（数据集正常的情况下，GetNext算子超时后AI CPU记录的出队/入队相关信息中“enqueCnt”较小或者“lastEnqueTime”较大，则说明预处理阶段耗时大），如果确认预处理阶段耗时较久，可通过“op_execute_timeout”配置参数修改算子超时时间。
