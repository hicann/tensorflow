# What Do I Do If the GetNext Operator Times Out and Returns Error Code E30008?

## Symptom

The GetNext operator times out during training script execution.

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

## Possible Cause

You can query the device log to demarcate the fault. The log records the dequeue/enqueue information in the driver queue on the device after the GetNext operator times out.

```text
debug/device-0/device-1927_20230406202333033.log:147:[ERROR] AICPU(3480,aicpu_scheduler):2023-04-06-20:24:34.559.758 [kernel_util.cc:101][AICPU][operator():101][tid:3543]:device_id:0, queue_name:12658048656348665736, queue_id:1, size:0, depth:2, status:1, workMode:1, type:2, enqueCnt:0, dequeCnt=0, enqueFailCnt=0, dequeFailCnt=1, enqueEventOk=0, enqueEventFail=0, FullToNotFullEventOkCnt=0, FullToNotFullEventFailCnt = 0, lastEnqueTime.tv_sec:0, lastEnqueTime.tv_usec:0, lastDequeTime.tv_sec:0, lastDequeTime.tv_usec:0
```

It can be confirmed whether the dataset has been sent to the device normally by checking the information of enqueue/dequeue. If the number of enqueued data records is small, it may be due to unstable dataset generation, unstable dataset transmission network, or long preprocessing time.

## Solution

To rectify the fault, perform the following steps:

1. Check if the input dataset for the training model is generated normally and if the data transmission is stable.
2. Check whether there is a long processing time in the host preprocessing logic \(if the dataset is normal, a small  **enqueCnt**  or a large  **lastEnqueTime**  in the dequeue/enqueue statistics recorded by the AI CPU after GetNext operator timeout indicates that the preprocessing stage takes a long time\). If it is confirmed that the preprocessing stage takes a long time, you can change the operator timeout interval using the  **op_execute_timeout**  parameter.
