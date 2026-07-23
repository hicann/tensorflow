# Exception Remedy

## stream_sync_timeout

Timeout for stream synchronization during graph execution. If the timeout exceeds the configured value, a synchronization failure is reported. The unit is ms.

The default value is -1, indicating that there is no waiting time and no error is reported when the synchronization fails.

Note: For cluster training, the value of this option (stream synchronization waiting timeout) must be greater than the collective communication timeout, which means the value of the environment variable HCCL_EXEC_TIMEOUT. For details about HCCL_EXEC_TIMEOUT, see section Collective Communication in the [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

Example:

```python
npu.global_options().stream_sync_timeout=600000
```

## event_sync_timeout

Timeout for event synchronization during graph execution. If the timeout exceeds the configured value, a synchronization failure is reported. The unit is ms.

The default value is -1, indicating that there is no waiting time and no error is reported when the synchronization fails.

Example:

```python
npu.global_options().event_sync_timeout=600000
```
