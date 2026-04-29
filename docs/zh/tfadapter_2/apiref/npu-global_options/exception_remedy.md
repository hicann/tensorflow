# 异常补救

## stream_sync_timeout

图执行时，stream同步等待超时时间，超过配置时间时报同步失败。

单位：ms，默认值-1，表示无等待时间，出现同步失败不报错。

说明：集群训练场景下，此配置的值（即stream同步等待超时时间）需要大于集合通信超时时间，即环境变量HCCL_EXEC_TIMEOUT的值。HCCL_EXEC_TIMEOUT的详细说明可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》的“集合通信”章节。

配置示例：

```python
npu.global_options().stream_sync_timeout=600000
```

## event_sync_timeout

图执行时，event同步等待超时时间，超过配置时间时报同步失败。

单位：ms，默认值-1，表示无等待时间，出现同步失败不报错。

配置示例：

```python
npu.global_options().event_sync_timeout=600000
```
