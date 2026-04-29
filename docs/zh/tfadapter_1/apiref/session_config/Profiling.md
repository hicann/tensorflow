# Profiling

## profiling_mode

是否开启Profiling功能。

- True：开启Profiling功能，从profiling_options读取Profiling的采集选项。
- False（默认值）：关闭Profiling功能。

配置示例：

```python
custom_op.parameter_map["profiling_mode"].b = True
```

说明：此配置项的优先级高于环境变量PROFILING_MODE，关于环境变量的详细说明可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》中的“性能数据采集”章节。

## profiling_options

Profiling配置选项。

- output：Profiling采集结果文件保存路径。支持配置绝对路径或相对路径（相对执行命令行时的当前路径）。路径中不能包含特殊字符："\\n"、"\\f"、"\\r"、"\\b"、"\\t"、"\\v"、"\\u007F"。
  - 绝对路径配置以“/”开头，例如：/home/output。
  - 相对路径配置直接以目录名开始，例如：output。
  - 该参数优先级高于ASCEND_WORK_PATH。
  - 该路径无需用户提前创建，采集过程中会自动创建。

- storage_limit：指定落盘目录允许存放的最大文件容量。当Profiling数据文件在磁盘中即将占满本参数设置的最大存储空间或剩余磁盘总空间即将被占满时（总空间剩余<=20MB），则将磁盘内最早的文件进行老化删除处理。

  范围\[200, 4294967295\]，单位为MB，设置该参数时必须带单位，例如200MB。

  未配置本参数时，采集前，如果磁盘可用空间小于20MB时，则不落盘数据。

- training_trace：采集迭代轨迹数据开关，即训练任务及AI软件栈的软件信息，实现对训练任务的性能分析，重点关注前后向计算和梯度聚合更新等相关数据。当采集正向和反向算子数据时该参数必须配置为on。
- task_trace、task_time：控制采集算子下发耗时和算子执行耗时的开关。涉及在task_time、op_summary、op_statistic等文件中输出相关耗时数据。配置值：

  - on：开启，默认值，和配置为l1的效果一样。
  - off：关闭。
  - l0：采集算子下发耗时、算子执行耗时数据。与l1相比，由于不采集算子基本信息数据，采集时性能开销较小，可更精准统计相关耗时数据。
  - l1：采集算子下发耗时、算子执行耗时数据以及算子基本信息数据，提供更全面的性能分析数据。

  当训练profiling mode开启即采集训练Profiling数据时，配置task_trace为on的同时training_trace也必须配置为on。

- ge_api：采集动态shape算子在Host调度阶段的耗时数据。取值：
  - off：关闭，默认为off。
  - l0：采集动态shape算子在Host调度主要阶段的耗时数据，可更精准统计相关耗时数据。
  - l1：采集动态shape算子在Host调度阶段更细粒度的耗时数据，提供更全面的性能分析数据。

- hccl：控制通信数据采集开关，可选on或off，默认为off。

  > [!NOTE]说明
  > 此开关后续版本会废弃，请使用task_time开关控制相关数据采集。

- aicpu：采集AICPU算子的详细信息，如：算子执行时间、数据拷贝时间等。取值on/off，默认为off。
- fp_point：指定训练网络迭代轨迹正向算子的开始位置，用于记录前向计算开始时间戳。配置值为指定的正向第一个算子名字。用户可以在训练脚本中，通过tf.io.write_graph将graph保存成.pbtxt文件，并获取文件中的name名称填入；也可直接配置为空，由系统自动识别正向算子的开始位置，例如"fp_point":""。
- bp_point：指定训练网络迭代轨迹反向算子的结束位置，记录后向计算结束时间戳，bp_point和fp_point可以计算出正反向时间。配置值为指定的反向最后一个算子名字。用户可以在训练脚本中，通过tf.io.write_graph将graph保存成.pbtxt文件，并获取文件中的name名称填入；也可直接配置为空，由系统自动识别反向算子的结束位置，例如"bp_point":""。
- aic_metrics：AI Core性能指标采集项，取值如下：

  - ArithmeticUtilization：各种计算类指标占比统计。
  - PipeUtilization：计算单元和搬运单元耗时占比，该项为默认值。
  - Memory：外部内存读写类指令占比。
  - MemoryL0：内部L0内存读写类指令占比。
  - MemoryUB：内部UB内存读写类指令占比。
  - ResourceConflictRatio：流水线队列类指令占比。
  - L2Cache：读写L2 Cache命中次数和缺失后重新分配次数。

    Atlas 推理系列产品不支持该参数。

    Atlas 训练系列产品不支持该参数。

  - MemoryAccess：算子在核上访存的带宽数据量。

    Atlas 推理系列产品不支持该参数。

    Atlas 训练系列产品不支持该参数。

    Ascend 950PR/Ascend 950DT：不支持该参数。

    > [!NOTE]说明
    > 支持自定义需要采集的寄存器，例如："aic_metrics":"**Custom:**_0x49,0x8,0x15,0x1b,0x64,0x10_"。
    > - Custom字段表示自定义类型，配置为具体的寄存器值，范围\[0x1, 0x6E\]。
    > - 配置的寄存器数最多不能超过8个，寄存器通过“,”区分开。
    > - 寄存器的值支持十六进制或十进制。

- l2：控制L2 Cache和TLB页表缓存命中率的开关，可选on或off，默认为off。
  - Atlas 推理系列产品：支持采集L2 Cache的命中率
  - Atlas 训练系列产品：支持采集L2 Cache的命中率
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持采集L2 Cache和TLB页表缓存的命中率；分析AI Core命中L2次数推荐使用aic-metrics=L2Cache。
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持采集L2 Cache和TLB页表缓存的命中率；分析AI Core命中L2次数推荐使用aic-metrics=L2Cache。
  - Ascend 950PR/Ascend 950DT：支持采集L2 Cache和TLB页表缓存的命中率；分析AI Core命中L2次数推荐使用aic-metrics=L2Cache。

- msproftx：控制msproftx用户和上层框架程序输出性能数据的开关，可选on或off，默认值为off。

  需要先在应用程序脚本中添加如下mstx API或msproftx API，推荐使用mstx API。

- runtime_api：控制runtime API性能数据采集开关，可选on或off，默认为off。可采集runtime API性能数据，包括Host与Device之间、Device间的同步异步内存复制时延等。
- sys_hardware_mem_freq：片上内存、QoS传输带宽、LLC三级缓存带宽、加速器带宽、SoC传输带宽、组件内存占用等的采集开关。不同产品的采集内容略有差异，请以实际结果为准。范围\[1,100\]，单位Hz。

  Ascend 950PR/Ascend 950DT，Qos和SoC支持的采集频率最大支持配置10000，其他采集项支持的最大采集频率仍为100，若配置超出范围，其他采集项则按照最大采集频率100进行采集。

  已知在安装有glibc<2.34的环境上采集memory数据，可能触发glibc的一个已知[Bug 19329](https://sourceware.org/bugzilla/show_bug.cgi?id=19329)，通过升级环境的glibc版本可解决此问题。

  对于以下型号，采集任务结束后，不建议用户增大采集频率，否则可能导致SoC传输带宽数据丢失。

  - Atlas A2 训练系列产品/Atlas A2 推理系列产品

  - Atlas A3 训练系列产品/Atlas A3 推理系列产品

- llc_profiling：LLC Profiling采集事件，可以设置为：
  - read：读事件，三级缓存读速率，默认为read。
  - write：写事件，三级缓存写速率。

- sys_io_sampling_freq：NIC、ROCE、UB带宽数据采集频率。范围\[1,100\]，单位Hz。

  - Atlas 推理系列产品：不支持该参数。

  - Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持采集NIC、ROCE

  - Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持采集NIC、ROCE

  - Ascend 950PR/Ascend 950DT：支持采集UB带宽数据

- sys_interconnection_freq：集合通信带宽数据（HCCS）、集合通信硬件加速单元（CCU）带宽数据、SIO数据、PCIe数据、UB带宽数据采集频率以及片间传输带宽信息采集频率。范围\[1,50\]，单位Hz。
  - Atlas 训练系列产品：支持采集HCCS、PCIe数据。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持采集HCCS、PCIe数据、片间传输带宽信息。
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持采集HCCS、PCIe数据、片间传输带宽信息、SIO数据。
  - Ascend 950PR/Ascend 950DT：支持采集PCIe数据、片间传输带宽信息、CCU带宽数据、SIO数据、UB带宽数据。

- dvpp_freq：DVPP采集频率。范围\[1,100\]，单位Hz。
- instr_profiling：AI Core和AI Vector的带宽和延时采集开关。取值on/off，默认为off。
  - Atlas 训练系列产品：不支持该功能。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持该开关，通过instr_profiling_freq控制该功能。
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持该开关，通过instr_profiling_freq控制该功能。
  - Ascend 950PR/Ascend 950DT：支持，但可能会因最后一段指令的统计时间超长导致统计不准确，建议使用msprof op方式采集。

- instr_profiling_freq：AI Core和AI Vector的带宽和延时采集开关，配置了采集频率即开启相关采集能力。范围\[300,30000\]，单位Hz。
  - Atlas 训练系列产品：不支持该功能。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持，但instr_profiling_freq与training_trace、task_trace、hccl、aicpu、fp_point、bp_point、aic_metrics、l2、task_time、runtime_api互斥，无法同时执行。
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持，但instr_profiling_freq与training_trace、task_trace、hccl、aicpu、fp_point、bp_point、aic_metrics、l2、task_time、runtime_api互斥，无法同时执行。
  - Ascend 950PR/Ascend 950DT：不支持该开关，通过instr_profiling控制该功能。

- host_sys：Host侧性能数据采集开关。取值如下，可选其中的一项或多项，选多项时用英文逗号隔开，例如"host_sys": "cpu,mem"。
  - cpu：进程级别的CPU利用率。
  - mem：进程级别的内存利用率。
  - disk：进程级别的磁盘I/O利用率。采集Host侧disk性能数据需要安装第三方开源工具iotop，采集osrt性能数据需要安装第三方开源工具perf和ltrace，其安装方法参见“附录 \> 安装perf、iotop、ltrace工具”。
  - network：系统级别的网络I/O利用率。
  - osrt：进程级别的syscall和pthreadcall。

- host_sys_usage：采集Host侧系统及所有进程的CPU和内存数据。取值包括cpu和mem，可选其中的一项或多项，选多项时用英文逗号隔开。
- host_sys_usage_freq：配置Host侧系统和所有进程CPU、内存数据的采集频率。范围\[1,50\]，默认值50，单位Hz。

> [!NOTE]说明
>
> - 除动态shape场景外的其他场景，fp_point、bp_point为自动配置项，无需用户手动配置。动态shape场景不支持自动配置fp/bp，需要用户手动设置。
> - 在线推理支持task_trace和aicpu，不支持training_trace。

配置示例：

```python
custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/tmp/profiling","training_trace":"on","fp_point":"","bp_point":""}')
```
