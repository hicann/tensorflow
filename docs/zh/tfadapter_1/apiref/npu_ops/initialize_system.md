# initialize_system

## 功能说明

一般执行训练不需要调用该接口，如果用户统计训练时间时不想包括GE初始化时间，可以使用该接口。使用集合通信接口时，需要先调用该接口进行集合通信初始化。

## 函数原型

```python
def initialize_system(name = None)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| name | 输入 | 算子名称。 |

## 返回值

返回一个op，供用户通过sess.run\(op\)完成GE初始化。

## 约束说明

如果需要调用initialize_system接口，且训练执行时需要使能如下功能时，则必须在initialize_system起session时配置。

- **profiling_mode**：是否开启Profiling功能。
  - True：开启Profiling功能，从profiling_options读取Profiling的采集选项。
  - False：关闭Profiling功能，默认关闭。

- **profiling_options**：Profiling采集选项，取值如下。

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
      >- Custom字段表示自定义类型，配置为具体的寄存器值，范围\[0x1, 0x6E\]。
      >- 配置的寄存器数最多不能超过8个，寄存器通过“,”区分开。
      >- 寄存器的值支持十六进制或十进制。

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

    Atlas A2 训练系列产品/Atlas A2 推理系列产品

    Atlas A3 训练系列产品/Atlas A3 推理系列产品

  - llc_profiling：LLC Profiling采集事件，可以设置为：
    - read：读事件，三级缓存读速率，默认为read。
    - write：写事件，三级缓存写速率。

  - sys_io_sampling_freq：NIC、ROCE、UB带宽数据采集频率。范围\[1,100\]，单位Hz。

    Atlas 推理系列产品：不支持该参数。

    Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持采集NIC、ROCE

    Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持采集NIC、ROCE

    Ascend 950PR/Ascend 950DT：支持采集UB带宽数据

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
    - disk：进程级别的磁盘I/O利用率。采集Host侧disk性能数据需要安装第三方开源工具iotop，采集osrt性能数据需要安装第三方开源工具perf和ltrace。
    - network：系统级别的网络I/O利用率。
    - osrt：进程级别的syscall和pthreadcall。

  - host_sys_usage：采集Host侧系统及所有进程的CPU和内存数据。取值包括cpu和mem，可选其中的一项或多项，选多项时用英文逗号隔开。
  - host_sys_usage_freq：配置Host侧系统和所有进程CPU、内存数据的采集频率。范围\[1,50\]，默认值50，单位Hz。

    > [!NOTE]说明
    > - 除动态shape场景外的其他场景，fp_point、bp_point为自动配置项，无需用户手动配置。动态shape场景不支持自动配置fp/bp，需要用户手动设置。
    > - 在线推理支持task_trace和aicpu，不支持training_trace。

- **enable_dump**：是否开启Data Dump功能。

  - True：开启Data Dump功能，从dump_path读取Dump文件保存路径，dump_path为None时会产生异常。
  - False（默认值）：关闭Data Dump功能。

    > [!NOTE]说明
    > - 不能同时开启Data Dump与溢出数据采集功能，即不同时将enable_dump和enable_dump_debug参数配置为“True”。
    > - 若“enable_dump/enable_dump_debug（二选一）”参数配置为“True”，同时“enable_exception_dump”配置为了“1”（即开启普通异常算子dump）。此时，针对动态shape网络，仅“enable_exception_dump”生效；针对静态shape网络，“enable_exception_dump”与“enable_dump/enable_dump_debug（二选一）”都生效。

- **dump_path**：Dump文件保存路径。enable_dump或enable_dump_debug为true时，该参数必须配置。

    该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限，支持配置绝对路径或相对路径（相对执行命令行时的当前路径）。

  - 绝对路径配置以“/”开头，例如：/home/test/output。
  - 相对路径配置直接以目录名开始，例如：output。

- **dump_step**：指定采集哪些迭代的Data Dump数据。默认值：None，表示所有迭代都会产生dump数据。

    多个迭代用“|”分割，例如：0|5|10；也可以用"-"指定迭代范围，例如：0|3-5|10。

- **dump_mode**：Data Dump模式，用于指定dump算子输入还是输出数据。取值如下：

  - input：仅Dump算子输入数据
  - output：仅Dump算子输出数据，默认为output。
  - all：Dump算子输入和输出数据

    > [!NOTE]说明
    > 配置为all时，由于部分算子在执行过程中会修改输入数据，例如集合通信类算子HcomAllGather、HcomAllReduce等，因此系统在进行dump时，会在算子执行前dump算子输入，在算子执行后dump算子输出，这样，针对同一个算子，算子输入、输出的dump数据是分开落盘，会出现多个dump文件，在解析dump文件后，用户可通过文件内容判断是输入还是输出。

- **enable_dump_debug**：是否开启溢出检测功能，默认值：False。
  - True：开启溢出检测功能，从dump_path读取Dump文件保存路径，dump_path为None时会产生异常。
  - False：关闭溢出检测功能。

    > [!NOTE]说明
    >- 不能同时开启Data Dump与溢出数据采集功能，即不同时将enable_dump和enable_dump_debug参数配置为“True”。
    >- 若“enable_dump/enable_dump_debug（二选一）”参数配置为“True”，同时“enable_exception_dump”配置为了“1”（即开启普通异常算子dump）。此时，针对动态shape网络，仅“enable_exception_dump”生效；针对静态shape网络，“enable_exception_dump”与“enable_dump/enable_dump_debug（二选一）”都生效。

- **dump_debug_mode**：溢出检测模式，取值如下：
  - aicore_overflow：AI Core算子溢出检测，检测在算子输入数据正常的情况下，输出是否不正常的极大值（如float16下65500,38400,51200这些值）。一旦检测出这类问题，需要根据网络实际需求和算子逻辑来分析溢出原因并修改算子实现。
  - atomic_overflow：Atomic Add溢出检测模式，在AI Core计算完，由UB搬运到OUT时，产生的Atomic Add溢出问题。
  - all：同时进行AI Core算子溢出检测和Atomic Add溢出检测。默认值为“all”。

    > [!NOTE]说明
    > 针对Ascend 950PR/Ascend 950DT，Atlas A3 训练系列产品/Atlas A3 推理系列产品，Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持配置为默认值“all”。

- **precision_mode**：算子精度模式，配置要求为string类型。

  - allow_fp32_to_fp16：
    - 对于矩阵类算子：
      - 如果原图中算子精度为float32，优先降低精度到float16，如果AI Core中算子不支持float16，则继续选择float32，如果AI Core中算子不支持float32，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。
      - 如果原图中算子精度为bfloat16，则优先使用原图精度bfloat16，如果AI Core中算子不支持bfloat16，则选择float32，如果AI Core中算子不支持float32，则直接降低精度到float16；如果AI Core中算子不支持float16，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。

    - 对于矢量类算子，优先保持原图精度：
      - 如果原图中算子精度为float32，则优先使用原图精度float32，如果AI Core中算子不支持float32，则直接降低精度到float16；如果AI Core中算子不支持float16，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。
      - 如果原图中算子精度为bfloat16，则优先使用原图精度bfloat16，如果AI Core中算子不支持bfloat16，则选择float32，如果AI Core中算子不支持float32，则直接降低精度到float16；如果AI Core中算子不支持float16，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。

  - force_fp16：

    算子同时支持float16、bfloat16和float32数据类型时，强制选择float16数据类型。**此参数仅适用于在线推理场景。**

  - force_fp32/cube_fp16in_fp32out：

    配置为force_fp32或cube_fp16in_fp32out，效果等同，该选项用来表示AI Core中该算子既支持float32又支持float16数据类型时，系统内部都会根据算子类型不同，选择不同的处理方式。cube_fp16in_fp32out为新版本中新增的，对于矩阵计算类算子，该选项语义更清晰。

    - 对于矩阵计算类算子，系统内部会按算子实现的支持情况处理：
            1. 优先选择输入数据类型为float16且输出数据类型为float32；
            2. 如果1中的场景不支持，则选择输入数据类型为float32且输出数据类型为float32；
            3. 如果2中的场景不支持，则选择输入数据类型为float16且输出数据类型为float16；
            4. 如果3中的场景不支持，则报错。

    - 对于矢量计算类算子，表示原图中算子精度为float16或bfloat16，强制选择float32。

      如果原图中存在部分算子，在AI Core中该算子的实现不支持float32，比如某算子仅支持float16类型，则该参数不生效，仍然使用支持的float16；如果在AI Core中该算子的实现不支持float32，且又配置了黑名单（precision_reduce = false），则会使用float32的AI CPU算子；如果AI CPU算子也不支持，则执行报错。

  - must_keep_origin_dtype：

    保持原图精度。

    - 如果原图中某算子精度为float16，AI Core中该算子的实现不支持float16、仅支持float32和bfloat16，则系统内部会自动采用高精度float32。
    - 如果原图中某算子精度为float16，AI Core中该算子的实现不支持float16、仅支持bfloat16，则会使用float16的AI CPU算子；如果AI CPU算子也不支持，则执行报错。
    - 如果原图中某算子精度为float32，AI Core中该算子的实现不支持float32类型、仅支持float16类型，则会使用float32的AI CPU算子；如果AI CPU算子也不支持，则执行报错。

  - allow_mix_precision_fp16/allow_mix_precision：

    配置为allow_mix_precision或allow_mix_precision_fp16，效果等同，均表示使用混合精度float16、bfloat16和float32数据类型来处理神经网络的过程。allow_mix_precision_fp16为新版本中新增的，语义更清晰，便于理解。

    针对原始模型中float32和bfloat16数据类型的算子，按照内置的优化策略，自动将部分float32和bfloat16的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

  - allow_mix_precision_bf16：

    表示使用混合精度bfloat16和float32数据类型来处理神经网络的过程。针对原始模型中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到bfloat16，从而在精度损失很小的情况下提升系统性能并减少内存使用；如果AI Core中算子不支持bfloat16和float32，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。

    说明：仅Ascend 950PR/Ascend 950DT，Atlas A3 训练系列产品/Atlas A3 推理系列产品，Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持此配置。

  - allow_fp32_to_bf16：

    - 如果原图中算子精度为float32，则优先使用原图精度float32，如果AI Core中算子不支持float32，则降低精度到bfloat16；如果AI Core中算子不支持bfloat16，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。
    - 如果原图中算子精度为bfloat16，则优先使用原图精度bfloat16，如果AI Core中算子不支持bfloat16，则选择float32，如果AI Core中算子不支持float32，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。

      说明：Ascend 950PR/Ascend 950DT，Atlas A3 训练系列产品/Atlas A3 推理系列产品，Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持此配置。

    针对Atlas 训练系列产品，默认配置项为“allow_fp32_to_fp16”。

    针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，默认配置项为“must_keep_origin_dtype”。

- **graph_run_mode**：图执行模式，取值：
  - 0：在线推理场景下，请配置为0。
  - 1（默认值）：训练场景下，请配置为1。

- **op_debug_level**：算子debug功能开关，取值：
  - 0：不开启算子debug功能。
  - 1：开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中生成TBE指令映射文件（算子cce文件\*.cce、python-cce映射文件\*_loc.json、.o和.json文件），用于后续工具进行AI Core Error问题定位。

    注意：Ascend 950PR/Ascend 950DT不会生成TBE指定映射文件。

  - 2：开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中生成TBE指令映射文件（算子cce文件\*.cce、python-cce映射文件\*_loc.json、.o和.json文件），并关闭ccec编译器的编译优化开关且打开ccec调试功能（ccec编译器选项设置为-O0-g），用于后续工具进行AI Core Error问题定位。

    注意：Ascend 950PR/Ascend 950DT不会生成TBE指定映射文件。

  - 3：不开启算子debug功能，且在训练脚本执行目录下的kernel_meta文件夹中保留.o和.json文件。
  - 4：不开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中**保留**.o（算子二进制文件）和.json文件（算子描述文件），生成TBE指令映射文件（算子cce文件\*.cce）和UB融合计算描述文件（\{$kernel_name\}_compute.json）。

    注意：Ascend 950PR/Ascend 950DT不会生成TBE指定映射文件和UB融合计算描述文件。

    注意:
  - 当该参数取值为0时，同时又配置了“op_debug_config”参数，则训练执行时，仍会在当前执行路径下生成算子编译目录kernel_meta，目录中生成的内容以“op_debug_config”配置为准。
  - 训练执行时，建议配置为0或3。如果需要进行问题定位，再选择调试开关选项1和2，是因为加入了调试功能后，会导致网络性能下降。
  - 配置为2（即开启ccec编译选项）的场景下，不能与“op_debug_config”中的“oom”同时使用，会导致AI Core Error报错，报错信息示例如下。

    ```text
    ...there is an aivec error exception, core id is 49, error code = 0x4 ...
    ```

  - 配置为2（即开启ccec编译选项）的场景下，会增大算子Kernel（\*.o文件）的大小。动态shape场景下，由于算子编译时会遍历可能存在的所有场景，最终可能会导致由于算子Kernel文件过大而无法进行编译的情况，此种场景下，建议不要配置为2。
    由于算子kernel文件过大而无法编译的日志显示如下：

    ```text
    message:link error ld.lld: error: InputSection too large for range extension thunk ./kernel_meta_xxxxx.o:(xxxx)
    ```

  - 当该参数取值不为0时，可通过“debug_dir”参数指定调试相关过程文件的存放路径。
  - 该参数取值为0，同时设置了NPU_COLLECT_PATH环境变量的场景，执行命令当前路径**下仍旧会生成**算子编译目录kernel_meta；若设置了ASCEND_WORK_PATH环境变量，则在该环境变量指定路径下生成kernel_meta。关于环境变量的详细说明，可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》。
  - debug功能开关打开场景下，若模型中含有如下通算融合算子，算子编译目录kernel_meta中，不会生成下述算子的\*.o、\*.json、\*.cce文件。

    ```text
    MatMulAllReduce
    MatMulAllReduceAddRmsNorm
    AllGatherMatMul
    MatMulReduceScatter
    AlltoAllAllGatherBatchMatMul
    BatchMatMulReduceScatterAlltoAll
    ```

- **enable_exception_dump**：是否dump异常算子数据。

  - 0：关闭异常算子数据dump功能。
  - 1：开启普通ExceptionDump，dump异常算子的输入输出数据、tensor描述信息（shape、dtype、format等）以及workspace信息。

    dump数据存储路径优先级为：环境变量NPU_COLLECT_PATH \> 环境变量ASCEND_WORK_PATH \> 默认路径（当前脚本执行路径下的extra-info目录）。

  - 2（默认值）：开启LiteExceptionDump，dump异常算子的输入输出数据、workspace信息、Tiling信息等，导出的数据用于分析AI Core Error问题。

    dump数据存储路径优先级为：环境变量ASCEND_WORK_PATH \> 默认路径（指当前脚本执行路径下的extra-info/data-dump/<device_id\>目录）。

    > [!NOTE]说明
    > 若配置了环境变量NPU_COLLECT_PATH，不论配置项“enable_exception_dump”的取值如何，都按照“1：普通ExceptionDump”进行异常算子数据dump，且dump数据存储在环境变量NPU_COLLECT_PATH的指定目录下。

- **op_select_implmode**：NPU内置算子有高精度和高性能实现方式，用户可以通过该参数配置模型编译时选择哪种算子。取值包括：
  - high_precision：表示算子选择高精度实现。高精度实现算子是指在fp16输入的情况下，通过泰勒展开/牛顿迭代等手段进一步提升算子的精度。
  - high_performance：表示算子选择高性能实现。高性能实现算子是指在fp16输入的情况下，不影响网络精度前提的最优性能实现。默认为high_performance。

- **optypelist_for_implmode**：列举算子optype的列表，该列表中的算子使用op_select_implmode参数指定的模式，当前支持的算子为Pooling、SoftmaxV2、LRN、ROIAlign，多个算子以英文逗号分隔。

    该参数需要与op_select_implmode参数配合使用，例如：

    op_select_implmode配置为high_precision。

    optypelist_for_implmode配置为Pooling。

    默认值为空，代表不使能此配置。

## 调用示例

如果在sess.run或者estimator.train之前调用get_local_rank_id/get_rank_size/get_rank_id等HCCL接口，需要先另起session执行initialize_system，进行集合通信初始化，然后在训练结束后执行shutdown_system，同时关闭session。

```python
import tensorflow as tf
from npu_bridge.npu_init import *
npu_int = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()

config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

init_sess = tf.Session(config=config)
init_sess.run(npu_int)

#调用HCCL接口...
#执行训练...

init_sess.run(npu_shutdown)
init_sess.close()
```

或者：

```python
import tensorflow as tf
from npu_bridge.npu_init import *
npu_init = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()

config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

with tf.Session(config=config) as sess:
    sess.run(npu_init)
    #调用HCCL接口...
    #执行训练...
    sess.run(npu_shutdown)
```
