# Affinity-based Automatic CPU Core Binding Optimization

Compared with x86 servers, Arm servers usually have more CPU cores but weaker single-core performance. Therefore, the kernel load balancing policy is more likely to be triggered when Arm servers are used. This policy enables process migration to reduce pressure on busy processors. Process migration causes process context switching, reduces the cache hit ratio, introduces cross-NUMA memory access, degrading training performance.

Ascend provides an automatic CPU core binding tool for CPU affinity. You can obtain it from the  [source code link](https://gitcode.com/Ascend/mstt/tree/master/profiler/affinity_cpu_bind). The tool requires no project script modifications. Run it directly to automatically bind training processes to CPU cores on ARM servers and improve CPU affinity.
