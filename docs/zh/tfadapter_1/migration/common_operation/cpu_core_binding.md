# 绑定训练进程到指定CPU

多P训练场景，为了使Host CPU调度均匀，从而进一步提高训练性能，用户可以参考如下步骤将训练进程绑定到指定的CPU上，用于平均分配Host CPU调度数，下面以8P举例说明：

1. 查询HOST CPU个数，例如：Total CPU =96。

    ![](../figures/find_host_cpu.png)

2. 计算每个训练进程分配的Host CPU调度数n。

    n = Total CPU / 8 = 12。

3. 修改训练进程启动脚本，在启动训练脚本前，使用“taskset -c ”绑定进程到指定的Host CPU。例如：

    Device0：

    ```bash
    taskset -c 0-11 python3 /home/test/xxx.py /
    ```

    Device7：

    ```bash
    taskset -c 84-95 python3 /home/test/xxx.py /
    ```
