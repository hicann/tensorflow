# How Do I Bind Training Processes to CPU Cores?

In multi-device training, to evenly schedule between the host CPU cores and further improve the training performance, you can bind the training processes to respective host CPU cores. The following uses the 8-device scenario as an example.

1. Query the total number of host CPU cores. In this example, the total number of CPU cores is  **96**.

    ![](../figures/find_host_cpu.png)

2. Calculate the number \(_n_\) of host CPU cores allocated to each training process.

    n = Total CPU cores/8 = 12

3. Update the training process startup script. Before starting the training script, use  **taskset -c**  to bind the processes to the specified host CPU cores. See the following example:

    Device 0:

    ```bash
    taskset -c 0-11 python3 /home/test/xxx.py /
    ```

    Device 7:

    ```bash
    taskset -c 84-95 python3 /home/test/xxx.py /
    ```
