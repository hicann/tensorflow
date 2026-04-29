# Hook的index顺序改变导致训练异常

## 问题现象

训练报错：AttributeError:'xxxxHook' object has no attribute 'xxxx'。

![](../figures/hook_index_faq.png)

## 原因分析

迁移工具针对Estimator脚本默认添加了NPUBroadcastHook，而出错的场景中对于hooks这个list，在添加完NPU的hook之后，改变了hooks这个list中的index顺序，所以出现了取hooks\[-1\]进行其他操作报错的问题。

```python
    training_hooks.append(LogTrainRunHook(global_batch_size, hvd_rank, FLAGS.save_checkpoints_steps, num_steps_ignore_xla=25))
...
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=npu_hooks_append(hooks_list=training_hooks))
    train_time_elapsed = time.time() - train_start_time
    train_time_wo_overhead = training_hooks[-1].total_time
    avg_sentences_per_second = num_train_steps * global_batch_size * 1.0 / train_time_elapsed
    ss_sentences_per_second = (training_hooks[-1].count - training_hooks[-1].skipped) * global_batch_size * 1.0 / train_time_wo_overhead
```

## 解决方案

需要修改训练脚本中hook list中的index。针对以上脚本，进行如下修改即可执行成功。

```python
    training_hooks.append(LogTrainRunHook(global_batch_size, hvd_rank, FLAGS.save_checkpoints_steps, num_steps_ignore_xla=25))
...
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=npu_hooks_append(hooks_list=training_hooks))
    train_time_elapsed = time.time() - train_start_time
    train_time_wo_overhead = training_hooks[-2].total_time
    avg_sentences_per_second = num_train_steps * global_batch_size * 1.0 / train_time_elapsed
    ss_sentences_per_second = (training_hooks[-2].count - training_hooks[-2].skipped) * global_batch_size * 1.0 / train_time_wo_overhead
```
