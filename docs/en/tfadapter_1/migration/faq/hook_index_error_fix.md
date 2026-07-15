# How Do I Fix Training Errors Caused by Hook Index Changes?

## Symptom

An error is reported during training: AttributeError:'xxxxHook' object has no attribute 'xxxx'.

![](../figures/hook_index_faq.png)

## Possible Cause

The porting tool will add  **NPUBroadcastHook**  to an Estimator-based script. In this case, the newly added NPU hook has changed the hook indexes in the hooks list, resulting in this  **hook\[-1\]**  error.

```python
    training_hooks.append(LogTrainRunHook(global_batch_size, hvd_rank, FLAGS.save_checkpoints_steps, num_steps_ignore_xla=25))
    ...
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=npu_hooks_append(hooks_list=training_hooks))
    train_time_elapsed = time.time() - train_start_time
    train_time_wo_overhead = training_hooks[-1].total_time
    avg_sentences_per_second = num_train_steps * global_batch_size * 1.0 / train_time_elapsed
    ss_sentences_per_second = (training_hooks[-1].count - training_hooks[-1].skipped) * global_batch_size * 1.0 / train_time_wo_overhead
```

## Solution

Correct the hook index in the hooks list in the training script as follows:

```python
    training_hooks.append(LogTrainRunHook(global_batch_size, hvd_rank, FLAGS.save_checkpoints_steps, num_steps_ignore_xla=25))
    ...
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=npu_hooks_append(hooks_list=training_hooks))
    train_time_elapsed = time.time() - train_start_time
    train_time_wo_overhead = training_hooks[-2].total_time
    avg_sentences_per_second = num_train_steps * global_batch_size * 1.0 / train_time_elapsed
    ss_sentences_per_second = (training_hooks[-2].count - training_hooks[-2].skipped) * global_batch_size * 1.0 / train_time_wo_overhead
```
