# tf.train.batch接口手工迁移

## 迁移原因

TensorFlow不建议使用tf.train.batch接口通过队列形式处理输入数据，建议替换为tf.data.Dataset.batch\(batch_size\)。

如果用户仍然希望在CANN上使用tf.train.batch接口，那么建议num_threads参数配置为1。

## 迁移示例

原始脚本：

```python
(texts, texts_tests, mels, mags, dones) = tf.train.batch([text, texts_test, mel, mag, done], shapes=[(hp.T_x,), (hp.T_x,), ((hp.T_y // hp.r), (hp.n_mels * hp.r)), (hp.T_y, (1 + (hp.n_fft // 2))), ((hp.T_y // hp.r),)], num_threads=2, batch_size=batch_size, capacity=(batch_size * 8), dynamic_pad=False)
```

迁移后脚本：

```python
(texts, texts_tests, mels, mags, dones) = tf.train.batch([text, texts_test, mel, mag, done], shapes=[(hp.T_x,), (hp.T_x,), ((hp.T_y // hp.r), (hp.n_mels * hp.r)), (hp.T_y, (1 + (hp.n_fft // 2))), ((hp.T_y // hp.r),)], num_threads=1, batch_size=batch_size, capacity=(batch_size * 8), dynamic_pad=False)
```
