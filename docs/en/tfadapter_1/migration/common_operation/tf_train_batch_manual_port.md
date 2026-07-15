# How Do I Manually Port the tf.train.batch API?

## Porting Cause

In TensorFlow, it is not recommended that the  **tf.train.batch**  API be used to process input data in queue mode. You are advised to use  **tf.data.Dataset.batch\(batch_size\)**  instead.

If you still want to use tf.train.batch on CANN, set  **num_threads**  to  **1**.

## Porting Example

Original script:

```python
(texts, texts_tests, mels, mags, dones) = tf.train.batch([text, texts_test, mel, mag, done], shapes=[(hp.T_x,), (hp.T_x,), ((hp.T_y // hp.r), (hp.n_mels * hp.r)), (hp.T_y, (1 + (hp.n_fft // 2))), ((hp.T_y // hp.r),)], num_threads=2, batch_size=batch_size, capacity=(batch_size * 8), dynamic_pad=False)
```

Script after porting:

```python
(texts, texts_tests, mels, mags, dones) = tf.train.batch([text, texts_test, mel, mag, done], shapes=[(hp.T_x,), (hp.T_x,), ((hp.T_y // hp.r), (hp.n_mels * hp.r)), (hp.T_y, (1 + (hp.n_fft // 2))), ((hp.T_y // hp.r),)], num_threads=1, batch_size=batch_size, capacity=(batch_size * 8), dynamic_pad=False)
```
