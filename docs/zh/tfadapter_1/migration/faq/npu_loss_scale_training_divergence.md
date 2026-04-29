# 使用NPU Loss Scale优化器后，训练不收敛

## 问题现象

word2vec网络关闭NPU Loss Scale时，训练收敛；使用NPU Loss Scale优化器后，训练不收敛：

![](../figures/word2vec_faq.png)

## 原因分析

开启溢出检测和Loss Scale打印，发现Loss Scale一直递减到0，说明存在溢出，分析溢出数据，发现Log算子持续溢出。

进一步分析dump数据：

![](../figures/word2vec_npuloss_scale_faq.png)

从上图可以看到，Log算子的输入中存在0，而根据Log函数的曲线，0位置对应无穷大，代表出现溢出，因此问题是由于Log算子输入为0导致一直有溢出，导致动态Loss Scale无法使能。

继续分析Log算子0输入的来源，可以发现，是因为Log算子使用了NZ格式（后面有MatMul运算，要求使用NZ格式，为了提高运算效率，避免整网中出现过多转换算子，会将NZ Format扩散至Log算子），相比原先的格式，转换成NZ格式需要进行TransData补0操作，在TransData之前的数据实际并没有出现0值。

**总结一下：混合精度模式下，Log算子使用了NZ格式，导致输入数据被TransData补0，补齐的0会造成计算过程中出现溢出，但这些溢出数据实际并非有效数据。**

## 解决方案

NPU Loss Scale机制中，默认会对计算过程中影响最终梯度结果的溢出问题进行检查，并在出现浮点异常的迭代放弃本次迭代中计算梯度的更新。

用户可以通过开关enable_overflow_check控制是否进行溢出检查：

```python
FixedLossScaleManager(loss_scale=FLAGS.loss_scale, enable_overflow_check=True)
```

- True：检测到有溢出的迭代，会放弃参数更新，默认是True。
- False：始终更新参数，不检查迭代中是否出现溢出。

word2vec网络，由于溢出数据实际并非有效数据，即使不检测浮点异常，仍然可以保证收敛，因此，可以使用静态Loss Scale并关闭溢出检查解决问题。

除此之外，在实际网络计算中，还有其他场景可以关闭溢出检测，例如：

- 网络脚本中有不影响结果的溢出计算（GNMT）
- 在脚本中已知有溢出风险，后续进行clip操作（Faster-RCNN）
- 可能部分网络中饱和或溢出操作导致溢出对收敛影响有限
