# 混合计算

## mix_compile_mode

是否开启混合计算模式。

- True：开启混合计算模式。
- False（默认值）：关闭混合计算模式，即为全下沉模式。

计算全下沉模式即所有的计算类算子全部在Device侧执行，混合计算模式作为计算全下沉模式的补充，将部分不可离线编译下沉执行的算子留在前端框架中在线执行，提升NPU支持TensorFlow的适配灵活性。

配置示例：

```python
config = NPURunConfig(mix_compile_mode=True)
```
