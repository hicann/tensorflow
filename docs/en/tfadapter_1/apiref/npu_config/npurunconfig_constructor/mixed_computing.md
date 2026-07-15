# Mixed Computing

## mix_compile_mode

Whether to enable mixed computing.

- True: enabled.
- False (default): disabled (full offload mode)

In full offload mode, all compute operators are offloaded to the device. As a supplement to the full offload mode, mixed computing allows certain operators to be executed online within the frontend framework, improving the AI processor's adaptability to TensorFlow.

Example:

```python
config = NPURunConfig(mix_compile_mode=True)
```
