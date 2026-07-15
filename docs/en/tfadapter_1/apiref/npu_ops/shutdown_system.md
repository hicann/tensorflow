# shutdown_system

## Description

Disables all devices. This API is used in conjunction with  [initialize_system](initialize_system.md).

## Prototype

```python
def shutdown_system(name = None)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| name | Input | Operator name. |

## Returns

An operator for the user to shut down devices by using  **sess.run\(op\)**.
