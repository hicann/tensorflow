# CM_WORKER_SIZE

## Description

In  TensorFlow  distributed training or inference scenarios, you can choose not to use the rank table file. By jointly configuring environment variables [CM_CHIEF_IP](CM_CHIEF_IP.md), [CM_CHIEF_PORT](CM_CHIEF_PORT.md), [CM_CHIEF_DEVICE](CM_CHIEF_DEVICE.md), [CM_WORKER_SIZE](CM_WORKER_SIZE.md), and [CM_WORKER_IP](CM_WORKER_IP.md), resource information can be automatically generated to complete the initialization of the collective communication component.

Configures the number of devices in the service communicator.

The value of this environment variable must be an integer ranging from 0 to 32768.

## Example

```bash
export CM_WORKER_SIZE=8
```

## Constraints

This environment variable cannot be used together with  [RANK_TABLE_FILE](RANK_TABLE_FILE.md),  [RANK_ID](RANK_ID.md), or  [RANK_SIZE](RANK_SIZE.md).

## Applicability

Atlas training product

Atlas A2 training product/Atlas A2 inference product
