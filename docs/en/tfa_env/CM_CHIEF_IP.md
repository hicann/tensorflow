# CM_CHIEF_IP

## Description

In  TensorFlow  distributed training or inference scenarios, you can choose not to use the rank table file. By jointly configuring environment variables  [CM_CHIEF_IP](CM_CHIEF_IP.md),  [CM_CHIEF_PORT](CM_CHIEF_PORT.md),  [CM_CHIEF_DEVICE](CM_CHIEF_DEVICE.md),  [CM_WORKER_SIZE](CM_WORKER_SIZE.md), and  [CM_WORKER_IP](CM_WORKER_IP.md), resource information can be automatically generated to complete the initialization of the collective communication component.

The  **CM_CHIEF_IP**  environment variable is used to configure the listening host IP address of the master node.

The value is a character string in IPv4 or IPv6 format.

## Example

```bash
export CM_CHIEF_IP=192.168.1.1
```

## Constraints

This environment variable cannot be used together with [RANK_TABLE_FILE](RANK_TABLE_FILE.md), [RANK_ID](RANK_ID.md), or [RANK_SIZE](RANK_SIZE.md).

## Applicability

Atlas training product

Atlas A2 training product/Atlas A2 inference product
