# RANK_TABLE_FILE

## Description

In TensorFlow distributed training or inference scenarios, this environment variable is used to specify the rank table resource configuration file of the  AI processor  involved in collective communication, including the path and name of the rank table file.

For details about the rank table configuration file, see "Reference \> Cluster Information Configuration" in  [Huawei Collective Communication Library \(HCCL\)](https://www.hiascend.com/document/detail/en/canncommercial/900/API/hcclug/hcclug_000001.html).

## Example

```bash
export RANK_TABLE_FILE=/home/test/ranktable.json
```

## Constraints

None

## Applicability

Ascend 950PR/Ascend 950DT

Atlas A3 training product/Atlas A3 inference product

Atlas A2 training product/Atlas A2 inference product

Atlas training product

Atlas 300I Duo Inference Card
