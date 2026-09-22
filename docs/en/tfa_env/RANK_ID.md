# RANK_ID

## Description

Sets the rank ID of the current process in the collective communication process group in the  TensorFlow  distributed training or inference scenario.

## Example

```bash
export RANK_ID=0
```

## Constraints

The value of this environment variable must be the same as that of the  **rank_id**  field in the rank table file. For details about the rank table configuration file, see "Reference \> Cluster Information Configuration" in  [Huawei Collective Communication Library \(HCCL\)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/hcclug/hcclug_000001.html).

## Applicability

Ascend 950PR&950DT products

Atlas A3 products

Atlas A2 products

Atlas training products

Atlas 300I Duo Inference Card
