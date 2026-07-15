# RANK_ID

## Description

Sets the rank ID of the current process in the collective communication process group in the  TensorFlow  distributed training or inference scenario.

## Example

```bash
export RANK_ID=0
```

## Constraints

The value of this environment variable must be the same as that of the  **rank_id**  field in the rank table file. For details about the rank table configuration file, see "Reference \> Cluster Information Configuration" in  [Huawei Collective Communication Library \(HCCL\)](https://www.hiascend.com/document/detail/en/canncommercial/900/API/hcclug/hcclug_000001.html).

## Applicability

Ascend 950PR/Ascend 950DT

Atlas A3 training product/Atlas A3 inference product

Atlas A2 training product/Atlas A2 inference product

Atlas training product

Atlas 300I Duo Inference Card
