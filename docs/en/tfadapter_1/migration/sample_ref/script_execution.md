# Script Execution

## Preparing a Dataset

Prepare a dataset and upload it to a directory in the operating environment, for example,  **/home/data/resnet50/imagenet**.

## Preparing the Ranktable File

For details about the rank table file example and description, see  Reference \> Cluster Information Configuration  in  [Huawei Collective Communication Library \(HCCL\)](https://www.hiascend.com/document/detail/en/canncommercial/900/API/hcclug/hcclug_000001.html).

## Setting Environment Variables

For details about the environment variable configuration, see  [Training with a Single Device](../model_training/single_device_training.md).

## Running the Command

```bash
python3 /home/official/r1/resnet/imagenet_main.py --batch_size=32 --hooks=ExamplesPerSecondHook --data_dir=/home/data/resnet50/imagenet
```
