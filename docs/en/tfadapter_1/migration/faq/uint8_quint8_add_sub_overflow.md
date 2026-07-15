# What Do I Do If uint8 and quint8 Sum/Subtraction Overflows/Underflows?

## Symptom

The cumulative sum and subtraction results of operators of uint8 and quint8 overflow/underflow, which are different from the theoretical results.

For example, the compute result with the uint8 data type is theoretically  **257**. However,  AI processor  outputs  **255**.

## Possible Cause

On  AI processor, any result that overflows is saturated to the representable maximum. As  **257**  exceeds the maximum representable value \(255\) of a uint8, it is saturated to  **255**.

## Solution

You can choose to scale the value range to avoid overflows/underflows.
