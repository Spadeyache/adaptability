# Official-policy baseline

[Experiment index](../experiments.md)

**Status:** Baseline complete; planned sweep stopped by its quality gate.

## Question

Is the published policy reliable enough in this CPU setup to support a sweep?

## Method

Run ten baseline attempts using the official-policy collection configuration
and compare success frequency with the configured minimum of 90%.

## Result

Seven of ten attempts succeeded (70%). The quality gate rejected the planned
sweep.

## Limitation

This is a small baseline in the configured setup. It is separate from the custom
policy's 29,830-attempt dataset and does not establish general policy quality.

## Next step

Resolve the baseline shortfall before launching that official-policy sweep.
Continue to identify the custom and official policy datasets separately.

## Code and evidence

- [Official-policy collection guide](../../robot_adaptation/collection/STOCK_SHORT.md)
- [Configuration](../../configs/failure_collection_stock_short.json)
- [Quality-gate result](../../runs/failure_collection/stock_short/baseline_check.json)
- [Saved attempts](../../runs/failure_collection/stock_short/)
