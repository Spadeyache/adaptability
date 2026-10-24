# Initial visual mass

[Experiment index](../experiments.md)

**Status:** Completed controlled baseline.

## Question

Can an initial image provide a useful physical estimate before the robot acts?

## Method

Render 384 RGB images of 96 cube sizes at fixed density, material, friction and
camera calibration. Train a small CNN on image pixels to predict mass. Keep all
views of an object in the same split and select the checkpoint by validation loss.

## Result

The test set contains 12 held-out sizes and 48 images.

| Measure | Result |
| --- | ---: |
| Test mass MAE | 2.76 g |
| Test mass RMSE | 4.43 g |
| Constant training-mean baseline MAE | 106.37 g |

The selected checkpoint was at epoch 64 of 80.

## Limitation

This is size-to-mass interpolation under a fixed-density prior. It does not
establish mass inference for identical-looking objects with different densities,
new materials or new camera setups. Estimation accuracy alone does not establish
improved task performance.

## Next step

Evaluate the estimate through the [policy integration](estimator_policy_integration.md),
then investigate corrections when interaction contradicts the initial prior.

## Code and evidence

- [Estimator guide and commands](../../robot_adaptation/estimators/visual_mass/README.md)
- [Run](../../runs/visual_mass/initial_rgb_mass/)
- [Dataset and splits](../../runs/visual_mass/initial_rgb_mass/dataset/dataset.json)
- [Metrics](../../runs/visual_mass/initial_rgb_mass/model/metrics.json)
- [Predictions](../../runs/visual_mass/initial_rgb_mass/model/predictions.csv)
- [Result plot](../../runs/visual_mass/initial_rgb_mass/model/predictions.png)
