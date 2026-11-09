# Mass belief across five lifts

[Experiment index](../experiments.md)

## Question

Can a robot estimate a cube's mass more accurately by remembering evidence from previous attempts on the same cube?

## Method

The historical sweep recorded actual mass and both friction coefficients separately from the 40 g guess supplied to the frozen policy. Of its 600 attempts, 30 matched the nominal geometry and mass-only sweep used for training. These supplemented 100 freshly simulated training episodes with randomized mass. Validation and filter calibration used separate sets of 24 and 40 episodes.

The estimator predicts mass in grams from up to 25 steps of joint positions, velocities, finger compression and applied commands. We replayed identical observations from 12 independently seeded sequences of five attempts on a 400 g cube with friction 0.4. We compared the raw terminal estimate, a filter reset at each attempt and the same filter carried across all five attempts. All arms start each attempt with a fresh estimator history.

## Result

Fifth-trial mean absolute errors were 102.8 g for the raw estimator, 83.5 g for the reset filter and 37.4 g for the carried filter. The paired advantage of carrying the filter was 46.2 g against resetting it, with a 95% sequence-bootstrap interval of 14.6–73.7 g. The raw estimator settled within 40 g through the end of 10 of 60 attempts. The median settling step among those ten was 79 of 100. In closed-loop lifts the original policy succeeded on 18 of 60 starts with belief feedback and 23 of 60 with true mass. [Results and paired intervals](../../runs/mass_belief/nov17_v1/RESULTS.md)

## Limitation and next step

The evidence covers one heavy mass at fixed friction. All 60 fixed-input probes failed to hold the object at the final step. The filter's variance follows a fixed measurement schedule and declines by construction. Evaluating interval coverage across additional held-out masses would test how well variance describes estimation error.

## Code and evidence

[Reproduction instructions](../../experiments/mass_belief/README.md) · [Configuration](../../configs/mass_belief.json) · [Across-trial plot](../../runs/mass_belief/nov17_v1/mass_across_trials.pdf) · [Within-trial plot](../../runs/mass_belief/nov17_v1/within_trial.png) · [Trial data](../../runs/mass_belief/nov17_v1/trials.csv) · [Run manifest](../../runs/mass_belief/nov17_v1/manifest.json)
