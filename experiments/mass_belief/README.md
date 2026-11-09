# Mass belief over repeated lifts

Train a scalar mass estimator in ManiSkill3, preserve its weights, and follow the same heavy cube for five trials. The policy is the frozen scalar-weight policy that recorded the 600-episode sweep. All masses for inference and plotting are in grams. Its weight input is converted to newtons using the scene gravity.

From the repository root:

```bash
source scripts/activate_grip.sh
python -m experiments.mass_belief --output runs/mass_belief/new_run
```

The output directory must be new. To inspect individual stages, use the module commands in `collect.py`, `train.py`, `evaluate.py`, and `report.py`. All settings for data collection and the five-trial evaluation are in `configs/mass_belief.json`. Collection and inference run on CPU.

The primary comparison replays identical fixed-40-g-input trials. At each step the estimator sees up to 25 post-action proprioceptive measurements paired with the commands that produced them. A is the terminal raw estimate with a fresh history. B carries the Kalman mean and variance across trials. The reset-filter control follows the same updates as B but restarts the prior every trial. A fresh history is used by all arms. The side check feeds the carried belief into an executing policy and compares against a matched true-mass policy. It uses fresh-scene resets and verifies equal initial observations. No-contact episodes, misses and drops are retained.

The filter uses a uniform-training-range prior. Each non-overlapping 25-step measurement is calibrated on a disjoint simulation split by fitting a linear relation between true mass and the estimator output. The per-window residual variance is inflated by the number of updates per episode to reduce overconfidence from correlated observations. Q is zero for a cube with constant mass. The held-out coverage diagnostic tests the resulting uncertainty model.

Generated evidence includes raw per-episode NPZ files, step and trial CSVs, the frozen estimator, calibration parameters, a two-panel across-trial figure, a within-trial settling figure and a shareable RESULTS.md. Intervals resample independent five-trial sequences with replacement. The seeded starts and paired comparisons are conditional on the designated mass and simulator setting.

## Switch the cube after five attempts

Continue each saved sequence from its fifth-trial posterior with five fresh attempts on a lighter, higher-friction cube. The original fixed-input probe collects all trajectories. Every method receives the same motions. A fourth comparison resets the filter once at the known switch time and then carries it. This measures the potential benefit of being told when the cube changes.

```bash
python -m experiments.mass_belief.switch runs/mass_belief/nov17_v1 runs/mass_belief/switch_80g_mu1 --mass-g 80 --friction 1.0
```

The output directory contains the ten-attempt figure, post-switch trajectories and trial metrics. Shaded bands resample whole sequences. Both cube properties change together, so the comparison measures their combined effect on mass estimation.
