# Direct-force prediction gate (revision 5)

**Completed:** the pilot passed, all 230 cubes were collected, and the five-seed
Plain–Oracle prediction gate failed. See [RESULTS_DIRECT.md](RESULTS_DIRECT.md)
for the saved test metrics, seed spread, plots and limitations. No belief model
was trained and no test-based tuning followed.

The controlling request is [DIRECT_FORCE_REQUEST.txt](DIRECT_FORCE_REQUEST.txt).
Previous pusher experiments, source and results are preserved in `versions/v4/`
and earlier versions. The active experiment removes the pusher entirely.

The same independently sampled 200 training and 30 test cube definitions are
used: mass 0.18–0.50 kg, cube friction 0.4–1.0, fixed floor friction 0.3.
Configured combined friction is their average. Per-cube true properties remain
only in `data/cubes.json`; the script cannot access that table.

Each shove applies an external horizontal force at the cube COM, with zero
external torque. Magnitude is uniform 4–7 N, direction uniform over a full turn,
and duration is uniform over the simulation grid 60, 65, …, 100 ms. Duration uses
a separate property-independent RNG stream. The previously sampled force draws,
directions, positions and yaws are preserved. There is no contact-point setting
because there is no pusher. All cube degrees of freedom remain unconstrained.

The saved state has 15 cube coordinates: position, 6D rotation, linear velocity,
and angular velocity. The action has exactly three applied force components;
coast actions are exactly zero. The simulator applies force directly without
dividing by hidden mass. A fixed 100 ms settling period precedes each episode.

Physics runs every 1 ms; model transitions every 5 ms. The raw diagnostic coast
is 1.5 s to cover the strongest allowed shove plus stop confirmation. Saved model
episodes end five steps (25 ms) after stop onset, rounded up to a model boundary.
Stop onset requires 100 ms continuously below 1 mm/s linear and 0.05 rad/s
angular speed. Full raw traces retain the confirmation window. IDs, phase,
length, masks and stop metadata do not enter model inputs.

The ten-cube pilot must achieve at least 90% launches above 2 cm/s and maximum
coast distance below 1 m. A passed pilot and independent data audit authorize
full collection under the user's explicit conditional instruction.

Plain and Oracle are three-hidden-layer, 256-unit ReLU MLPs, with five paired
seeds. Plain receives normalized state+applied force; Oracle additionally gets
true mass and combined friction. Both predict normalized state change.
Training uses episodes 0–7 of training cubes, with episodes 8–9 for checkpoint
selection. State/action normalization uses only the training subset. Each model
runs 10,000 Adam updates, batch size 512, cosine learning rate 0.001→0.0001.
Pair members use identical minibatch streams. Checkpoints select by validation
one-step error and all ten are sealed before the sole test evaluation.

Every comparison reports **push**, **moving coast**, and **all retained steps**.
Moving coast uses observed linear/angular velocity thresholds above. One-step
and recursive ten-step errors are normalized MSE; stopping-position error is
Euclidean XY distance in meters. Forecasts to stopping position always run to
the preplanned release time plus 1.5 s, never the observed trimmed endpoint.
All rollouts use their own predicted intermediate states and preplanned forces.
Metrics weight cubes equally, then episodes, then eligible starts.

The primary gate is moving-coast one-step NMSE: Oracle mean ≤80% of Plain mean
and Oracle mean+sample SD < Plain mean−sample SD across five paired seeds. Test
results cannot change this rule or select another model. No belief model is
part of this step.

Commands, from `/home/ubuntu/faliure`:

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove_gate.collect --stage pilot
PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.shove_gate.audit --stage pilot
PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.shove_gate.probe_v5
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove_gate.collect --stage full
PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.shove_gate.audit --stage full
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.shove_gate.models_gate run --device cuda
```

CUDA requires host GPU access in this environment. The CPU collector does not.
Completed traces are never overwritten. Final evaluation can resume only with
the same sealed checkpoints and data, and training cannot resume after test
evaluation starts.
