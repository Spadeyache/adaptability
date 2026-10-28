# Robot adaptation through deployment experience

We study how robots adapt by continuously forming and refining physical
understanding of their environment through interaction. Sim-to-real transfer is
the starting problem; learning from accumulated deployment experience is the
longer-term goal.

The implementation connects an initial visual mass estimate to a saved
pick-and-lift policy in simulation. An interaction estimator now predicts mass
and friction from completed attempts, with held-condition tests on historical
recordings. Policy training ended at the user's request, and final controlled
evaluation supports accurate-input and adaptive-retry benefits in simulation.
Estimator transfer to the final policy supports mass recovery; friction remains
inconclusive.

## Start here

| If you want to understand… | Open |
| --- | --- |
| What we tried, what we found, and what is next | [Experiment index](docs/experiments.md) |
| Where code lives and how to run it | [Development guide](docs/development.md) |
| The research presentation and its local assets | [Presentation guide](presentations/README.md) |

## Research map

| Area | What we have | Status |
| --- | --- | --- |
| Movement capability | RL training, saved policies, grasp-and-lift simulation | Foundation |
| Initial physical understanding | [Visual mass estimator](docs/experiments/initial_visual_mass.md) and [estimator–policy integration](docs/experiments/estimator_policy_integration.md) | Implemented in simulation |
| Understanding failures | [Behavior grouping](docs/experiments/failure_behavior_grouping.md), [parameter sweeps](docs/experiments/physical_parameter_sweeps.md), and [intervention responses](docs/experiments/failure_intervention_responses.md) | Exploratory results |
| Learning through interaction | [Interaction-based mass/friction estimation](docs/experiments/interaction_estimate_updates.md) and [policy conditioning](docs/experiments/physics_conditioned_policy.md) | Accurate-input and adaptive-retry benefits supported; new-policy estimator transfer supports mass, with friction inconclusive |

The visual estimator reached **2.76 g test MAE** on fixed-density cubes. The combined
runner completed three configured pick-and-lift examples. Behavior grouping
found two broad groups; neither that grouping nor the sweep plots established
reliable diagnosis of an individual failure. Each result page explains its scope.

## Workspace map

```text
robot_adaptation/   Reusable policy, estimator, simulation and analysis code
experiments/       Experiment-specific drivers
configs/           Experiment settings
runs/              Generated datasets, models, logs and recordings
docs/              Research index, result pages and development guide
presentations/     Slide sources, presentation figures and deck snapshots
tests/             Code verification
scripts/           Environment setup and convenience launchers
work/              Temporary previews and communication drafts
```

Active code lives in `robot_adaptation/` and `experiments/`. Older top-level
folders retain compatibility commands and historical results; the
[development guide](docs/development.md#legacy-entrypoints-and-historical-artifacts)
maps them to their active locations.

## Current next steps

The [final evaluation](runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/final_summary.md)
is complete. The user ended training after 19,489 rollouts; the saved policy
contains 19,000 cycles, and the original 50,000-cycle budget was not completed.
Accurate inputs and adaptive retries help on the tested simulated tasks.
Reliable friction recovery under the new policy remains unestablished, and more
attempts do not consistently reduce estimation errors. The next research step
is to isolate useful prior estimates, mass updates and friction updates before
attributing retry gains to a recovered physical cause.

Use the shared environment with `source scripts/activate_grip.sh`.
Commands, version differences and experiment-recording conventions are in the
[development guide](docs/development.md).
