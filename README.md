# Robot adaptation through deployment experience

We study how robots adapt by continuously forming and refining physical
understanding of their environment through interaction. Sim-to-real transfer is
the starting problem; learning from accumulated deployment experience is the
longer-term goal.

The implementation connects an initial visual mass estimate to a saved
pick-and-lift policy in simulation. An interaction estimator now predicts mass
and friction from completed attempts, with held-condition tests on historical
recordings. A friction-conditioned policy and actual retry driver are implemented;
policy training and controlled evaluation of retry benefits are ongoing.

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
| Learning through interaction | [Interaction-based mass/friction estimation](docs/experiments/interaction_estimate_updates.md) and [policy conditioning](docs/experiments/physics_conditioned_policy.md) | Historical estimator evaluated; policy/retry benefit pending |

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

1. Complete the friction-conditioned policy pilot and requested long training run.
2. Evaluate whether correct physical inputs improve the fixed policy.
3. Compare estimated-input retries against fixed estimates and true parameters;
   retain the held-condition estimator and training-seed limitations.

Use the shared environment with `source scripts/activate_grip.sh`.
Commands, version differences and experiment-recording conventions are in the
[development guide](docs/development.md).
