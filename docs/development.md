# Architecture and development

[Project overview](../README.md) · [Experiment index](experiments.md)

Use the experiment index for research questions and results. This guide covers
code locations, commands, interfaces and recording conventions.

## Code layout

```text
robot_adaptation/
  simulation/                  environments, controllers, task variants
  policy/                      model, training, inspection, frozen v1 inference
  estimators/visual_mass/       image generation, model, training, prediction
  collection/                  physical sweeps, recorder, schemas, source provenance
  analysis/failure_discovery/   grouping and intervention-response reports
  analysis/plots/               plot plans and report generation
  runtime/                     initial image to estimator to policy rollout
experiments/crest/              CREST-specific search, repair and experiment driver
configs/                       experiment settings
runs/                          generated outputs and source snapshots
docs/                          project-level explanations and experiment index
docs/experiments/              one result page per research study
presentations/                 slide sources and local presentation artifacts
work/                          temporary previews and communication drafts
```

A model's training and inference belong together. Shared simulation code is
independent of collection commands. Analysis consumes saved records. Experiment
drivers compose reusable components and own their experimental choices.

The image generator is local to the visual-mass baseline because its fixed-density
scene is part of that estimator experiment. General robot episode collection lives
in `collection/`.

## Where to work

| Component | Active location | Responsibility |
| --- | --- | --- |
| Policy | [policy guide](../robot_adaptation/policy/README.md) | Model, PPO training, checkpoint loading, inspection and examples. |
| Estimators | [visual-mass guide](../robot_adaptation/estimators/visual_mass/README.md) | Image generation, initial mass prediction, training and inference. |
| Simulation | [simulation guide](../robot_adaptation/simulation/README.md) | Physics, sensors, controllers and distinct task variants. |
| Collection | [collection guide](../robot_adaptation/collection/README.md) | Physical sweeps, recording, schemas and provenance. |
| Failure discovery | [analysis guide](../robot_adaptation/analysis/failure_discovery/README.md) | Behavior grouping and matched intervention responses. |
| Plots | [plot guide](../robot_adaptation/analysis/plots/README.md) | Editable comparison plans, plots and reports. |
| Runtime | [runtime guide](../robot_adaptation/runtime/README.md) | Initial RGB → estimated mass → policy actions in simulation. |
| CREST | [experiment guide](../experiments/crest/README.md) | Context selection and controller-repair experiment. |

## Commands

Start from the repository root using the existing shared environment:

```bash
source scripts/activate_grip.sh
```

| Action | Command |
| --- | --- |
| Prepare policy training without starting it | `python -m robot_adaptation.policy.train --prepare-only` |
| Train policy | `python -m robot_adaptation.policy.train` |
| Inspect saved policy | `python -m robot_adaptation.policy.inspect` |
| Generate images and train mass estimator | `python -m robot_adaptation.estimators.visual_mass` |
| Run estimator with policy in simulation | `python -m robot_adaptation.runtime --record-video` |
| Preview dense collection | `bash scripts/collect_varied.sh --plan-only` |
| Collect dense physical sweeps | `bash scripts/collect_varied.sh` |
| Preview official-policy collection | `bash scripts/collect_stock_short.sh --plan-only` |
| Group saved failure behavior | `python -m robot_adaptation.analysis.failure_discovery.analyze` |
| Compare saved intervention responses | `python -m robot_adaptation.analysis.failure_discovery.response_groups` |
| Generate planned plots | `python -m robot_adaptation.analysis.plots.plot` |
| Check the plot plan | `python -m robot_adaptation.analysis.plots.plot --check` |
| Preview CREST | `python -m experiments.crest --plan-only` |

Each command supports its existing options; consult its component guide or
`--help`. Training learns weights. Collection, inspection and analysis use saved
policies or records. Policy settings live in `robot_adaptation/policy/train.py`;
other experiment settings are under `configs/` and recorded with each run.

## Dependencies and interfaces

- `policy/model.py` defines the current v2 learner. `policy/frozen_v1.py` defines
  the older inference interface and strict checkpoint loader for collection.
  Matching input/output counts alone does not make these architectures interchangeable.
- `simulation/training_lift.py` and `simulation/collection_lift.py` remain
  separate task variants. Training's mass/friction relationship differs from
  independent-physics collection; moving code does not merge those assumptions.
- `simulation/grip_controller.py` is the shared controller implementation.
- Estimator training and prediction receive RGB pixels. Actual simulator masses
  are supervised labels, not prediction inputs. The initial visual baseline
  needs size correlated with mass and does not establish arbitrary-density inference.
- Estimator outputs are kilograms. The current policy's supplied-weight input
  is in newtons. The runtime uses simulator gravity for that conversion, preserves the observation
  ordering and architecture version, and checks the supplied weight each step.
- Episode schemas and input/label separation are owned by `collection/`.
  Simulator-only labels do not automatically become deployment observations.

Use the existing `.venv`, `scripts/activate_grip.sh`, and root
`requirements-lock.txt`. Analysis and estimator code share that environment;
there are no separate, conflicting environments for individual experiments.

## Adding an experiment

1. Create a result page using [the template](experiments/_template.md) and link
   it from [the experiment index](experiments.md). Mark proposed work explicitly.
2. Put reusable changes in the relevant component; keep experimental choices in
   a driver under `experiments/` or the component's existing command.
3. Save the configuration, seed, input dataset/checkpoint, and relevant source
   identity with a new run under `runs/<experiment>/<run>/`.
4. Update the study page with its method, result, limitation, next step and
   evidence links. Update the index's status and one-line finding.

Give each study a stable, descriptive name such as `initial_visual_mass`,
`estimator_policy_integration`, `failure_behavior_grouping`,
`physical_parameter_sweeps` or `interaction_estimate_updates`. Keep repeated runs
distinguishable with a timestamp or identifier. Preserve each command's existing
output root; an explicit output option can select a descriptive run name, for
example `runs/deployment/estimator_policy_integration_<run>/`.

Study pages link to the original run rather than copying its dataset, checkpoint
or frozen source. Historical run names and locations stay unchanged. A proposed
study does not need an empty code directory: add a driver when it has executable
experimental choices; put reusable implementation in `robot_adaptation/`.

Policy and estimator architectures are explicit in saved manifests/checkpoints.
Keep that identity meaningful when introducing a new model. Checkpoint shapes
and old source snapshots retain their original interpretation.

## Legacy entrypoints and historical artifacts

Old top-level Python entrypoints are small forwards to canonical modules.
Canonical commands are listed above. Update active code only in
`robot_adaptation/` and `experiments/`.

| Older folder / command | Active implementation |
| --- | --- |
| `learning-withoutCLI/` | `robot_adaptation/policy/` and `robot_adaptation/simulation/` |
| `visual_mass_estimator/` | `robot_adaptation/estimators/visual_mass/` |
| `failure_collection/` | `robot_adaptation/collection/` |
| `failure_discovery/` | `robot_adaptation/analysis/failure_discovery/` |
| `data_check/` | `robot_adaptation/analysis/plots/` |
| `crest/` | `experiments/crest/` |

The older `grip_support/` setup tools are separate; their unfinished training
interface is not the active learner. Existing historical reports remain under
`failure_discovery/results/`, `data_check/plots/`, `data_check/plots_archive/` and
`data_check/code_archive/`.

Historical `runs/`, analysis results, plots, and code archives were not moved.
New policy outputs use `runs/policy/`; latest-checkpoint lookup also searches
historical `runs/learning/`. New collection, plotting, and discovery outputs use
fresh run directories by default.

New collection and CREST snapshots include the shared implementations, with
package-relative source paths. New PPO preparation keeps its standalone frozen
runtime source format. Pre-migration source fingerprints differ from the new
layout, so continuing an old collection requires its original saved code; the
migration does not rewrite old experiment identities.

## Source versus generated work

Keep code, configuration, experiment plans, research summaries and licenses in
Git. Generated datasets, checkpoints, caches, reports, presentation renders and
temporary preview media have scoped ignore rules. Presentation source assets
that must be versioned belong in an `assets/` directory; the presentation rules
keep those visible.

Existing tracked historical outputs remain tracked. Ignore rules do not remove
them from the index or delete them. New runs stay local under `runs/`; research
pages may link to these local artifacts, which are unavailable in a clean clone
until the corresponding saved run is supplied.
