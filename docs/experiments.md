# Experiment index

[Project overview](../README.md) · [Development guide](development.md)

Each study has one result page: **question → method → result → limitation →
next step**, followed by links to its code and saved evidence. The pages organize
the research; datasets, checkpoints and historical reports stay at their original
paths.

## Movement capability

| Study | Status | What it establishes |
| --- | --- | --- |
| [Movement capability](experiments/movement_capability.md) | Foundation | Policy training and inspection infrastructure; model versions remain explicit. |

## Initial physical understanding

| Study | Status | What it establishes |
| --- | --- | --- |
| [Initial visual mass](experiments/initial_visual_mass.md) | Completed baseline | 2.76 g test MAE for fixed-density cubes with held-out sizes. |
| [Estimator–policy integration](experiments/estimator_policy_integration.md) | Completed integration | Initial image → mass estimate → fixed policy → simulated lift; three configured examples. |

## Understanding failures

| Study | Status | What it establishes |
| --- | --- | --- |
| [Failure behavior grouping](experiments/failure_behavior_grouping.md) | Exploratory | Two broad behavior groups among 154 failures; causal failure types unvalidated. |
| [Physical parameter sweeps](experiments/physical_parameter_sweeps.md) | Completed collection; exploratory analysis | 29,830 attempts; mixed outcomes across starts at some settings. |
| [Failure intervention responses](experiments/failure_intervention_responses.md) | Exploratory | Seven observed response profiles in 90 matched mass/friction trials. |

## Learning through interaction

| Study | Status | What it establishes |
| --- | --- | --- |
| [Policy conditioning on mass and friction](experiments/physics_conditioned_policy.md) | Final evaluation complete; training ended by user | Accurate physics improves the saved 19,000-cycle policy: true-input success 95.3%, versus 87.3% with constant friction. |
| [Interaction estimate updates](experiments/interaction_estimate_updates.md) | Final evaluation complete; estimator transfer partial | Historical mass/friction recovery and actual adaptive retry benefit are supported. New-policy mass recovery transfers; friction recovery and consistent improvement with more attempts remain unestablished. |

The [final objective report](../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/final_summary.md)
separates accurate-input benefit, historical recovery, new-policy transfer and
actual retry success. Native-carry retries succeed in 86/90 adaptive sequences
versus 78/90 with fixed guesses. The user ended training after 19,489 rollouts;
the evaluated checkpoint contains 19,000 cycles, and 50,000 was not completed.
Reliable friction inference remains the central open question.

## Supporting experiments

| Study | Status | Role |
| --- | --- | --- |
| [CREST context selection](experiments/crest_context_selection.md) | Implemented experiment; completed selection not established here | Explore which context factors affect a repairable parameterized skill. |
| [Official-policy baseline](experiments/official_policy_baseline.md) | Baseline complete; sweep stopped | Ten CPU attempts achieved 7/10 success, below the configured quality gate. |

## Reading the status labels

- **Foundation:** infrastructure or a starting skill; not a claim of adaptation.
- **Completed baseline / integration:** a bounded experiment was recorded.
- **Exploratory:** observed patterns need further interpretation or validation.
- **Proposed:** a question or design, not a completed result.

To add a study, use the [result-page template](experiments/_template.md), link it
from this index, and follow the [recording conventions](development.md#adding-an-experiment).
