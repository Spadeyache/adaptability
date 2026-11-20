# Addendum: privileged-input comparisons

This additive label preserves the original reports and their numbers. It applies to the supplied-physics conditions listed below, not automatically to other conditions in the same reports.

| Earlier report | Conditions relabeled | Permitted interpretation |
|---|---|---|
| [Policy conditioning](/home/ubuntu/faliure/docs/experiments/physics_conditioned_policy.md) | True-input conditions and interventions retaining true or truth-derived parameter inputs | Privileged-input comparisons of a policy given physics. Not recovery of physics from observations. |
| [Interaction updates](/home/ubuntu/faliure/docs/experiments/interaction_estimate_updates.md) | True-input retry reference only | Privileged-input comparison. No blanket relabeling of fixed/adaptive estimator arms. |
| [Earlier context world model](/home/ubuntu/faliure/runs/context_dynamics/pick_gpu_2m/world_model_z8/RESULTS.md) | World model conditioned on E(true constants), including its latent readouts | Privileged-input comparison; readout measures representation of supplied labels. |
| [Baseline study](/home/ubuntu/faliure/BASELINE.md) | B2 Oracle only | Privileged-input comparison by design; B0 and B1 are separate conditions. |

The main slip study's 0.755 mass readout and 11.6% same-task gain are not relabeled as privileged-input results: the matching training sources use observed state/action and prediction-fitted z. The separate original two-task runs have a missing historical training-source hash and remain unverified. See [the per-run audit](/home/ubuntu/faliure/results/loop/002/MAIN_RUN_AUDIT.md).
