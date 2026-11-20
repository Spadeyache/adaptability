# Interaction estimate updates

[Experiment index](../experiments.md)

**Status:** Final saved-policy evaluation complete in simulation. Adaptive retries improve success on the tested tasks. Mass recovery transfers to the final policy; reliable friction transfer and consistent refinement with every attempt remain unestablished.

## Question

Can a robot improve its physical understanding through deployment interaction
and use that understanding to perform better on later actions or attempts?

## Evaluated training scope

The user ended training after **19,489 executed rollouts / 58,467,000 collected
transitions**. The evaluated `ckpt_19001.pt` contains **19,000 completed learning
cycles / 57,000,000 transitions**; the 489 later rollouts are not represented in
those saved weights. The originally requested **50,000-cycle budget was not
completed**, and training remains stopped. The final estimator checkpoint was
frozen throughout evaluation.

Final evaluation used the declared [post-stop plan](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/plan.json).
Both the legacy vertical picking task and a native carry adapter were tested.
The latter preserves the trained randomized goal XY, floating-height range and
reward; vertical-only success would describe a narrower capability. The
[completion audit](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/completion_audit.json)
and [independent scientific review](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/final_scientific_review.json)
passed without material bugs. Their positive status applies to the revised
evaluation scope, not completion of 50,000 training cycles.

## Implemented first study

Use the existing dense collection, retaining only conditions where mass and object friction differ from nominal. All 440 selected episodes across 44 physical conditions have equal object static/dynamic friction; other physical readbacks were audited. Split whole physical conditions into 26 training, nine validation and nine test groups before extracting snippets or fitting normalization. Target pairs and every episode/snippet stay within one split.

Starting seeds recur across physical settings, so this split holds out physics rather than unseen starting states. The historical condition-bootstrap intervals are conditional on those shared starts. The separate controlled transfer test uses fresh starting seeds and a new collecting policy.

Extract contact bouts from measured compression, ending on sustained release, observed stable holding or timeout. This produces 2,369 snippets, including unsuccessful touches. Simulator contact masks are excluded from deployment extraction. A small GRU encodes sensor/command histories and outcomes computed from observations available at the end of a snippet. Average all snippets within each attempt, then accumulate equally weighted attempt embeddings to predict positive mass and friction. Train and inference use the same aggregation. Neither true/supplied mass nor friction labels enter the feature vector.

Supervise every accumulated evidence prefix and choose the checkpoint using validation only. Compare held-condition errors with a constant training-median predictor. Confidence intervals bootstrap physical conditions, retaining repeated attempts. Historical attempts are independently reset fixed-policy episodes; pooling them is distinct from adaptive retries.

## Historical held-condition recovery

For the preselected deployment training seed 101:

| Evidence | Mass MAE | Friction MAE |
| --- | ---: | ---: |
| Constant training median | 111.9 g | 0.1867 |
| One attempt | 48.4 g | 0.1381 |
| Three attempts | 33.9 g | 0.0897 |
| Five attempts | 29.6 g | 0.0725 |

At five attempts, the 95% condition-bootstrap intervals for improvement over the median prior are 48.0–121.8 g for mass and 0.0041–0.2349 for friction. Both exclude zero. Five-versus-one improvement is positive for both targets across three fixed training seeds. Mass improves over the prior across all seeds; the friction prior-comparison interval crosses zero for seed 102. This limits a robust friction-generalization claim.

## Earlier pilot retry diagnostic

The actual retry driver runs the policy, records an attempt, updates estimates
using only observable inputs and retries. Using the corrected estimator and
completed 200-cycle policy pilot on nine estimator-held physical settings and
two fresh starting-seed sequences per setting, success within three attempts
was adaptive **15/18**, fixed estimates/blind retries **16/18**, and true
parameters **17/18**. The adaptive-minus-fixed interval was −11.1 to 0 percentage
points with only two independent seed blocks. This early diagnostic demonstrated
integration without a measured adaptive benefit; it did not establish broad
degradation either. The larger final evaluation below is a separate experiment.

## Earlier pilot-policy transfer falsification

Freeze the same estimator and the completed 200-cycle policy. Collect 80 new CPU attempts on the full cross of masses 80, 160, 320 and 480 g and friction 0.25, 0.45, 0.65 and 0.85. Each condition uses five fresh starts (62000–62004), fixed supplied policy guesses of 100 g and friction 0.4, and nominal nuisance physics. The grid and supplied guesses are independent of the target labels; no retraining or model selection follows these errors.

| Evidence | Mass MAE | Friction MAE |
| --- | ---: | ---: |
| Same training-median predictor, on this new grid | 140.0 g | 0.2250 |
| One attempt | 100.0 g | 0.3978 |
| Three attempts | 68.1 g | 0.1844 |
| Five attempts | 52.1 g | 0.1413 |

Five-attempt mass improvement over the prior is supported (95% condition-bootstrap interval 49.1–130.6 g). Friction improvement is inconclusive (−0.0160 to 0.1838), and one-attempt friction is worse than the prior. Both errors improve from one to five attempts. These results restrict the positive historical finding: mass transfers in this test; reliable friction transfer is not established. Confidence intervals are conditional on the five shared starts, rather than a broad starting-state population.

## Final-policy transfer: mass supported, friction inconclusive

Freeze the saved 19,000-cycle policy and the same estimator. On each CPU task,
collect **80 independent attempts** across the full 16-condition cross of masses
80, 160, 320 and 480 g and friction 0.25, 0.45, 0.65 and 0.85. Each condition uses
five reserved starts (64000–64004), fixed supplied policy guesses of 100 g and
friction 0.4, and nominal nuisance physics. The estimator accumulates completed
attempts, while its predictions do not change the collecting policy's inputs.
Only observable input fields and timestamp metadata cross the prediction
boundary. Raw-input replay reproduced all 160 final transfer predictions exactly.

| Evidence | Vertical mass MAE | Vertical friction MAE | Native carry mass MAE | Native carry friction MAE |
| --- | ---: | ---: | ---: | ---: |
| Constant training median | 140.0 g | 0.2250 | 140.0 g | 0.2250 |
| One attempt | 77.1 g | 0.1766 | 55.1 g | 0.1837 |
| Two attempts | 59.6 g | 0.1517 | 63.9 g | 0.1816 |
| Three attempts | 53.7 g | 0.2018 | 47.3 g | 0.1799 |
| Four attempts | 54.8 g | 0.1730 | 60.0 g | 0.1870 |
| Five attempts | 56.3 g | 0.1824 | 59.4 g | 0.1785 |

At five attempts, mass improves over the constant prior on both tasks: the
95% condition-bootstrap improvement interval is **47.5–125.6 g vertically** and
**41.9–122.9 g in native carry**. Friction's intervals include zero on both:
−0.0708 to 0.1639 vertically and −0.0465 to 0.1382 in native carry. This supports
mass transfer on these cases, with friction transfer inconclusive.

The refinement curves are **nonmonotonic**. Vertical five-versus-one mass
improvement is supported (2.8–42.0 g), while the corresponding friction interval
includes zero. In native carry, five-versus-one improvement is not established
for either parameter: mass's interval is −26.7 to 15.2 g and friction's is
−0.0572 to 0.0734. Native carry's mass point error worsens from 55.1 to 59.4 g.
Consequently, the historical improvement from more evidence cannot be claimed
as a reliable property under the final policy.

These intervals resample physical conditions and remain conditional on the
five shared starts. Prefix order is fixed, so the number of attempts and the
particular added starting seeds are not isolated effects. The earlier pilot
used different starting seeds; the two transfer studies do not isolate an
effect of additional policy training.

## Final actual retries: success gains on both tasks

Run the frozen saved policy with fixed initial estimates, interaction-updated
estimates, and true parameters. The nine estimator-held physical settings each
receive ten matched starting-seed sequences, for **90 sequences per method per
task**, with an equal budget of up to three attempts. Each attempt rebuilds the
physical scene; object properties stay fixed within a sequence. Fixed and
adaptive methods receive the same initial 100 g / friction 0.4 guess and have
identical first-attempt inputs. Subsequent adaptive inputs use only completed
attempts. The ten sequence blocks use reserved seeds starting at 54000.

| Task | Fixed estimates / blind retries | Adaptive estimates | True parameters | Adaptive − fixed, 95% seed-block interval |
| --- | ---: | ---: | ---: | ---: |
| Vertical pick-and-hold | 78/90 (86.7%) | 89/90 (98.9%) | 90/90 (100.0%) | +12.2 pp [8.9, 15.6] |
| Native carry | 78/90 (86.7%) | 86/90 (95.6%) | 89/90 (98.9%) | +8.9 pp [2.2, 14.4] |

Both adaptive-versus-fixed intervals exclude zero. This establishes an actual
retry benefit on these tested simulation cases, beyond the earlier integration
diagnostic. The confidence calculation retains all physical conditions within
each of the ten paired starting-seed sequence blocks.

The gain is concentrated in the **500 g / friction 0.4** condition. Fixed
estimates fail all ten sequences on both tasks; adaptive estimates succeed on
10/10 vertical sequences and 7/10 native carry sequences, versus true parameters
at 10/10 and 9/10. Other conditions include smaller gains and occasional losses.
Broad improvement across failure types is therefore not established.

Accurate supplied physical inputs also benefit this final policy in the
separate [physics-conditioned policy study](physics_conditioned_policy.md).
However, retry success does not establish that accurate friction recovery
caused the improvement. A mass correction or useful learned prior may explain
much of the effect. A prior-only ablation has not been performed, and final
new-policy friction recovery remains inconclusive.

## Limitation

The historical dataset holds out nine physical settings, and its collection covers single-factor sweeps and two diagonals rather than a full factorial grid. The separate final transfer tests cover the full 16-condition crossed grid described above. All selected historical episodes have some observable contact. A compression-based contact proxy can include table contact; an independent audit found 38 of 2,369 snippets without oracle object contact. Sensor observations are ideal simulator proxies.

An unfamiliar outcome does not identify its physical cause. Historical recovery,
controlled new-policy transfer and actual retry benefit are separate findings.
The final retry gain is supported on the tested grids and starts; broad transfer,
reliable friction recovery, real-robot sensing and causal attribution are not
established. Predictions outside training bounds are flagged; no uncertainty
calibration or active probing policy is implemented.

An adaptive-versus-fixed retry gain could also reflect a useful estimator prior or parameter shifts that do not accurately recover physics. Interpret retry success together with the independent transfer-error measurements before attributing a gain to physical inference.

## Next step

Keep the evaluated checkpoints frozen and training stopped. To explain the
observed benefit, compare the interaction updater with a prior-only parameter
change and mass-only versus friction-only updates. Investigate what new-policy
interactions distinguish friction, then test refinement with independently
varied evidence order and fresh starts. These are research next steps, not
completed results or authorization to resume training. Continue to judge
recovery and task benefit against the
[saved assessment criteria](../../configs/adaptation_assessment.json).

## Code and evidence

- [Estimator implementation and commands](../../robot_adaptation/estimators/interaction/README.md)
- [Prepared snippets and split manifest](../../runs/interaction_estimator/contact_dataset_v1/manifest.json)
- [Physical readback audit](../../runs/interaction_estimator/contact_dataset_v1/physical_readback_audit.json)
- [Corrected checkpoint run](../../runs/interaction_estimator/gru_v2_corrected/manifest.json)
- [Held-condition errors](../../runs/interaction_estimator/gru_v2_corrected/test_metrics.json)
- [All training-seed results](../../runs/interaction_estimator/training_seed_robustness.json)
- [Retry driver](../../experiments/interaction_adaptation/evaluate.py)
- [Assessment generator](../../experiments/interaction_adaptation/assess.py)
- [Corrected-estimator pilot retry diagnostic](../../runs/interaction_adaptation/pilot200_retry_diagnostic/summary.json)
- [Combined interim assessment](../../runs/interaction_adaptation/pilot200_assessment/assessment.md)
- [Transfer test specification](../../configs/estimator_transfer_assessment.json)
- [Crossed-grid transfer results and raw episodes](../../runs/interaction_adaptation/transfer_pilot200_crossed/summary.json)
- [Combined assessment including transfer falsification](../../runs/interaction_adaptation/pilot200_transfer_assessment/assessment.md)
- [Explicit training stop and saved-checkpoint accounting](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/early_stop.json)
- [Final post-stop evaluation plan](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/plan.json)
- [Final vertical transfer errors and raw episodes](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/transfer_vertical/summary.json)
- [Final native carry transfer errors and raw episodes](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/transfer_native_carry/summary.json)
- [Final vertical retry outcomes](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/retries_vertical/summary.json)
- [Final native carry retry outcomes](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/retries_native_carry/summary.json)
- [Final vertical assessment](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/assessment_vertical/assessment.md)
- [Final native carry assessment](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/assessment_native_carry/assessment.md)
- [Independent final scientific review and recomputed intervals](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/final_scientific_review.json)
- [Revised-scope completion audit](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/completion_audit.json)

## Reading

- [Xie & Finn — Lifelong Robotic Reinforcement Learning by Retaining Experiences](https://arxiv.org/abs/2109.09180):
  reuse past data and policies when learning successive tasks.
- [REFLECT — Summarizing Robot Experiences for Failure Explanation and Correction](https://arxiv.org/abs/2306.15724):
  use multisensory experience summaries for LLM failure explanations.
- [AHA — A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation](https://arxiv.org/abs/2410.00371):
  train a vision-language model on generated failures to detect and explain them.

## Starting points

- [Failure replays and grouping](failure_behavior_grouping.md)
- [Parameter sweeps](physical_parameter_sweeps.md)
- [Earlier experiment proposal](../../presentations/interaction_adaptation_experiment.md) — design notes, not completed results
- [Runtime guide](../../robot_adaptation/runtime/README.md)
