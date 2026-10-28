# Interaction estimate updates

[Experiment index](../experiments.md)

**Status:** Interaction estimator implemented and evaluated on historical episodes. Actual retry driver implemented; improved retry success remains unestablished.

## Question

Can a robot improve its physical understanding through deployment interaction
and use that understanding to perform better on later actions or attempts?

## Implemented first study

Use the existing dense collection, retaining only conditions where mass and object friction differ from nominal. All 440 selected episodes across 44 physical conditions have equal object static/dynamic friction; other physical readbacks were audited. Split whole physical conditions into 26 training, nine validation and nine test groups before extracting snippets or fitting normalization. Target pairs and every episode/snippet stay within one split.

Starting seeds recur across physical settings, so this split holds out physics rather than unseen starting states. The historical condition-bootstrap intervals are conditional on those shared starts. The separate controlled transfer test uses fresh starting seeds and a new collecting policy.

Extract contact bouts from measured compression, ending on sustained release, observed stable holding or timeout. This produces 2,369 snippets, including unsuccessful touches. Simulator contact masks are excluded from deployment extraction. A small GRU encodes sensor/command histories and outcomes computed from observations available at the end of a snippet. Average all snippets within each attempt, then accumulate equally weighted attempt embeddings to predict positive mass and friction. Train and inference use the same aggregation. Neither true/supplied mass nor friction labels enter the feature vector.

Supervise every accumulated evidence prefix and choose the checkpoint using validation only. Compare held-condition errors with a constant training-median predictor. Confidence intervals bootstrap physical conditions, retaining repeated attempts. Historical attempts are independently reset fixed-policy episodes; pooling them is distinct from adaptive retries.

## Result

For the preselected deployment training seed 101:

| Evidence | Mass MAE | Friction MAE |
| --- | ---: | ---: |
| Constant training median | 111.9 g | 0.1867 |
| One attempt | 48.4 g | 0.1381 |
| Three attempts | 33.9 g | 0.0897 |
| Five attempts | 29.6 g | 0.0725 |

At five attempts, the 95% condition-bootstrap intervals for improvement over the median prior are 48.0–121.8 g for mass and 0.0041–0.2349 for friction. Both exclude zero. Five-versus-one improvement is positive for both targets across three fixed training seeds. Mass improves over the prior across all seeds; the friction prior-comparison interval crosses zero for seed 102. This limits a robust friction-generalization claim.

The actual retry driver successfully runs the policy, records an attempt, updates estimates using only observable inputs and retries. Using the corrected estimator and completed 200-cycle policy pilot on nine estimator-held physical settings and two fresh starting-seed sequences per setting, success within three attempts was adaptive **15/18**, fixed estimates/blind retries **16/18**, and true parameters **17/18**. The adaptive-minus-fixed interval was −11.1 to 0 percentage points with only two independent seed blocks. This diagnostic demonstrates integration, with no measured adaptive benefit; it does not establish broad degradation either. The final long-run policy evaluation and larger retry comparison remain pending.

## Controlled transfer falsification

Freeze the same estimator and the completed 200-cycle policy. Collect 80 new CPU attempts on the full cross of masses 80, 160, 320 and 480 g and friction 0.25, 0.45, 0.65 and 0.85. Each condition uses five fresh starts (62000–62004), fixed supplied policy guesses of 100 g and friction 0.4, and nominal nuisance physics. The grid and supplied guesses are independent of the target labels; no retraining or model selection follows these errors.

| Evidence | Mass MAE | Friction MAE |
| --- | ---: | ---: |
| Same training-median predictor, on this new grid | 140.0 g | 0.2250 |
| One attempt | 100.0 g | 0.3978 |
| Three attempts | 68.1 g | 0.1844 |
| Five attempts | 52.1 g | 0.1413 |

Five-attempt mass improvement over the prior is supported (95% condition-bootstrap interval 49.1–130.6 g). Friction improvement is inconclusive (−0.0160 to 0.1838), and one-attempt friction is worse than the prior. Both errors improve from one to five attempts. These results restrict the positive historical finding: mass transfers in this test; reliable friction transfer is not established. Confidence intervals are conditional on the five shared starts, rather than a broad starting-state population.

## Limitation

Only nine physical settings are held out, and the underlying collection covers single-factor sweeps and two diagonals rather than a full factorial grid. All selected episodes have some observable contact. A compression-based contact proxy can include table contact; an independent audit found 38 of 2,369 snippets without oracle object contact. Sensor observations are ideal simulator proxies.

An unfamiliar outcome does not identify its physical cause. Historical recovery, controlled new-policy transfer and actual retry benefit are separate findings. Broad transfer, improved adaptive retry success, real-robot sensing and causal attribution are not established. Predictions outside training bounds are flagged; no uncertainty calibration or active probing policy is implemented.

An adaptive-versus-fixed retry gain could also reflect a useful estimator prior or parameter shifts that do not accurately recover physics. Interpret retry success together with the independent transfer-error measurements before attributing a gain to physical inference.

## Next step

Complete the [mass/friction-conditioned policy study](physics_conditioned_policy.md), freeze that policy, and compare interaction-updated estimates, fixed estimates and true parameters on matched actual retry sequences. Judge both estimator recovery and policy benefit against the [saved assessment criteria](../../configs/adaptation_assessment.json).

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
