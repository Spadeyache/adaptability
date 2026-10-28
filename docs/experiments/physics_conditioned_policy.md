# Policy conditioning on mass and friction

[Experiment index](../experiments.md)

**Status:** Guarded correction passed its 100-cycle pilot. A replacement full 50,000-cycle run is running. The original continuation stopped at logged cycle 2,436 remains preserved and incomplete; final capability assessment is pending.

## Question

Does supplying accurate mass and friction improve a fixed pick-and-lift policy's performance compared with fixed or incorrect estimates?

## Method

The explicit mode appends one object-friction coefficient at observation index 45. Weight remains in newtons at index 42, and the two measured compression forces retain indices 43 and 44. The original 45-input interface remains available. Extending a saved policy copies its existing parameters and sets the added first-layer columns to zero; verification checks that initial deterministic actions and critic values are preserved.

Train on the independent cross of five masses (40, 64, 100, 250 and 500 g) and three object-friction coefficients (0.2, 0.5 and 1.0). Static and dynamic object friction are equal; finger friction is fixed at 0.2. Material readbacks accompany evaluation. All five movement/opening/strength actions are optimized jointly.

The short pilot uses 200 rollout/learning cycles. After inspecting its direction, the requested long run uses 50,000 further cycles: 150 million transitions with 60 environments and 50 steps per rollout. These cycles differ from the four internal PPO optimization epochs per rollout. Continuation preserves learned weights and restarts the optimizer, as in the existing trainer.

Evaluation changes only the supplied physical inputs while preserving actual mass, friction and starting state. Compare true inputs, fixed friction, incorrect friction, fixed mass, incorrect mass and both fixed. Rebuild scenes between paired conditions and compare initial-state fingerprints to remove stale solver-history effects. Bootstrap whole starting-seed blocks, retaining the complete mass/friction grid within each block. Final seeds 91000 onward are reserved for the completed long run.

## Result

The 46-input interface, checkpoint migration, CPU physical readbacks and a three-cycle CUDA training integration run passed. The fresh-scene initialization control has exactly zero sensitivity to the added friction input, as expected from zero initialization. It therefore provides no evidence of a friction benefit before learning.

The 200-cycle pilot completed 600,000 transitions with finite parameters. On 30 matched diagnostic starts (15 physical settings across only two independent seed blocks), true-input final success increased from 50.0% at initialization to 80.0%; continuous success over the final half-second increased from 43.3% to 80.0%.

| Pilot supplied inputs | Final success |
| --- | ---: |
| True mass and friction | 80.0% |
| Fixed friction | 76.7% |
| Incorrect friction | 76.7% |
| Fixed mass | 56.7% |
| Incorrect mass | 73.3% |
| Fixed both | 56.7% |

The friction advantage is one of 30 attempts, with a seed-block interval of 0–6.7 percentage points. This does not demonstrate a friction-specific benefit. The small positive signed grip response to lower supplied friction is secondary evidence only.

The pilot gate passed based on stable optimization and improved overall task performance, allowing the first long continuation to start. Updated weights or increasing reward alone do not pass the capability assessment.

### Continuation stopped for correction

A same-start CPU diagnostic at checkpoint 2,001 found true-input retry success fell from 17/18 to 2/18; fixed and adaptive inputs each fell to 0/18. All 82 common attempts have identical physical reset hashes, settings, controller configurations and estimator weights. The dominant failure is stable placement error: the object is grasped and lifted with a static arm, but misses the goal by about 44.5 mm on average, beyond the 25 mm tolerance. This establishes decline on the tested CPU/off-grid cases; it does not establish an equivalent global GPU decline.

Sequential fresh-GPU comparisons also found decline on their matched cases: training-grid success fell from 24/30 for the stable pilot to 10/30 at checkpoint 2,001; an intermediate-parameter cross fell from 13/18 to 5/18. Both use two starting-seed blocks with fresh scenes and matched fingerprints. This confirms that the observed decline is not confined to the CPU diagnostic; broader reliability still requires the final reserved tests.

The code review found large sampled KL estimates around optimizer updates (including 2.607 against a 0.1 target). The stopping check does not roll back an already oversized update. These minibatch estimates do not prove an exact full-rollout KL violation or establish the sole cause of placement failure. Internal GPU validation already rebuilds scenes; stale-contact exploitation is unproven. Canonical and frozen loaders agree, and a matched CPU task-interface probe produced identical observations throughout an attempt.

The continuation was gracefully stopped at logged cycle 2,436, retaining all sources, checkpoints and diagnostics. It remains an incomplete run. The ensuing correction retained joint five-action training and the same physics/reward, lowered the warm-start learning rate, and added an exact full-rollout Gaussian KL bound with model/Adam rollback and bounded backtracking. Fresh-GPU and same-start CPU pilot checks were required before the replacement **full 50,000-cycle** run began.

### Guarded correction and replacement

The 100-cycle corrective pilot uses learning rate 1e-5 and a 0.05 bound on full-rollout mean analytic KL(old policy || candidate policy), summed over the five effective raw Gaussian actions. The old rollout comparator stays fixed across all minibatches. Rejected candidates restore model weights and Adam moments/step counters; temporary backtracking does not permanently shrink future rollout learning rates.

All predeclared task screens passed: training-grid GPU success 24/30 (minimum 23), intermediate-grid GPU success 13/18 (minimum 12), and true-input CPU retry success 16/18 (minimum 16). Root independently checked the actual episode counts, matching reset fingerprints, checkpoint hash and guard logs. Both actor branches changed across 4,000 accepted updates; all weights remained finite. Maximum accepted rollout-mean KL was 0.03603. The maximum per-state KL was 5.972: this is an average constraint, with no uniform bound on individual states. Transaction rollback/backtracking passed separate GPU stress and unit checks.

The replacement starts from this reviewed corrective checkpoint and retains the full 50,000-cycle / 150-million-transition budget. Training runtime, model, task, controller and guard source must match the reviewed pilot before it launches. The learning-rate and optimizer-guard changes were made together, so their individual effects are not isolated. These local screens establish readiness to continue; accurate-friction benefit and improved adaptive retries still require the reserved final tests.

The replacement's first scheduled checkpoint, 501, also completed a CPU diagnostic on nine physical settings and two diagnostic seed blocks, with at most three attempts per sequence. True-input success was 17/18, adaptive success 16/18, and fixed-guess success 15/18. These compare with 16/18, 16/18 and 15/18 at the corrective pilot, respectively; no aggregate decline is evident on this bounded sample. Adaptive minus fixed success is 5.6 percentage points with a seed-block interval of 0–11.1 points, so this does not establish an adaptive-retry benefit. The reserved final test starts remain unused and the full run remains incomplete.

At checkpoint 2,001, a further CPU diagnostic using the same nine settings, two diagnostic seed blocks and three-attempt budget produced true-input success 18/18, adaptive success 17/18 and fixed-guess success 15/18. These counts maintain or improve the checkpoint-501 aggregates, unlike the earlier unguarded checkpoint-2,001 diagnostic (2/18, 0/18 and 0/18, respectively). The adaptive-minus-fixed descriptive bootstrap interval is degenerate at +11.1 percentage points because both of only two seed blocks have the same observed difference. It does not establish precise uncertainty or broad retry benefit. The final capability tests remain pending, and these comparisons do not isolate the learning-rate change from the update guard.

## Limitation

The first model conditions on one object material coefficient, with equal static/dynamic friction and fixed finger/table materials. It uses ideal simulator state and force observations. Readback confirms native training-table static/dynamic friction is 0.3, matching the historical clean collection subset and retry driver.

The earlier reset-only diagnostic produced misleading differences even with zero friction columns; those results are invalid and superseded by fresh-scene comparisons.

## Next step

Complete the guarded replacement with frozen sources and estimator artifacts. After training, run final policy comparisons, actual three-attempt retries, crossed-grid estimator transfer and the completion audit with guarded-update verification. The completion monitor is active on the replacement run.

## Code and evidence

- [Assessment criteria](../../configs/adaptation_assessment.json)
- [Training interface](../../robot_adaptation/policy/train.py)
- [Matched evaluation](../../experiments/physics_conditioning/evaluate.py)
- [Long-run supervisor](../../experiments/physics_conditioning/run_long.py)
- [Three-cycle integration run](../../runs/policy/20261026T005037-126ced42/manifest.json)
- [Fresh-scene initialization diagnostic](../../runs/physics_conditioning/baseline_freshscene_diagnostic/summary.json)
- [200-cycle pilot](../../runs/policy/20261026T005423-2de4a38c/manifest.json)
- [Matched pilot diagnostic and gate](../../runs/physics_conditioning/pilot200_freshscene_diagnostic/summary.json)
- [Superseded continuation record](../../runs/physics_conditioning/long_50000_20261026/superseded.json)
- [Matched checkpoint-2,001 comparison](../../runs/interaction_adaptation/diagnostic_iteration2001_retries/comparison_to_pilot.json)
- [Optimization diagnosis and limitations](../../runs/physics_conditioning/optimization_diagnosis/audit_findings.json)
- [Fresh-GPU pilot/checkpoint-2,001 comparisons](../../runs/physics_conditioning/corrective_gpu_baselines_20261026/)
- [Corrective plan and pilot gates](../../runs/physics_conditioning/corrective_training_20261026/plan.json)
- [Independently reviewed corrective gate](../../runs/physics_conditioning/corrective_training_20261026/reviewed_pilot_gate.json)
- [Guarded replacement status](../../runs/physics_conditioning/long_50000_guarded_20261026/status.json)
- [Guarded checkpoint-501 CPU diagnostic](../../runs/physics_conditioning/long_50000_guarded_20261026/diagnostics/checkpoint501_cpu_retries/summary.json)
- [Guarded checkpoint-2,001 CPU diagnostic](../../runs/physics_conditioning/long_50000_guarded_20261026/diagnostics/checkpoint2001_cpu_retries/summary.json)
- [Combined interim assessment](../../runs/interaction_adaptation/pilot200_assessment/assessment.md)
