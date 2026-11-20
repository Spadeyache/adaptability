# Policy conditioning on mass and friction

[Experiment index](../experiments.md)

**Status:** Final evaluation is complete. Accurate mass/friction inputs improve the fixed policy on the tested cases, and adaptive retries beat fixed guesses. Estimator transfer supports mass recovery; friction recovery remains inconclusive. Training ended at the user's request after 19,489 rollouts; the evaluated saved checkpoint contains 19,000 learning cycles. The original 50,000-cycle budget was not completed.

## Question

Does supplying accurate mass and friction improve a fixed pick-and-lift policy's performance compared with fixed or incorrect estimates?

## Method

The explicit mode appends one object-friction coefficient at observation index 45. Weight remains in newtons at index 42, and the two measured compression forces retain indices 43 and 44. The original 45-input interface remains available. Extending a saved policy copies its existing parameters and sets the added first-layer columns to zero; verification checks that initial deterministic actions and critic values are preserved.

Train on the independent cross of five masses (40, 64, 100, 250 and 500 g) and three object-friction coefficients (0.2, 0.5 and 1.0). Static and dynamic object friction are equal; finger friction is fixed at 0.2. Material readbacks accompany evaluation. All five movement/opening/strength actions are optimized jointly.

The short pilot uses 200 rollout/learning cycles. The original long-run plan requested 50,000 further cycles: 150 million transitions with 60 environments and 50 steps per rollout. The user subsequently ended training early. These cycles differ from the four internal PPO optimization epochs per rollout. Continuation preserves learned weights and restarts the optimizer, as in the existing trainer.

Evaluation changes only the supplied physical inputs while preserving actual mass, friction and starting state. Compare true inputs, fixed friction, incorrect friction, fixed mass, incorrect mass and both fixed. Rebuild scenes between paired conditions and compare initial-state fingerprints to remove stale solver-history effects. Bootstrap whole starting-seed blocks, retaining the complete mass/friction grid within each block. The final evaluation uses reserved starting seeds 91000–91009.

## Result

The 46-input interface, checkpoint migration, CPU physical readbacks and a three-cycle CUDA training integration run passed. The fresh-scene initialization control has exactly zero sensitivity to the added friction input, as expected from zero initialization. It therefore provides no evidence of a friction benefit before learning.

The 200-cycle pilot completed 600,000 transitions with finite parameters. On 30 matched diagnostic starts (15 physical settings across only two independent seed blocks), true-input final success increased from 50.0% at initialization to 80.0%; sustained success at the last ten sampled endpoints increased from 43.3% to 80.0%.

The following rows all use that same 200-cycle pilot checkpoint. They change the supplied mass/friction values while matching actual physics and starts; they do not compare different amounts of training.

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

The replacement started from this reviewed corrective checkpoint with the original 50,000-cycle / 150-million-transition budget. Training runtime, model, task, controller and guard source matched the reviewed pilot before launch. The learning-rate and optimizer-guard changes were made together, so their individual effects are not isolated. These local screens established readiness to continue; final accurate-friction and adaptive-retry findings came from the later reserved tests below.

The replacement's first scheduled checkpoint, 501, also completed a CPU diagnostic on nine physical settings and two diagnostic seed blocks, with at most three attempts per sequence. True-input success was 17/18, adaptive success 16/18, and fixed-guess success 15/18. These compare with 16/18, 16/18 and 15/18 at the corrective pilot, respectively; no aggregate decline is evident on this bounded sample. Adaptive minus fixed success is 5.6 percentage points with a seed-block interval of 0–11.1 points, so this does not establish an adaptive-retry benefit. At that point the reserved final test starts were unused and the full run was incomplete.

At checkpoint 2,001, a further CPU diagnostic using the same nine settings, two diagnostic seed blocks and three-attempt budget produced true-input success 18/18, adaptive success 17/18 and fixed-guess success 15/18. These counts maintain or improve the checkpoint-501 aggregates, unlike the earlier unguarded checkpoint-2,001 diagnostic (2/18, 0/18 and 0/18, respectively). The adaptive-minus-fixed descriptive bootstrap interval is degenerate at +11.1 percentage points because both of only two seed blocks have the same observed difference. It does not establish precise uncertainty or broad retry benefit. These comparisons do not isolate the learning-rate change from the update guard and do not replace the final capability tests below.

### Checkpoint budget comparison

An additional diagnostic compares guarded checkpoints 2,001, 5,001 and 8,501, saved after 2,000, 5,000 and 8,500 completed learning cycles. It uses ten matched diagnostic starting seeds, the full 15-condition training grid and nine intermediate mass/friction conditions, with six supplied-input interventions per checkpoint: 4,320 episodes in total. The CPU probe preserves the native training task's randomized goal XY and floating-height rule, including its reward and grip-force penalty. An independent 50-step check matched native CPU `LiftTask` observations, actions and rewards exactly.

All 4,320 episodes completed with matched physical resets, materials, controllers and source/checkpoint identities. With true inputs:

| Completed learning cycles | Training physics (150 cases) | Intermediate physics (90 cases) |
| --- | ---: | ---: |
| 2,000 | 88.7% | 95.6% |
| 5,000 | 90.0% | 96.7% |
| 8,500 | 92.7% | 98.9% |

The training-grid gain from 2,000 to 8,500 cycles is 4.0 percentage points, with a 95% seed-block interval of 1.3–6.7 points. The more recent 5,000-to-8,500 gain is 2.7 points on training physics (−1.3 to 6.7) and 2.2 points on intermediate physics (0 to 5.6). These recent intervals do not establish a positive success gain or prove a plateau. They do not justify the full 50,000-cycle budget by themselves.

Friction conditioning changed substantially: at 8,500 cycles, correct friction beats the deliberately incorrect-friction intervention by 10.0 points on both grids (training interval 7.3–12.7; intermediate 6.7–13.3). Correct versus fixed friction 0.5 is only +0.7 points on training physics (−2.7 to 4.0) and +2.2 on intermediate physics (0 to 5.6). Thus, accurate friction helps relative to incorrect values, while a success advantage over the chosen constant remains unestablished. The growing correct-versus-incorrect gap partly reflects declining performance with incorrect friction, rather than only improving true-input performance; estimator-error tolerance and actual adaptive retry benefit still require separate evaluation.

Measured grip force fell between 2,000 and 5,000 cycles on matched episodes that lifted under both policies. After 5,000 cycles, that conditional reduction is inconclusive on training physics and remains small on intermediate physics. The report retains pair counts, intervals and differing contact-duration limitations, so failed lifts cannot masquerade as lower-force efficiency.

The report separates changes from more training from benefits of accurate physics inputs. These diagnostic starts can inform budget/model selection; the reserved final starts remain separate. Earlier CPU retry diagnostics above used vertical targets and do not substitute for this carrying-task comparison. The checkpoint comparison does not evaluate estimator updates or establish a new-policy closed-loop benefit.

### User-requested training stop

The user explicitly ended policy training. The supervisor and its GPU worker were terminated, with all checkpoint files, source snapshots, guard logs and earlier diagnostics retained. The execution log contains 19,489 completed rollouts (58,467,000 transitions). The latest saved model is `ckpt_19001.pt`, saved before cycle 19,001 and therefore containing 19,000 completed learning cycles (57,000,000 transitions). Its tensors are finite and its 46-input interface is intact. The final 489 executed cycles are recorded in logs but are not part of that saved checkpoint.

The stop is recorded as `stopped_by_user`, with the original supervisor termination output preserved separately. This is not completion of the original 50,000-cycle budget. Final evaluation used the fixed saved checkpoint and pinned estimator; policy training remains stopped.

### Final reserved policy comparison

The saved 19,000-cycle policy completed 900 GPU episodes: all 15 training physics settings, ten new starting-seed blocks and six supplied-input modes. Each row below uses the same policy weights, actual object physics and matched starts. The test measures the benefit of accurate inputs; it does not measure the benefit of more training.

| Supplied inputs | Final success | True-input advantage, 95% seed-block interval |
| --- | ---: | ---: |
| True mass and friction | 143/150 (95.3%) | Reference |
| True mass, fixed friction 0.5 | 131/150 (87.3%) | +8.0 pp [4.0, 12.0] |
| True mass, incorrect friction | 106/150 (70.7%) | +24.7 pp [21.3, 28.0] |
| Fixed mass 40 g, true friction | 91/150 (60.7%) | +34.7 pp [30.7, 38.7] |
| Incorrect mass, true friction | 120/150 (80.0%) | +15.3 pp [12.7, 18.0] |
| Fixed mass 40 g and friction 0.5 | 85/150 (56.7%) | +38.7 pp [33.3, 42.7] |

All required comparisons in the assessment specification have positive intervals. This supports accurate-input benefit on the tested grid, including a friction-specific advantage over the chosen constant. Sustained success at the last ten sampled endpoints was 142/150 with true inputs versus 120/150 with fixed friction. Samples occur at 20 Hz; this does not measure behavior between samples.

The earlier 8,500-cycle CPU diagnostic and this final GPU test use different backends and starting seeds. Their difference cannot be attributed solely to additional training.

### Final estimator and retry checks

With policy and estimator weights frozen, actual retries used nine estimator-held physics settings, ten seed blocks per method, and a maximum of three attempts. The original vertical task and an additional task preserving native trained carry-goal geometry were declared before final collection.

| CPU task | Adaptive | Fixed guesses / blind retry | True inputs | Adaptive advantage, 95% seed-block interval |
| --- | ---: | ---: | ---: | ---: |
| Vertical goal | 89/90 (98.9%) | 78/90 (86.7%) | 90/90 (100%) | +12.2 pp [8.9, 15.6] |
| Native carry goal | 86/90 (95.6%) | 78/90 (86.7%) | 89/90 (98.9%) | +8.9 pp [2.2, 14.4] |

Adaptive retries help on both tested tasks. Much of the gain occurs at the 500 g, friction-0.4 condition: fixed guesses succeed in 0/10 sequences, versus adaptive 10/10 vertically and 7/10 carrying. This does not establish accurate friction inference as the cause.

Separate controlled transfer tests collected five independent attempts at every setting of the specified 4×4 mass/friction grid, using new starting seeds 64000–64004 and fixed supplied guesses of 100 g / 0.4. The estimator receives observable inputs only. Five-attempt mass MAE is 56.3 g vertically and 59.4 g carrying, versus a 140.0 g constant prior, with positive condition-bootstrap improvement intervals. Friction MAE is 0.1824 / 0.1785 versus prior 0.2250, but both improvement intervals cross zero. Error curves are nonmonotonic; native-carry mass error increases from 55.1 g at one attempt to 59.4 g at five.

Historical held-condition recovery supports both variables, while new-policy transfer supports mass and leaves friction inconclusive. Across three historical estimator training seeds, friction improvement over the prior has a positive interval in only two. See the [estimator result page](interaction_estimate_updates.md) for the full separation of these findings.

The revised-scope audit passed all 15 evidence requirements, including exact saved/executed counters, complete 19,489-rollout guard coverage, finite pinned weights, source identities, reserved starts and matched raw retry/transfer data. Maximum accepted full-rollout mean KL was 0.049999912, within 0.05; individual states are not bounded. Independent review recomputed outcomes and intervals and replayed all 160 transfer predictions. The full unit suite ran 158 tests: 154 passed and four opt-in checks were skipped. Process completion and partial scientific success remain separate conclusions.

## Limitation

The first model conditions on one object material coefficient, with equal static/dynamic friction and fixed finger/table materials. It uses ideal simulator state and force observations. Readback confirms native training-table static/dynamic friction is 0.3, matching the historical clean collection subset and retry driver. Final policy intervals describe new starts conditional on the training physics grid. Transfer intervals describe physical-condition variation conditional on five shared starts. Real-robot transfer, general causal attribution and discovery of new variables remain untested.

The earlier reset-only diagnostic produced misleading differences even with zero friction columns; those results are invalid and superseded by fresh-scene comparisons.

## Next step

The remaining research question is reliable friction recovery under a changed collecting policy. A future study should distinguish information from attempts from a useful learned prior, and test separate mass-only/friction-only updates before attributing retry improvements to either variable. No further training is part of this completed evaluation.

## Code and evidence

- [Assessment criteria](../../configs/adaptation_assessment.json)
- [Training interface](../../robot_adaptation/policy/train.py)
- [Matched evaluation](../../experiments/physics_conditioning/evaluate.py)
- [Checkpoint comparison plan](../../runs/physics_conditioning/checkpoint_comparison_20261026/plan.json)
- [Checkpoint comparison progress](../../runs/physics_conditioning/checkpoint_comparison_20261026/comparison_status.json)
- [Completed checkpoint comparison](../../runs/physics_conditioning/checkpoint_comparison_20261026/assessment/comparison.md)
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
- [User-requested early stop and saved-checkpoint identity](../../runs/physics_conditioning/long_50000_guarded_20261026/early_stop.json)
- [Final objective synthesis](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/final_summary.md)
- [Final native-carry combined assessment](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/assessment_native_carry/assessment.md)
- [Final vertical combined assessment](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/assessment_vertical/assessment.md)
- [Revised-scope completion audit](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/completion_audit.json)
- [Independent raw-data scientific review](../../runs/physics_conditioning/long_50000_guarded_20261026/post_stop_evaluation/final_scientific_review.json)
- [Guarded checkpoint-501 CPU diagnostic](../../runs/physics_conditioning/long_50000_guarded_20261026/diagnostics/checkpoint501_cpu_retries/summary.json)
- [Guarded checkpoint-2,001 CPU diagnostic](../../runs/physics_conditioning/long_50000_guarded_20261026/diagnostics/checkpoint2001_cpu_retries/summary.json)
- [Combined interim assessment](../../runs/interaction_adaptation/pilot200_assessment/assessment.md)
