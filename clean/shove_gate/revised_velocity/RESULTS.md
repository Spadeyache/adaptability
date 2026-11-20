# Revised horizontal-velocity prediction gate

**Gate: FAIL.** On 30 fresh test cubes, Oracle’s moving-coast horizontal-velocity-change NMSE is **0.24984 ± 0.0191**, versus **0.20969 ± 0.00927** for Plain across five paired seeds. Relative improvement: **-19.15%**.

The ≥20% reduction condition is **not met**; the seed-spread condition is **not met**. The mean gap is -0.040153 versus a sum of seed SDs of 0.028384. The gate requires `Oracle mean + SD < Plain mean − SD`.

**Oracle is worse by 19.15%** on the new primary metric in all five seeds. The gap in Plain’s favor (0.0401532) exceeds the combined seed SD (0.0283839); the clear separation favors Plain.

![Fresh-test primary gate](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/results/figures/velocity_gate.png)

## Written revision and retired test set

The [revised protocol and reasons](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/REVISED_PROTOCOL.md) were saved before physics scoring, fresh collection, and revised training. The previous 15-coordinate average could obscure a horizontal friction signal; this motivated a new metric and does not establish the cause of the earlier failure. The new primary score isolates horizontal velocity change and uses a fitting-only change-energy scale. Checkpoints now minimize this validation score, with a predeclared extension rule.

The old test results informed this revision. IDs 200–229 are retired; their data, checkpoints and reported results remain unchanged in the parent experiment. New test IDs 230–259 use seed **2026111517**, the same generator, and no outcome-based rejection. The 200 training cubes and their original traces are reused byte-for-byte.

## Physics sanity reference measured before learning

The requested prediction is `−combined_friction × 9.81 × 0.005 × vxy/|vxy|`, with zero at exactly zero speed. It is not clipped. Near-zero was defined before measurement as NMSE ≤0.01. The all-training reference **meets** that threshold.

| Training subset | Requested NMSE | Requested RMSE (m/s) | Stop-clipped diagnostic NMSE | Moving-coast steps |
|---|---:|---:|---:|---:|
| Fitting episodes 0–7 | 0.00974751 | 0.00170626 | 0.00004532 | 70,238 |
| Validation episodes 8–9 | 0.01074347 | 0.00179131 | 0.00003667 | 17,632 |
| All training | 0.00994671 | 0.00172360 | 0.00004359 | 87,870 |

The full-step decrement would cross zero in 1,887 of 87,870 starts. The recorded cube stops instead of reversing. Clipping the decrement to available speed reduces reference error by 99.56%; outside these stopping crossings, NMSE is 0.00004517. At speeds ≥0.1 m/s, measured deceleration divided by combined friction × gravity is 0.99998913. These measurements closely match simple sliding friction; the dominant deviation is the final partial stopping step. Small remaining contact-simulation residuals have no identified solver cause.

The frozen normalization scalar is **0.000298673152081 (m/s)²**, the equal-cube/episode average horizontal delta energy on fitting moving-coast steps. A zero-change predictor has NMSE 1 on that subset. Validation-only sanity error is reported separately even when it lies above 0.01. The analytic reference is a diagnostic baseline, not a proven irreducible error floor.

[Detailed pretraining physics interpretation and diagnostics](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/PHYSICS_FLOOR_REPORT.md)

## Fresh test collection and action independence

| Measurement | Reused 200 training cubes | Fresh 30 test cubes |
|---|---:|---:|
| Episodes | 2,000 | 300 |
| Launch rate | 100.00% | 100.00% |
| Moving fraction of retained coast | 89.78% | 89.05% |
| Maximum coast distance | 1.448709 m | 0.845645 m |
| Retained transitions | 130,002 | 18,437 |
| Trimmed stationary transitions | 502,130 | 76,298 |

Fresh collection took 17.17 s wall time. Force remains uniform 4–7 N at the COM, direction/start position/yaw random, and duration uniform over 60,65,…,100 ms. Actions are exactly zero in coast. Every episode reached confirmed rest. The fixed 1.5 s raw coast, 100 ms stop confirmation and five-step trimmed tail are unchanged. No moving step was trimmed, and no sampled episode was discarded.

| Actions-only probe, planned commands | Mass R² ± seed SD | Combined friction R² ± seed SD |
|---|---:|---:|
| extra trees | -0.03763 ± 0.00171 | -0.00218 ± 0.00400 |
| polynomial ridge | -0.03878 ± 0.00721 | -0.00906 ± 0.00582 |

Probe hyperparameters use training-cube validation only. Commands describe the fixed first 100 ms force schedule and independently commanded duration. No observed length, padding mask, stop time, ID, phase or termination signal enters either dynamics model. Per-cube true mass and friction remain in `cubes.json`; only Oracle receives mass and combined friction.

## Fresh held-out prediction scores

All scores weight starts within episode, episodes within cube, then cubes equally. Positive improvement favors Oracle. “Beyond spread” means the positive Plain-minus-Oracle mean gap exceeds the sum of the two sample SDs. Five-seed SD measures optimization variation.

| Group | Metric | Plain mean ± SD | Oracle mean ± SD | Gap | Improvement | Beyond spread? |
|---|---|---:|---:|---:|---:|---|
| Push | Horizontal Δvelocity NMSE | 1.077 ± 0.0316 | 0.47187 ± 0.0697 | 0.60512 | +56.19% | yes |
| Push | Full-state one-step NMSE | 0.067897 ± 0.00409 | 0.063241 ± 0.00365 | 0.0046551 | +6.86% | no |
| Push | Ten-step rollout NMSE | 0.14702 ± 0.0168 | 0.11739 ± 0.00961 | 0.02963 | +20.15% | yes |
| Push | Tenth-step-only NMSE | 0.19876 ± 0.0379 | 0.14942 ± 0.0272 | 0.049336 | +24.82% | no |
| Push | Stopping-position error (m) | 0.37818 ± 0.0958 | 1.6623e+05 ± 3.72e+05 | -1.6623e+05 | -43954026.21% | no |
| **Moving coast — gate** | Horizontal Δvelocity NMSE | 0.20969 ± 0.00927 | 0.24984 ± 0.0191 | -0.040153 | -19.15% | no |
| Moving coast | Full-state one-step NMSE | 0.060894 ± 0.00327 | 0.060327 ± 0.00232 | 0.00056741 | +0.93% | no |
| Moving coast | Ten-step rollout NMSE | 0.09896 ± 0.00664 | 0.11557 ± 0.00586 | -0.016614 | -16.79% | no |
| Moving coast | Tenth-step-only NMSE | 0.1329 ± 0.0157 | 0.15823 ± 0.00783 | -0.025327 | -19.06% | no |
| Moving coast | Stopping-position error (m) | 0.28296 ± 0.0461 | 25956 ± 5.8e+04 | -25956 | -9173158.16% | no |
| All retained steps | Horizontal Δvelocity NMSE | 0.36702 ± 0.00978 | 0.26392 ± 0.0232 | 0.1031 | +28.09% | yes |
| All retained steps | Full-state one-step NMSE | 0.055501 ± 0.00298 | 0.055016 ± 0.00175 | 0.00048509 | +0.87% | no |
| All retained steps | Ten-step rollout NMSE | 0.097412 ± 0.00808 | 0.10627 ± 0.0042 | -0.0088575 | -9.09% | no |
| All retained steps | Tenth-step-only NMSE | 0.13467 ± 0.0193 | 0.14808 ± 0.00997 | -0.01341 | -9.96% | no |
| All retained steps | Stopping-position error (m) | 0.30038 ± 0.0509 | 80565 ± 1.8e+05 | -80565 | -26820941.19% | no |

![All groups and metrics](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/results/figures/all_metrics.png)

The new NMSE divides horizontal-change MSE by the fixed fitting scale. Old full-state metrics retain their original 15-coordinate normalization. Ten-step forecasts recursively use predicted states; the primary ten-step secondary metric averages all ten horizons, with tenth-only error also shown. Push windows stay in push; moving-coast windows start moving and remain in coast. Stopping-position forecasts run to planned release +1.5 s, never to an observed stop or trimmed length. No clipping, physics correction, or observed-state resets are applied.

| Seed | Plain primary NMSE | Oracle primary NMSE | Paired Oracle improvement |
|---:|---:|---:|---:|
| 101 | 0.2177635 | 0.22518772 | -3.41% |
| 211 | 0.21454307 | 0.24944452 | -16.27% |
| 307 | 0.19732049 | 0.25777422 | -30.64% |
| 401 | 0.21655499 | 0.27634305 | -27.61% |
| 503 | 0.20227209 | 0.24047078 | -18.88% |

**Long-horizon instability remains in the reported results.** Finite rollout values can still be catastrophic. The per-seed stopping-position errors below are retained in the five-seed means and SDs; no outlier was dropped, clipped, or replaced. These are prediction errors, not physical distances traveled by the cubes.

| Seed | Plain moving-coast stop error (m) | Oracle moving-coast stop error (m) | Oracle push stop error (m) | Oracle all-step stop error (m) |
|---:|---:|---:|---:|---:|
| 101 | 0.352373967 | 0.258802868 | 0.29443064 | 0.26448959 |
| 211 | 0.2970324 | 0.248518004 | 0.267512804 | 0.251789498 |
| 307 | 0.269821215 | 0.264787085 | 0.276521879 | 0.265721191 |
| 401 | 0.227493862 | 129779.987 | 831129.159 | 402824.213 |
| 503 | 0.268054527 | 0.292960652 | 0.323019449 | 0.300298267 |

Oracle seed **401** has moving-coast stopping error **129,779.987 m**, while the other Oracle seeds range 0.2485–0.2930 m. This unstable run drives the five-seed Oracle mean to **25,956.210 m**.

## Oracle relative to physics on the same fresh cubes

| Reference | Physics NMSE | Oracle NMSE | Oracle excess NMSE | Oracle / physics error | Physics RMSE (m/s) | Oracle RMSE (m/s) |
|---|---:|---:|---:|---:|---:|---:|
| Requested unclipped formula | 0.011296676 | 0.24984406 | 0.23854738 | 22.1166 | 0.0018368489 | 0.008638386 |
| Stop-clipped diagnostic | 2.7481353e-05 | 0.24984406 | 0.24981658 | 9091.4 | 9.0597695e-05 | 0.008638386 |

Oracle RMSE is the square root of its five-seed mean MSE. The unclipped formula is the requested comparison; the stop-clipped variant is a separately labeled diagnostic. Both use the same fresh moving-coast starts and the fitting-only normalization.

## Validation curves and adaptive training

Both models retain three 256-unit ReLU layers and the original full-state normalized-change training loss. Episodes 0–7 fit; episodes 8–9 select checkpoints using moving-coast horizontal velocity-change NMSE. Adam, batch512 and paired minibatch streams are unchanged. Learning rate cosine-decays 0.001→0.0001 over the first 10,000 updates, then remains at 0.0001.

Every 10,000 updates, the protocol compares the best validation score in the last 2,500 updates with the previous best. At least 0.1% improvement by either family extends both by another 10,000, up to 100,000. Checkpoint selection is independent of the final update budget.

![Validation curves](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/results/figures/validation_curves.png)

| Seed | Updates/model | Plain selected update | Oracle selected update | Plain selected validation NMSE | Oracle selected validation NMSE | Still improving at cap? |
|---:|---:|---:|---:|---:|---:|---|
| 101 | 40,000 | 10,500 | 34,500 | 0.21278581 | 0.2397007 | no |
| 211 | 20,000 | 16,500 | 15,000 | 0.20363388 | 0.25139715 | no |
| 307 | 20,000 | 12,000 | 15,500 | 0.20703451 | 0.26983749 | no |
| 401 | 20,000 | 10,500 | 16,000 | 0.22824645 | 0.28825772 | no |
| 503 | 20,000 | 14,500 | 9,500 | 0.20935784 | 0.26887084 | no |

| Seed | Block end | Plain final-window improvement | Oracle final-window improvement | Extend both? |
|---:|---:|---:|---:|---|
| 101 | 10,000 | +4.82% | +4.37% | yes |
| 101 | 20,000 | -2.27% | +1.28% | yes |
| 101 | 30,000 | -7.58% | +0.22% | yes |
| 101 | 40,000 | -8.69% | -0.35% | no |
| 211 | 10,000 | +6.13% | +4.75% | yes |
| 211 | 20,000 | -0.47% | -0.95% | no |
| 307 | 10,000 | +6.08% | +5.22% | yes |
| 307 | 20,000 | -3.88% | -0.84% | no |
| 401 | 10,000 | -0.82% | +2.72% | yes |
| 401 | 20,000 | -1.22% | -1.13% | no |
| 503 | 10,000 | +6.49% | +5.22% | yes |
| 503 | 20,000 | -2.35% | -4.40% | no |

Summed pair training wall time: **438.88 s**; summed model test-prediction time: **14.16 s**. These exclude collection, startup, preparation, audits, and reporting.

No scored metric/start/seed entries were nonfinite.
All ten checkpoints were frozen before the sole predictive test evaluation. Test results did not select checkpoints, extend training, or change the gate.

The revised gate failed. Work stops at this comparison; no test-based tuning or belief-model training follows.

## Saved evidence

- [Protocol and reasons written before running](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/REVISED_PROTOCOL.md)
- [Pretraining physics interpretation](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/PHYSICS_FLOOR_REPORT.md)
- [Saved training physics measurements](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/results/physics_floor.json)
- [Fresh-test physics measurements](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/results/test_physics_floor.json)
- [All five-seed comparisons and gate](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/results/gate_summary.json)
- [Completed final-evaluation ledger](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/results/final_evaluation.json)
- [Collection audit](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/results/full_audit.json)
- [Commands-only probe](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/results/action_probe.json)
- [Frozen checkpoint/source manifest](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/results/checkpoint_seal.json)
- [Fresh held-out byte hashes](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/results/test_data_seal.json)
- [Independent final results audit](/home/ubuntu/faliure/clean/shove_gate/revised_velocity/results/final_audit.json)

All figures come from saved measurements and validation logs. No fabricated runs or smoothed validation samples are used. PNG/SVG files are in `results/figures/`.

Rebuild this report only after final evaluation completes:

```bash
PYTHONPATH=clean/vendor:. MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove_gate.revised_velocity.report
```

