# Direct-force Plain–Oracle gate

**Primary gate: FAIL.** Oracle’s relative improvement in moving-coast one-step error is **+7.09%** relative to Plain. The five-seed mean ± sample SD is **0.065608 ± 0.00272** for Plain and **0.060959 ± 0.00374** for Oracle.

The required ≥20% reduction is **not met**; the requirement `Oracle mean + SD < Plain mean − SD` is **not met**. The criterion was frozen before fitting and the held-out evaluation. No belief model was trained.

The moving-coast one-step mean gap, Plain minus Oracle, is **0.0046494**. The sum of the two seed SDs is **0.0064628**. The test checks both the relative improvement and this stricter spread separation.

![Moving-coast prediction gate](/home/ubuntu/faliure/clean/shove_gate/results/figures/moving_coast_gate.png)

## Pilot and full collection

The pusher was removed. A native horizontal external force is applied at the cube center of mass, with zero external torque, followed by exactly zero applied force during coast. This is an idealized force actuator on an infinite plane. It does not establish Panda-controller performance.

The same cube identities remain independently sampled over 0.18–0.50 kg and material friction 0.4–1.0; the floor material is 0.3. Per-cube mass, cube friction, and configured combined friction remain only in `data/cubes.json`. Each shove uses independent force magnitude 4–7 N and world direction, with random start XY and yaw. Duration is uniform on the 5 ms simulation grid **60, 65, …, 100 ms**, using a separate property-independent random stream.

| Measurement | Same ten-cube pilot | Full 230-cube collection |
|---|---:|---:|
| Episodes | 100 | 2,300 |
| Launch fraction, speed >2 cm/s at release | 100.00% | 100.00% |
| Moving fraction of retained coast steps | 91.34% | 89.66% |
| Maximum coast distance | 0.6924 m | 1.4487 m |
| Retained model transitions | 7,345 | 148,076 |
| Stationary transitions trimmed | 24,225 | 578,816 |
| Maximum time to confirmed stop onset | 0.576 s | 0.902 s |

The pilot met the user’s continuation condition: at least 90% launches and maximum coast distance below 1 m. Collection therefore proceeded under that explicit conditional authorization. **The full sample included 3 episodes above 1 m (3 training, 0 held-out), reaching 1.4487 m.** The pilot distance bound did not hold for every sampled object/action combination; no such episodes were discarded.

All collected episodes reached confirmed rest. Raw coast recordings last 1.5 s, fixed before collection to allow stop confirmation across the allowed parameter range. Model episodes retain five 5 ms steps after stop onset, rounded up to a model boundary. Confirmation requires 100 ms continuously below 1 mm/s linear speed and 0.05 rad/s angular speed. No moving transitions were trimmed.

## Held-out prediction results

Each model was evaluated on the same 30 held-out cubes and their ten shoves. The table reports mean ± sample SD across five paired seeds. Within a seed, starts are averaged within episode, then episodes within cube, then cubes equally. Positive relative improvement favors Oracle. “Beyond spread” means the positive mean gap, Plain minus Oracle, exceeds the sum of the two seed SDs, equivalently `Oracle mean + SD < Plain mean − SD`.

| Group | Metric | Plain mean ± SD | Oracle mean ± SD | Mean gap: Plain − Oracle | Relative improvement | Beyond spread? |
|---|---|---:|---:|---:|---:|---|
| Push | One-step NMSE | 0.063629 ± 0.00347 | 0.05814 ± 0.00402 | 0.0054887 | +8.63% | no |
| Push | Ten-step rollout NMSE | 0.12519 ± 0.0129 | 0.11053 ± 0.0154 | 0.014654 | +11.71% | no |
| Push | Stopping-position error (m) | 1.1083 ± 0.0868 | 0.79502 ± 0.491 | 0.31325 | +28.26% | no |
| **Moving coast — gate** | One-step NMSE | 0.065608 ± 0.00272 | 0.060959 ± 0.00374 | 0.0046494 | +7.09% | no |
| Moving coast | Ten-step rollout NMSE | 0.091445 ± 0.00691 | 0.10288 ± 0.0108 | -0.011436 | -12.51% | no |
| Moving coast | Stopping-position error (m) | 0.89718 ± 0.151 | 0.59802 ± 0.218 | 0.29917 | +33.35% | no |
| All retained steps | One-step NMSE | 0.057374 ± 0.00252 | 0.053447 ± 0.00286 | 0.0039274 | +6.85% | no |
| All retained steps | Ten-step rollout NMSE | 0.085709 ± 0.00712 | 0.092204 ± 0.00939 | -0.0064952 | -7.58% | no |
| All retained steps | Stopping-position error (m) | 0.93922 ± 0.134 | 0.63118 ± 0.263 | 0.30804 | +32.80% | no |

![All three prediction groups and metrics](/home/ubuntu/faliure/clean/shove_gate/results/figures/gate_metrics.png)

One-step error is normalized MSE over all 15 cube-state coordinates. Ten-step rollout error averages normalized state MSE across all ten recursively predicted steps; the tenth-step-only diagnostic is also saved in the JSON. Push windows remain in push; moving-coast starts must be moving and their window remains in coast. All-step windows can cross release. Full raw traces provide targets beyond the trimmed endpoint.

Stopping-position error is Euclidean XY error in meters. From every eligible retained start, predictions continue to the **preplanned release time plus 1.5 s**, then compare with the recorded confirmed-rest position. Neither the actual stop time nor the trimmed file length determines the prediction horizon. Rollouts use their own intermediate states and the preplanned applied-force sequence.

**The primary gate is a one-step test.** For moving-coast starts, the independently reported ten-step improvement is -12.51%, and the stopping-position improvement is +33.35%. Oracle’s stopping-position error is 0.59802 ± 0.218 m. The stopping-position improvement also falls within the combined seed spread. No clipping, rotation projection, physics correction, or observed-state reset is applied during rollouts.

No scored forecasts were nonfinite. Finite errors remain reported at their measured size, including any large recursive errors.

## Actions-only probes and trimming

The actual pilot probe holds one cube out at a time and pools its out-of-fold predictions across all ten cubes. The fixed first 100 ms applied-force schedule includes the independently commanded pulse duration; it does not include observed stopping duration or trimmed length. A separate 200-train/30-test generator audit uses only preplanned commands, without dynamics.

| Probe scope/input | Family | Mass R² ± seed SD | Combined-friction R² ± seed SD |
|---|---|---:|---:|
| Pilot command schedule | extra trees | -0.3141 ± 0.0051 | -0.3425 ± 0.0050 |
| Pilot command schedule | polynomial ridge | -0.3091 ± 0.0000 | -0.3025 ± 0.0000 |
| Planned commands, 200/30 cubes | extra trees | -0.0584 ± 0.0004 | -0.0179 ± 0.0015 |
| Planned commands, 200/30 cubes | polynomial ridge | -0.0591 ± 0.0016 | -0.0173 ± 0.0008 |

The ten-cube leave-one-out fold-mean baseline is R² -0.2346 for mass and -0.2346 for combined friction. Pilot raw R² values are reported literally and are not called near zero. The wider planned-command audit has small negative R² and shows no useful predictive signal. Cube friction and combined friction are affine targets, so their R² values agree.

| Forbidden side-channel diagnostic | Family | Mass R² ± seed SD | Combined-friction R² ± seed SD |
|---|---|---:|---:|
| Trimmed length only | extra trees | 0.1574 ± 0.0022 | -0.1555 ± 0.0056 |
| Trimmed length only | polynomial ridge | 0.0942 ± 0.0000 | -0.1147 ± 0.0000 |
| Commands + trimmed length | extra trees | 0.3526 ± 0.0128 | -0.2210 ± 0.0108 |
| Commands + trimmed length | polynomial ridge | 0.1991 ± 0.0000 | -0.1936 ± 0.0000 |

**Trimmed length reveals physical outcomes.** The positive mass-prediction scores above demonstrate why length, termination indicators, padding masks, stop times and IDs are forbidden model features. Plain and Oracle are per-transition MLPs receiving only the explicit state/action whitelist, with true mass and combined friction added only for Oracle. Scoring-group masks and observation counts are bookkeeping, not network inputs. The length diagnostic is not evidence that the randomized force generator used object labels.

## Training, frozen evaluation, and limits

Both families have three 256-unit ReLU hidden layers and predict normalized state change. Training uses episodes 0–7 of the 200 training cubes; episodes 8–9 select checkpoints. State/action statistics use only the training subset, and context statistics use training cube labels only. All models receive 10,000 Adam updates, batch size 512, with cosine learning rate 0.001→0.0001. Sampling first chooses an episode uniformly, then a retained transition uniformly. Paired families have identical minibatch-stream hashes.

Validation checks every 500 updates select the lowest equal-cube/episode all-step one-step NMSE while still completing the entire training budget. All ten selected checkpoints and the held-out byte hashes were frozen before model scoring. The final evaluation ledger contains one completed invocation; test results did not choose a checkpoint or change the gate.

| Seed | Plain selected update | Oracle selected update | Plain training time (s) | Oracle training time (s) |
|---:|---:|---:|---:|---:|
| 101 | 500 | 500 | 19.43 | 19.37 |
| 211 | 1000 | 4000 | 19.39 | 19.73 |
| 307 | 500 | 3500 | 23.07 | 26.05 |
| 401 | 1000 | 1000 | 19.70 | 29.47 |
| 503 | 2000 | 2000 | 20.18 | 19.24 |

Summed recorded model-training time: **215.63 s**. Summed recorded test prediction time: **14.41 s**. These sums exclude collection, import/startup, data preparation, audits and report generation.

Oracle receives true parameters. This experiment tests whether those parameters improve prediction in the direct-force simulator; estimating them from experience remains untested. The setup includes exact simulator state, a fixed floor material, and a finite sampled object/action distribution. Five-seed SD measures optimization variation, not hardware noise or a confidence interval over all possible cubes.

The frozen prediction gate failed. Work stops at this reported comparison; no test-driven tuning or belief model was started.

## Saved evidence

- [Five-seed comparisons and gate decision](/home/ubuntu/faliure/clean/shove_gate/results/gate_summary.json)
- [Completed final-evaluation ledger](/home/ubuntu/faliure/clean/shove_gate/results/final_evaluation.json)
- [Frozen checkpoint and source manifest](/home/ubuntu/faliure/clean/shove_gate/results/checkpoint_seal.json)
- [Held-out trace byte hashes](/home/ubuntu/faliure/clean/shove_gate/results/test_data_seal.json)
- [Frozen training/evaluation protocol](/home/ubuntu/faliure/clean/shove_gate/results/model_protocol.json)
- [Pilot physics/data audit](/home/ubuntu/faliure/clean/shove_gate/results/pilot_audit.json)
- [Full collection audit](/home/ubuntu/faliure/clean/shove_gate/results/full_audit.json)
- [Independent final results and artifact audit](/home/ubuntu/faliure/clean/shove_gate/results/final_audit.json)
- [Actions-only and length probe results](/home/ubuntu/faliure/clean/shove_gate/results/action_probe.json)
- [Direct-force experiment setup and reproduction](/home/ubuntu/faliure/clean/shove_gate/README_DIRECT.md)

Every figure is generated only from the saved held-out seed metrics. Paired points represent actual saved runs; no synthetic samples, smoothing, or confidence intervals are introduced. PNG and SVG versions are saved under `results/figures/`.

Rebuild this report after the completed evaluation:

```bash
cd /home/ubuntu/faliure
PYTHONPATH=clean/vendor:. MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove_gate.report_direct
```

Input SHA-256 hashes:

- `results/gate_summary.json`: `d9d3b081b3eb772cbf87cebec991706746f7be2dbc58a28b43f5a604145883b8`
- `results/final_evaluation.json`: `34a7a7c6fe7a55398361a8d5c555719255712b7856b6c50b356ab4811c059641`
- `results/pilot_audit.json`: `21afb53126e53cdd00ed526a1064e423413baef54f21ecd2c45710dbd2a81c73`
- `results/full_audit.json`: `3787f77865329109e15a4605ee1717422b2e69a47fd6abde9bdb4b1168523fef`
- `results/pilot_gate.json`: `4694d22a12dde26f279d2973b5032edefb6467c283d728f67819823486caaafe`
- `results/action_probe.json`: `056f8903e1d29eb6c71c3055635c88de26192e276652dd4a6b4e59ef6a3a6a0e`
- `results/metrics.json`: `0ab38ec6820d884fa66c7328ce83e5dbab4eae2fb4c2628b2f6abc703486377b`
- `data/protocol.json`: `420818e027a8168123cd26ff88e19a44e80ef67abfef46d4ff951b2734d26c5f`
- `results/model_protocol.json`: `d534fcff81111ec723187588aec5e8a2e36e732918d7d9ff4303fd2fabe1a629`
- `results/checkpoint_seal.json`: `a596f003dd9c2d75708b3ee5e19938294a2a3297230b770e05d79c3791321ee1`
- `results/test_data_seal.json`: `f7882b9a3a5a67ef7d88b07163fac1c53992aab9d7f80fd27fa5b40518091112`
- `results/timing.json`: `92cc2ca68f2e89a403e29a6801c375e42f30b33737d9919cfcf4fc2556f13399`
- `results/pilot_timing.json`: `66ad10842986d430700f1100799c3a557541bda13a570c3f008881004711c080`
- `results/test_metrics/plain_seed_101.json`: `8134754c79eea52c707b707a3797be5b8ca9aaf238dd149e22e01204e28ac4b7`
- `results/training_logs/plain_seed_101.json`: `bfe9b71c447c8be33439aa02df833d8027dab538b1c9445b02484970bd36a9a8`
- `results/test_metrics/oracle_seed_101.json`: `b6b647a2ed45f1e6f75fdb547a569ab4bdeea105f46f53e0622e3d0c4de84f88`
- `results/training_logs/oracle_seed_101.json`: `028eae00e6b8ecf62d83f0bb1926c17dfedc0b03a1048fb2472aeefea045701e`
- `results/test_metrics/plain_seed_211.json`: `124fea9826c8f58ac523607fce95c47432f3ed4ef1e11ddae7c769f6d24dd6c0`
- `results/training_logs/plain_seed_211.json`: `e0d23221a71fa2637d3273f76dc96544e6031c32af853e364225d51335eab529`
- `results/test_metrics/oracle_seed_211.json`: `e32839fbe0322db0303fb0fd19fadd5603af67387100b33fe242524931d8ae73`
- `results/training_logs/oracle_seed_211.json`: `b9ea3ca0206d074cbbe34e7f63f36b0d0d82dcdc31b444f4668ca8c81bbf4bff`
- `results/test_metrics/plain_seed_307.json`: `29c6af585720d1bb9b6c2bcc5cf188635872b0f5b10f53a999942275ea704a5a`
- `results/training_logs/plain_seed_307.json`: `a0ac5b05b8cbeac67b7c80b65fb2a422107dd95c3e3a0ce69535c8c2f6801678`
- `results/test_metrics/oracle_seed_307.json`: `db19ec79a39e1f56f94cdbfd70b21cf607b5784e7a310bf743379cb1cedc6001`
- `results/training_logs/oracle_seed_307.json`: `fbe30fee462ddc1ef900effcb519b28c17014fc0ac8954b3015c51dba2fbfa87`
- `results/test_metrics/plain_seed_401.json`: `19c49ef055a8fd7e9b852312d57697ee1593017e8de1ef54cc1ce3973f4e9848`
- `results/training_logs/plain_seed_401.json`: `dc8b1da21b929c3578d9ceb15b819b537cc8762275d4b0fbc87a8d5bf25fc568`
- `results/test_metrics/oracle_seed_401.json`: `a3292d593e9b6c7114a7428c67222df09b4941473e68ad7df64b029d9a26bd25`
- `results/training_logs/oracle_seed_401.json`: `5343ec6415d6d3cb67c6d2eb91e7076f31f44e24d42e934c7129dd6f4e6ffd13`
- `results/test_metrics/plain_seed_503.json`: `d5d4c207e3a9c17dc4a33249f6c674c75ad3f1842ebc242621b1457832d221d7`
- `results/training_logs/plain_seed_503.json`: `ab8bc58ca0795bc647d43d01ed4b0e329d0c04e3d8d0e6204fab2eb910da3e57`
- `results/test_metrics/oracle_seed_503.json`: `4a46ede2df72acb0c9159152e40e48279034fea248f3512ae06c6d84deaf8ff3`
- `results/training_logs/oracle_seed_503.json`: `c48b210cf1db563457ba9462c73f0ea0fbf907a3a3e53ab21e30221dde47028f`
