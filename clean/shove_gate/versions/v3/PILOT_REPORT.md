# Varied-shove gate: ten-cube pilot

**Status: pilot complete; awaiting your go. The Oracle-versus-Plain prediction gate has not been run.**

The active dataset contains **100 shoves on 10 newly sampled training cubes** and **21,600 transitions**. No motion was collected for any other cube identity. No Plain, Oracle, or belief model has been trained. The only fitted models are the explicitly requested actions-only generator probes, described below.

## Range, randomization, and data

Mass is independently sampled uniformly over **0.18–0.5 kg**, narrowed from the previous 0.04–0.50 kg range to avoid the very long light-cube slides. Cube material friction is independently sampled over **0.4–1**. The fixed floor material is 0.3; configured combined friction uses PhysX averaging, `(cube friction + floor friction)/2`, giving **0.35–0.65**. Per-cube mass, cube friction, and combined friction values reside only in `data/cubes.json`.

The split and command plans are fixed for 200 training cubes and 30 test cubes, with 10 shoves each. Only training cube IDs 0–9 were physically simulated. Every shove independently randomizes its 2–4 N cap, world push direction, face contact offset, cube start XY, and cube yaw. Property sampling and command sampling use separate RNG streams; no plan is accepted or rejected based on true properties.

The physical setup remains an idealized 5 g spherical pusher and a six-DOF 4 cm cube on an infinite plane. This is not a Panda controller evaluation. In the accepted revision, the pusher translates freely in XYZ, has locked rotation, and has gravity and damping disabled. Each force pulse lasts 80 ms, followed by a fixed 1 s coast. Physics runs at 1 ms; model transitions are saved every 5 ms.

Each transition contains `cube_id`, `episode_id`, `step`, `phase`, `state`, `action`, and `next_state`. IDs and phase labels select bookkeeping and metrics; numeric IDs are never model features. State contains observed cube pose/velocities and pusher pose/velocities. Actions contain only commanded force, cap, engagement flag, and a release target computed from the initial plan. Measured force, true properties, stop time, and variable episode length are excluded from actions.

Every episode has **16 push and 200 coast transitions**: **1,600 push** and **20,000 coast** transitions in the pilot. Recording continues after rest, so sequence length cannot reveal the properties.

## Measured collection time

| Item | Recorded time |
|---|---:|
| Accepted 100-episode invocation, including startup | 16.065 s |
| Accepted episode collection + compression, summed | 13.456 s |
| Mean accepted episode | 0.1346 s |
| Median accepted episode | 0.1216 s |
| Generator-only action probe, separate from collection | 9.405 s |

Linear extrapolation of the measured serial episode time gives **309.5 s (5.16 min)** for all 2,300 episodes, or **296.0 s (4.93 min)** for the remaining 2,200. These are projections, not completed runs; they exclude startup, audits, probes, model training, and possible failures or reruns.

### Corrected attempts retained in the archive

The first 100-episode attempt used the contact API export as a full-force measurement. Oblique pushes revealed that it omitted tangential impulse, so that attempt was archived and excluded from the active dataset. The second attempt stopped after one episode because an other-contact guard counted zero-impulse proximity points as force-bearing contacts. The accepted third revision uses full XYZ momentum balance and verifies that other pusher contacts have zero normal impulse.

Across these corrections there were **201 physical episode executions** (100+1+100), all on the same ten identities and the same preplanned settings. Only the final 100 episodes are accepted. The archived first invocation took **17.449 s**. The second attempt records **0.115 s** of episode time; its complete process wall time was not saved. The summed recorded episode-processing time over all 201 executions is **24.454 s**. This sum excludes startup and debugging time and is not presented as total elapsed project time.

## Physical and leakage checks

The independent saved-trace audit **passed**. It reconstructs every command from the property-blind plan, checks fixed recording lengths, state/next-state alignment, immutable source/data seals, stopping windows, and the ten-cube boundary.

Actual transmitted contact force is reconstructed every physics step as `F_contact = F_command − m_pusher × Δv_pusher/Δt`. This includes all three components. The reconstruction uses the known actuator mass, not the hidden cube mass. There are no pusher translation constraints, gravity, damping, or force-bearing contacts with other bodies. The normal component independently agrees with the simulator contact export. The instantaneous release velocity reset is excluded from contact-force reconstruction.

- Largest full contact-force/cap ratio: **95.255%**; no cap exceedances.
- Largest full contact force: **3.7361 N**, under its **3.9780 N** cap.
- Maximum pusher–cube contact force after release: **0 N**.
- All **100/100** episodes reached confirmed rest.
- Maximum release-to-rest time: **0.254 s** (confirmation time excluded).
- Maximum release-to-rest displacement: **12.616 cm**.
- Maximum start-to-rest horizontal displacement: **16.675 cm** (displacement, not accumulated path length).
- Maximum launch speed: **1.0020 m/s**.
- **51/100** launches exceeded 1 mm/s (**49** were at or below that threshold); **49/100** exceeded 2 cm/s.
- **853/20,000** coast transitions (**4.265%**) meet the predefined moving mask: full linear speed>1 mm/s or angular speed>0.05 rad/s.

**Only 4.265% of coast transitions are moving; 95.735% are at or below the motion thresholds.** Many randomized low-strength or oblique shoves barely launch, and the fixed coast includes a long stationary tail. Thus the primary all-coast metric is dominated in sample count by resting states. The future report will also show the predefined moving-coast score; it will not substitute that score for the frozen primary gate after viewing test results.

## Actions-only generator probe

This probe uses all 2,300 preplanned action schedules: 2,000 from the 200 training identities and 300 from the 30 test identities. It does not read their motion or simulate additional cubes. Ten pilot cubes alone would be insufficient for the requested held-out-cube probe.

The inputs are ten features describing the commanded force/cap, direction, contact offset, start XY, and yaw. No IDs, measured force, stop time, or duration features are used. Polynomial ridge and ExtraTrees each use five seeds, cube-disjoint 180/20 training validation splits, then refit all 200 training identities. Hyperparameters are never selected using test labels. The two independent fitted targets are mass and combined friction; cube-friction predictions use their train-only affine relationship for reporting.

| Probe | Mass test R², mean ± seed SD | Cube friction test R² | Combined friction test R² |
|---|---:|---:|---:|
| Polynomial ridge | -0.05056 ± 0.00000 | -0.01723 ± 0.00000 | -0.01723 ± 0.00000 |
| ExtraTrees | -0.05517 ± 0.00091 | -0.02558 ± 0.00114 | -0.02558 ± 0.00114 |

For comparison, a constant training-mean predictor has test R² **-0.05662** for mass and **-0.01559** for combined friction. Every fitted probe score is below 0.1 and none is positive. The observed values are close to zero and provide no useful predictive signal from commands. Negative R² values are reported literally. Ridge has zero seed spread because every validation split selected the same regularization and the final full-training fit is deterministic.

Cube-cluster bootstrap intervals are saved with the scores. Thirty test identities still give limited precision; this empirical check and the code-path audit do not prove statistical independence. **This action-probe result is not a PASS of the Oracle-versus-Plain gate.**

## Frozen next step, requiring your go

After approval: collect the remaining 190 training cubes and 30 test cubes, retaining the accepted pilot episodes. Train Plain and Oracle with five paired seeds, three 256-unit hidden layers, shared training-only normalization, and normalized state-change targets. Plain receives state+action; Oracle additionally receives true mass and combined friction. No belief model is planned in this step.

Report one-step NMSE, ten-step recursive rollout error, and stopping-position error separately for push and coast. Rollout windows stay within their named phase; no observed intermediate state is fed into recursive predictions. Stopping-position rollouts begin at the first push or first coast state and compare terminal predicted XY with recorded confirmed-rest XY. Aggregate equally by cube, then episode, and report five-seed mean, sample SD, and paired improvements.

The frozen primary PASS rule is **all-coast one-step NMSE**: `mean(Oracle) ≤ 0.8 × mean(Plain)` and `mean(Oracle) + SD(Oracle) < mean(Plain) − SD(Plain)`. Moving-coast, ten-step, stopping-position, and cube-only scores remain required diagnostics. No test-driven tuning or metric substitution is allowed. If the gate fails, stop and report.

**Collection and gate-model training are stopped here, pending your go.**

## Saved evidence

- [Recorded timing](/home/ubuntu/faliure/clean/shove_gate/results/timing.json)
- [Per-episode physical metrics](/home/ubuntu/faliure/clean/shove_gate/results/pilot_metrics.json)
- [Independent data and force audit](/home/ubuntu/faliure/clean/shove_gate/results/pilot_audit.json)
- [Actions-only probe scores and intervals](/home/ubuntu/faliure/clean/shove_gate/results/action_probe.json)
- [Frozen protocol](/home/ubuntu/faliure/clean/shove_gate/data/protocol.json)
- [Saved state/action schema](/home/ubuntu/faliure/clean/shove_gate/data/schema.json)

This report reads saved evidence only. Rebuild it with:

```bash
cd /home/ubuntu/faliure
PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.shove_gate.report
```

Input SHA-256 hashes:

- `results/timing.json`: `c8d427ad8c504f8301481cda034a1de1748759e129e012c2c5964e1ff1393c46`
- `results/pilot_metrics.json`: `c3d50fede15173c83ed189c6056e016d544142266f133601f286b72d14d81f1e`
- `results/pilot_audit.json`: `06b83cced825f07db73a3f9fc1625eac1325619b53214df17d0b971e37729ddc`
- `results/action_probe.json`: `196da32c059144f5e51af6557ee44a6682d8e5b3341b8cc5de57d2a7e0cbda0a`
- `data/protocol.json`: `5facb7931cc04cfa0c14c936125f2366548c8f2498a863941b4d55627f9b8675`
- `data/plans.json`: `d27775504663c99593e824f8824f1f25e71558fa9137c4d8d0361d05c9d5079c`
- `versions/v1/results/pilot_metrics.json`: `4f3982181bd830966c98f15c41fba09977599b8bf658da594a077e1c970429aa`
- `versions/v2/results/pilot_metrics.json`: `a8f6aa4be76a68d26a73846855ec36ca40f491a540297ffcc7fc3f24ad0f4a89`
- `versions/v1/results/timing.json`: `b0380bdbd521737b5e4c5d89a9dfc6d565628b27d2b7b9a47b1c64789fb9f552`
