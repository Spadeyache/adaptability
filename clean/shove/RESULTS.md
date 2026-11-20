# Clean shove sensitivity experiment

**Correction discovered in the subsequent varied-shove pilot:** the force values
below come from PhysX's normal-only contact impulse field. They are not a bound
on full 3D contact force. The original independent momentum audit verified only
the pushing-axis component. The saved motion measurements and sensitivity plots
are unaffected. See [the corrected instrumentation](../shove_gate/README.md).

Analyzed **126 saved trials**; all 126 planned trials are present.

**This experiment uses an idealized 5 g spherical pusher on a horizontal rail and an infinite flat plane.** The plane avoids table-edge falls when light cubes travel far. It reuses the clean cube-property setter, 4 cm cube geometry, SAPIEN/PhysX contact simulation, and fixed floor material friction of 0.3. These results validate this shove configuration; they do not establish that the Panda arm controller delivers the same shove.

No world model was trained. The action policy has no mass/friction input. True properties are read from `data/cubes.json` only to install simulator properties and to label this post-run analysis. Original `clean/` training/test data and its sealed evaluation are separate.

The same approach and pulse settings are used for every cube: 0.005 m/s approach target, 0.03 N preload for 0.1 s, then an 80 ms force pulse commanded at 95% of the selected cap. The pusher releases and is parked away; the cube slides freely. Physics is sampled every 1 ms. Mass sweeps use seven values from 0.04 to 0.50 kg at friction 0.7; friction sweeps use seven values from 0.4 to 1.0 at mass 0.27 kg. Each condition has three fresh, identical resets.

## Actual contact force

Measured force is the sum of all pusher–cube contact-point impulse magnitudes divided by the physics timestep. This upper-bounds the resultant-force magnitude and includes approach, preload, pulse, and release. It is a timestep-averaged simulator contact force, not a continuous-time hardware impact peak.

| Cap (N) | Largest measured step force (N) | Peak / cap | Pulse-mean range across runs (N) |
|---:|---:|---:|---:|
| 2 | 1.907 | 95.35% | 1.7105–1.9 |
| 3 | 2.8482 | 94.939% | 2.555–2.846 |
| 4 | 3.7869 | 94.673% | 3.3994–3.7866 |

Measured cap violations: **0**. The statement is empirical for the saved runs and sampled physics steps. The 5 g pusher's own acceleration makes delivered contact force vary with cube mass even under the same command; the pulse-mean ranges above expose that difference.

![Measured contact force](/home/ubuntu/faliure/clean/shove/results/contact_force.png)

## Dependence compared with repeat spread

Effect size below is the range of the seven condition means. Repeat spread is the largest within-condition max–min across its three runs. Endpoint columns follow increasing mass or friction. These are observed ranges, not confidence intervals. Identical deterministic resets do not measure sensitivity to placement error, actuator noise, or model mismatch.

| Sweep | Cap (N) | Distance: first → last (m) | Distance effect (m) | Largest repeat spread (m) | Launch speed: first → last (m/s) | Speed effect (m/s) | Largest repeat spread (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mass | 2 | 0.9348 → 0 | 0.9348 | 0 | 3.031 → 0.0001224 | 3.0308 | 0 |
| friction | 2 | 0.011621 → 0.00017866 | 0.011442 | 0 | 0.2842 → 0.050861 | 0.23334 | 0 |
| mass | 3 | 2.2686 → 0.00037568 | 2.2682 | 0 | 4.7202 → 0.063435 | 4.6568 | 0 |
| friction | 3 | 0.045481 → 0.0081877 | 0.037294 | 0 | 0.56063 → 0.32629 | 0.23434 | 0 |
| mass | 4 | 4.1841 → 0.0045449 | 4.1796 | 0 | 6.4094 → 0.21389 | 6.1955 | 0 |
| friction | 4 | 0.1016 → 0.02817 | 0.07343 | 0 | 0.83705 → 0.60323 | 0.23382 | 0 |

![Distance and launch-speed sweeps](/home/ubuntu/faliure/clean/shove/results/sweeps.png)

**Mass:** Stopping-distance changes exceed the observed repeat spread at every collected strength. Mean distance decreases monotonically across the sampled grid.

**Friction:** Stopping-distance changes exceed the observed repeat spread at every collected strength. Mean distance decreases monotonically across the sampled grid.

The requested stop condition (dependence smaller than repeat spread) was not met in the collected stopping-distance sweeps.

## Can mass and friction be distinguished?

**Stopping distance alone is ambiguous:** increasing either property can shorten the coast. The additional observables are launch speed and free-slide deceleration. A heavier cube can launch more slowly under a given shove; friction also changes how quickly a launched cube slows after release. The table below checks the kinematic deceleration proxy `v_launch² / (2 × stopping_distance)` only where launch speed exceeds 0.02 m/s and distance exceeds 0.1 mm. It is a summary of measured motion, not a fitted world model or a claim that exact Coulomb sliding holds.

| Sweep | Cap (N) | Eligible runs | Deceleration proxy range (m/s²) | Runs ≤ 0.02 m/s at release |
|---|---:|---:|---:|---:|
| mass | 2 | 15 | 4.9137–5.5263 | 6 |
| friction | 2 | 21 | 3.4753–7.2394 | 0 |
| mass | 3 | 21 | 4.9106–5.3557 | 0 |
| friction | 3 | 21 | 3.4553–6.5015 | 0 |
| mass | 4 | 21 | 4.9091–5.0328 | 0 |
| friction | 4 | 21 | 3.4481–6.4587 | 0 |

Joint launch/coast observations can distinguish the mechanisms locally when their measured responses differ. Runs with negligible launch are uninformative about free sliding. These two axis sweeps establish sensitivity around the fixed middle values; they do **not** prove unique mass/friction recovery over the full two-dimensional parameter range. The friction axis is the cube material setting, not a direct measurement of the combined cube–floor coefficient.

A measured near-match illustrates the ambiguity: at 2 N, the 346.67 g cube with friction 0.7 stops after 0.19036 mm, while the 270 g cube with friction 1 stops after 0.17866 mm. Their launch speeds are 0.045869 and 0.050861 m/s. The distances are close, not identical; zero repeat spread here does not establish indistinguishability under a specified sensor-noise model.

At 4 N, the mass sweep's deceleration proxy spans 4.9091–5.0328 m/s², whereas the friction sweep spans 3.4481–6.4587 m/s². This supports using the free-coast portion to distinguish the two effects in this tested slice.

## Which push is clearest?

For mass, the largest observed stopping-distance range is **4 N** (4.1796 m). This compares absolute separation; with zero repeat spread, an infinite signal-to-noise ratio is neither calculated nor used to rank strengths.

For friction, the largest observed stopping-distance range is **4 N** (0.07343 m). This compares absolute separation; with zero repeat spread, an infinite signal-to-noise ratio is neither calculated nor used to rank strengths.

For launch speed alone, friction produces a similar absolute separation at the collected strengths: 0.23334–0.23434 m/s across the full friction range. The strongest-push advantage is clearer in stopping distance than in friction-related launch-speed change.

Negligible-launch counts by cap: 2 N: 6; 3 N: 0; 4 N: 0. Stronger pulses may improve separation while increasing required travel space.

## Stop time and physical checks

Time to stop is the first below-threshold state of the sustained stop window; the confirmation window itself is excluded. See the collection protocol for exact speed and angular-speed thresholds.

| Sweep | Cap (N) | Stop time: first → last (s) | Range of means (s) | Largest repeat spread (s) |
|---|---:|---:|---:|---:|
| mass | 2 | 0.618 → 0 | 0.618 | 0 |
| friction | 2 | 0.083 → 0.008 | 0.075 | 0 |
| mass | 3 | 0.963 → 0.013 | 0.95 | 0 |
| friction | 3 | 0.163 → 0.052 | 0.111 | 0 |
| mass | 4 | 1.307 → 0.044 | 1.263 | 0 |
| friction | 4 | 0.244 → 0.095 | 0.149 | 0 |

![Time-to-stop sweeps](/home/ubuntu/faliure/clean/shove/results/time_to_stop.png)

Stopped runs: 126/126. Largest recorded tilt: 0.00074733 degrees. Cube center-height range: 0.019999–0.02 m. Largest post-release pusher contact force: 0 N.

## Saved evidence and reproduction

Every plotted point comes from `results/metrics.json`, whose trial IDs refer to saved simulator traces. No curve smoothing, synthetic samples, fitted trend lines, or added repeat noise are used. `results/summary.json` records effect sizes, repeat ranges, and source hashes. `results/trial_metrics.csv` contains every measured trial outcome, indexed by cube ID; true property labels remain in `data/cubes.json`. Figures are saved as both PNG and SVG.

```bash
cd /home/ubuntu/faliure
PYTHONPATH=clean/vendor:. MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove.analyze
```

Input SHA-256 hashes:

- `data/cubes.json`: `48a772585b98adebd16c6b18e0e07b11e0501abf0430d6268f97d465b336e9a2`
- `results/metrics.json`: `e6999d3fcf0d64a417caa95a0d5dd9d7dbdd4b54b4a63c37e8249e48c10a20bb`
- `data/protocol.json`: `7ec2dbf4d3eb89820df0942d86a2e75d1e5bb5a1138b46afc15a63ed1777b4e9`
- `data/trials.json`: `9f9303316eb8ac596bd7323d654a812af377adbd988f12f3e7be3f2f0472d4cf`
