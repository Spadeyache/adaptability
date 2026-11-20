# Stronger-shove ten-cube pilot

**Launch target not met: 55/100 shoves launched; the target is at least 90%.**

**Stopped here, awaiting your go. No Plain, Oracle, or belief model has been trained; the prediction gate is untested.**

The same ten cube identities and all non-force randomized settings were retained. Only force caps changed from uniform 2–4 N to uniform 4–7 N using the same random quantiles. No episodes or cubes were discarded or resampled because of their physical outcomes. The previous pilot, source, probes, and data are archived in `versions/v3/`.

## Required pilot measurements

| Measurement | Previous 2–4 N pilot | New 4–7 N pilot |
|---|---:|---:|
| Launches above 2 cm/s at release | 49/100 (49%) | 55/100 (55.0%) |
| Launches above 1 mm/s, secondary threshold | 51/100 | 57/100 |
| Moving coast fraction before trimming | 4.26% | 8.21% |
| Moving coast fraction in retained data | 4.26% | 76.65% |
| Maximum coast distance | 12.616 cm | 40.413 cm |
| Retained transitions | 21,600 | 3,741 |
| Transitions trimmed | 0 | 17,859 |

The primary launch definition is horizontal release speed > 0.020 m/s. The looser 1 mm/s result is shown separately so numerical movement is not quietly substituted for a useful launch. Moving coast means a coast transition whose observed current full linear speed exceeds 1 mm/s or whose angular speed exceeds 0.05 rad/s.

Of the retained 2,141 coast transitions, **1,641 are moving (76.65%)**. Before trimming, the same moving steps were 1,641/20,000 (8.21%). No moving transition was removed.

## Force margin and observed contact

The specified maximum static-friction threshold is 0.50 × 9.8 × 0.65 = **3.185 N**. The minimum cap of 4 N exceeds it by **25.6%**. The fixed 95% drive fraction means the minimum applied pusher command is 3.8 N, giving **19.3%** margin. Caps remain independent of cube properties. This comparison concerns the configured force magnitude; it does not guarantee that an oblique, freely moving pusher maintains contact for the full pulse.

The full 3D contact-force audit passed. Maximum measured contact/cap ratio was **95.640%**; the largest absolute measured force was **6.4956 N** under its **6.9670 N** cap. All 100 episodes stopped, and maximum contact force after release was 0 N.

Launched episodes had contact during an average **66.66%** of the push interval; nonlaunching episodes averaged **12.33%**. This saved-trace contrast indicates that command magnitude alone has not resolved the short contact durations. It does not prove a unique failure mechanism. No contact-geometry changes or extra tuning runs were made in this pilot.

**99/100 moved above 2 cm/s during the push, but only 55/100 still exceeded that speed at release.** Among the 45 failed release launches, 44 moved earlier. Their median last contact above 0.1 N occurred at **8 ms** of the 80 ms pulse; the successful group reached **60 ms**. The traces show early contact loss followed by stopping before scheduled release. Movement during the push is not counted as passing the release-launch target.

## Tail trimming and leakage controls

The full one-second coast remains in raw diagnostics to verify stop confirmation and force compliance. Stop onset is the first state in 100 ms continuously below 1 mm/s linear and 0.05 rad/s angular speed. The saved model episode ends at the first 5 ms boundary at or after that onset, plus **five steps (25 ms)**. Confirmation duration itself is not counted as extra stationary training data.

**17,859/21,600 transitions were removed (82.68%)**, all stationary coast steps. Every episode retains all 16 push steps. The final dataset has 1,600 push and 2,141 coast steps; episode lengths span 21–118 transitions.

**Trimmed length is an outcome.** It may reveal properties even when every command was generated independently. Neither length, stop time, padding masks, IDs, nor termination indicators may be model features. Future stopping-position rollouts use the fixed physical horizon, extending the preplanned zero-force coast command rather than stopping at the observed file length. The probes below distinguish command-generation predictability from this separate length side channel.

## Actions-only probe rerun on the new pilot

The actual 100 saved episodes are evaluated by leave-one-cube-out prediction: nine cubes train, one cube is held out, repeated for all ten cubes. R² is computed from all pooled out-of-fold predictions, not separately on each constant-label cube. Fixed hyperparameters and five seeds avoid fitting choices to the held-out pilot predictions. Command-only inputs use the fixed 16-step push prefix; no observed state, force, sequence length, or stop information enters that input.

| Pilot input | Probe | Mass R² ± seed SD | Combined-friction R² ± seed SD |
|---|---|---:|---:|
| Command prefix only | polynomial_ridge | -0.2920 ± 0.0000 | -0.3239 ± 0.0000 |
| Command prefix only | extra_trees | -0.3354 ± 0.0036 | -0.3502 ± 0.0032 |
| Trimmed length only | polynomial_ridge | -0.2286 ± 0.0000 | -0.1956 ± 0.0000 |
| Trimmed length only | extra_trees | -0.2123 ± 0.0027 | -0.2301 ± 0.0027 |
| Commands + trimmed length | polynomial_ridge | -0.3069 ± 0.0000 | -0.2977 ± 0.0000 |
| Commands + trimmed length | extra_trees | -0.3661 ± 0.0045 | -0.3615 ± 0.0122 |

The fold-training-mean baseline gives raw R² **-0.2346** for mass and **-0.2346** for combined friction. With ten equal cube groups, leave-one-cube-out averaging itself makes this baseline negative. Scores are reported literally; a small-pilot negative score is not relabeled as zero. Cube friction and combined friction are affine targets, so their R² values agree; both are saved in the JSON.

The command-only probes perform worse than the fold-mean baseline for both properties; no useful command-to-property predictive signal was found. The raw pilot R² values are not numerically near zero, and ten held-out identities provide limited precision.

Length-only probes improve mean-squared error over the fold-mean baseline by 0.36%–3.15% in this pilot. This is a baseline-relative skill value, not raw R². It does not establish that the outcome-dependent length is safe to expose.

The length-only and commands-plus-length rows are explicit side-channel diagnostics. Their inputs are forbidden for the prediction models. A positive score there does not mean that force caps were generated from labels, but it prevents treating arbitrary variable-length action sequences as automatically leak-free.

### Separate full planned-command generator check

The updated 200-train/30-test preplanned schedules are also checked without collecting additional motion. This is supplementary to the pilot probe, not a replacement for it. It uses independent cube splits and training-only hyperparameter selection.

| Probe | Mass R² ± seed SD | Combined-friction R² ± seed SD |
|---|---:|---:|
| polynomial_ridge | -0.0507 ± 0.0000 | -0.0177 ± 0.0000 |
| extra_trees | -0.0553 ± 0.0008 | -0.0254 ± 0.0008 |

## Timing and required future comparisons

The revised 100-episode collection took **13.728 s** including startup; per-episode simulation and saving summed to **9.622 s**. The actions-only probe rerun took **26.911 s** separately. These are measured runtimes, not the total development/debugging wall time.

Every later comparison must report three groups separately: **push**, **moving coast**, and **all retained steps**. This applies to one-step error, ten-step recursive rollout error, and stopping-position error. Recursive rollouts never consume observed intermediate states. The exact group definitions and fixed-horizon stopping-position evaluation are saved in `data/protocol.json`.

The primary prediction gate is now **moving-coast one-step NMSE**: at least 20% lower mean Oracle error than Plain, and `Oracle mean + sample SD < Plain mean − sample SD` across five paired seeds. All-step error cannot replace the moving-coast score. Plain and Oracle remain 3×256 MLPs with train-only normalization; Oracle context is true mass plus combined friction. This gate has not been trained or evaluated.

**No further cubes or prediction models will run before your go.**

## Saved evidence

- [Timing](/home/ubuntu/faliure/clean/shove_gate/results/timing.json)
- [Episode metrics](/home/ubuntu/faliure/clean/shove_gate/results/pilot_metrics.json)
- [Independent audit](/home/ubuntu/faliure/clean/shove_gate/results/pilot_audit.json)
- [Action-probe results](/home/ubuntu/faliure/clean/shove_gate/results/action_probe.json)
- [Updated protocol](/home/ubuntu/faliure/clean/shove_gate/data/protocol.json)
- [Previous pilot](/home/ubuntu/faliure/clean/shove_gate/versions/v3/PILOT_REPORT.md)

Rebuild from saved runs only:

```bash
cd /home/ubuntu/faliure
PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.shove_gate.report
```

Input SHA-256 hashes:

- `results/timing.json`: `1c2e717085d5fdbbf8809861559ca9247c98f5068d421b2d2475695c9dda4fd2`
- `results/pilot_metrics.json`: `6baee0b8e4336451aa40c7d69dbf6b8ab4ab46a74361b9d15ebf2ebb3761b861`
- `results/pilot_audit.json`: `8f3dd2b1c6c000a72885a917cd1dd42009b91afd93e96cb6f3a4dad48fe711b8`
- `results/action_probe.json`: `4e1e42d961f9dcb63b5b085428ed6670d4d74b1db68a288c37f482236e792527`
- `data/protocol.json`: `63228bcf9f7f9722152da619f399aaf3f3fa819337f7c3b74bd1573d89753d8d`
- `data/plans.json`: `7c4952f9f2f5d996452632da34b9dae7c78c0d6a77fa4c7e89ea34f6955fc66c`
- `versions/v3/results/pilot_metrics.json`: `c3d50fede15173c83ed189c6056e016d544142266f133601f286b72d14d81f1e`
- `versions/v3/results/pilot_audit.json`: `06b83cced825f07db73a3f9fc1625eac1325619b53214df17d0b971e37729ddc`
