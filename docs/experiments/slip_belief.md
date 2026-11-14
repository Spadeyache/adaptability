# Grip variation and observable slip

This follow-up tests whether varying grip strength makes cube-specific mass and friction easier to infer. It compares fixed and varied grip strength with the same added measurements, so a difference between the two arms can be attributed to the action change rather than a different observation vector.

## Results

All ten models completed and passed the unchanged own-z versus shuffled-z validation check. Every result below includes all five paired model seeds and all 30 test cubes. The changed grip schedule created clean slip events, but it **did not establish better friction recovery and made the mass readout worse**.

The linear decoders are fitted only on the 200 training-cube latent vectors. Higher R² and lower mean absolute error are better.

| Experience used for z | Grip schedule | Mass R² | Mass MAE | Friction R² | Friction MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| Five lifts | Fixed | 0.755 | 51.7 g | 0.205 | 0.1364 |
| Five lifts | Varied | 0.423 | 80.1 g | 0.224 | 0.1357 |
| Five slow slides | Fixed | 0.307 | 87.3 g | −0.023 | 0.1605 |
| Five slow slides | Varied | 0.329 | 89.4 g | −0.026 | 0.1545 |
| Five lifts + five slides | Fixed | 0.745 | 52.5 g | 0.173 | 0.1383 |
| Five lifts + five slides | Varied | 0.437 | 79.1 g | 0.239 | 0.1343 |

For the combined belief, the friction R² gain was 0.066 with a paired 95% interval of −0.171 to 0.305. Its friction MAE reduction was 0.0040, with an interval of −0.0174 to 0.0261. Neither establishes an improvement. Mass MAE increased by 26.6 g, with an interval of 15.1–38.6 g; mass R² fell by 0.308, with an interval of −0.496 to −0.177. Lift-only mass readout also deteriorated, while its friction change was inconclusive.

These are descriptive paired bootstraps of 30 whole cubes, keeping their five seed results together, with 2,000 draws and bootstrap seed 7741. They describe cube sampling uncertainty conditional on the five trained seeds. The matched fixed-grip arm has the same 51-dimensional observations as the varied-grip arm; the earlier 43-state study is historical context rather than the experimental control.

![Physical-property readouts for the matched grip comparison](../../runs/lift_slide_belief/slip_probe/paired/study/readout_comparison.png)

[Readout table](../../runs/lift_slide_belief/slip_probe/paired/study/readout_table.csv) · [Paired changes and intervals](../../runs/lift_slide_belief/slip_probe/paired/study/paired_readout_comparison.csv) · [Per-cube predictions](../../runs/lift_slide_belief/slip_probe/paired/study/readout_predictions.csv)

## Does a belief help on the same task?

Yes for the fixed-grip lift test; the new varied-grip condition did not improve on its training-average baseline. These checks fit z from five episodes and predict the other five episodes of the same task, with the world model frozen. The primary metric includes all active-phase rows, including stationary outcomes.

| Grip schedule | Task | Mean-z normalized MSE | Own-z normalized MSE | Own-z change | Cube XYZ RMSE, mean → own |
| --- | --- | ---: | ---: | --- | --- |
| Fixed | Lift | 0.10127 | 0.08956 | 11.6% lower; paired interval favors own z | 0.1728 → 0.1630 mm |
| Fixed | Slow slide | 0.04859 | 0.04845 | 0.3% lower; inconclusive | 0.1699 → 0.1658 mm |
| Varied | Lift | 0.24195 | 0.24669 | 2.0% higher; paired interval favors mean z | 0.5693 → 0.5688 mm |
| Varied | Slow slide | 0.02693 | 0.02719 | 0.9% higher; inconclusive | 0.1730 → 0.1726 mm |

The fixed-grip lift's full-state MSE reduction was 0.01171, with a paired 95% interval of 0.00439–0.02324. It also beat another cube's lift belief by 21.1%. Its cube XYZ RMSE reduction was 0.00985 mm, with an interval of 0.00421–0.01665 mm. Thus the earlier successful lift data did support a useful same-task belief.

The varied-grip lift's full-state MSE increased by 0.00474 against the training mean, with an interval of 0.00154–0.00899. Its tiny physical XYZ gain was inconclusive: the reduction interval was −0.00366 to 0.00415 mm. The slide own-versus-mean comparisons were also inconclusive in both arms. Several comparisons with another cube's belief favor the correct cube, but that does not establish an improvement over the training mean.

Each arm has different lift and slide actions and resulting trajectories. Absolute same-task errors therefore should not be used as a direct comparison of model quality between arms. The relevant same-task test is own versus mean or wrong z on exactly the same held-out rows within an arm. Normalized losses also use each arm's own training scales. Active-moving-only results are retained as secondary checks.

[Same-task summaries](../../runs/lift_slide_belief/slip_probe/paired/study/same_task_summary.csv) · [Paired same-task comparisons](../../runs/lift_slide_belief/slip_probe/paired/study/same_task_comparisons.csv) · [All 4,800 same-task scores](../../runs/lift_slide_belief/slip_probe/paired/study/same_task.csv)

## Shared-shove prediction and verification

Both arms predict identical held-out shove trajectories. With the combined belief, physical one-step cube XY RMSE was 0.553 mm for fixed grip and 0.564 mm for varied grip; velocity RMSE was 18.14 and 18.95 mm/s. The point estimates do not show an overall advantage for varied grip. These physical metrics use Euclidean component RMS per cube, then average cubes and seeds. Normalized shove MSE is retained for within-arm comparisons only.

There are 900 unique readout predictions, 1,800 shove scores and 4,800 same-task scores. All ten checkpoint-and-belief reload checks reproduce the stored results exactly on the same GPU arithmetic setting. No seeds or test cubes were removed. This diagnostic did not rerun target-distance control or change the long-rollout model.

An independent audit recomputed the tables, confidence intervals and 26,400 frozen-model scores. CPU replay differed from the saved GPU scores by at most 0.315% relatively; the largest cube XYZ RMSE difference was 0.000825 mm. A targeted check reproduced the saved score exactly using the original GPU `high` matrix-precision setting, while GPU `highest` agreed with CPU arithmetic. This numerical difference is documented without modifying the original scores. The exact GPU checkpoint-and-bank reload differences remain zero. Eleven focused tests cover measurement timing, event classification, support/prove isolation and metric computation.

[Shove scores](../../runs/lift_slide_belief/slip_probe/paired/study/forecast_summary.csv) · [Reload verification](../../runs/lift_slide_belief/slip_probe/paired/study/retrieval_check.json) · [Independent result audit](../../runs/lift_slide_belief/slip_probe/paired/study/independent_results_audit.json) · [Complete study summary](../../runs/lift_slide_belief/slip_probe/paired/study/summary.json)

## Interpretation

Slip is a useful source of information, but it is not necessary for every physical estimate. The previous successful lifts already supported a mass readout with test R² around 0.75, and their joint-effort measurements provide a signal even when position tracking is good. Friction readout was weak. A small tracking error does not mean the forces required to produce the motion carry no information.

The minimum successful grip force mixes mass, acceleration, contact friction and grasp geometry. In an ideal symmetric static pinch, two normal forces N support weight when 2μN is at least mg. An observed threshold alone therefore cannot uniquely identify both mass and friction. Grip drive-force limits are also actuator settings, not direct measurements of the contact normal forces.

The simulator varies the cube material's friction parameter while holding finger and table materials fixed. The default PhysX combination rule averages the two contacting materials, so cube–finger and cube–table contact coefficients differ. This is an idealized interpretation of the configured materials; the experiment checks observed behavior directly. [PhysX material documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/class_px_material.html) · [Combination rules](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/struct_px_combine_mode.html)

The existing slow-slide script pushes with closed fingers; it does not pinch the cube between them. Randomizing the finger drive cap need not produce controlled grip slip in that task. All centers of mass remain at the cube center, so this study tests mass and friction rather than variation in balance.

## Comparison

The paired study retains 200 training cubes, 30 test cubes, ten episodes of each of the three tasks, and five model seeds. Physics and motion samples are matched between arms. Both arms receive the original 43 observed state dimensions plus cube-relative position, cube-relative velocity and finger target-position error, for 51 dimensions total. Actual issued grip commands remain in the action vector. Physical labels are reserved for final readout checks.

The fixed arm uses the previous 30.0625 N per-finger drive setting. The varied arm uses that same setting through pickup, then changes to its sampled cap at control step 45, 2.25 seconds into the episode. Lift motion starts at step 30. This fixed-time switch is independent of mass, friction and whether a particular cube has slipped. The lift and slow-slide motion scripts are otherwise unchanged. Shove commands remain the same.

The varied arm spans 0.25–30.0625 N with five equal-width bins in log force. Each five-episode block samples one cap per bin and randomly orders them, independently of cube properties. The first five adaptation episodes therefore cover weak through firm settings. Episodes 5–9 use an independent repeat. Both arms retain every outcome, including failed pickups and rotating losses.

Both arms use the same three-layer, 256-unit world model, four-dimensional latent, optimization settings, episode split, inference prior and adaptation budget. The existing own-latent versus shuffled-latent validation check remains unchanged. Failed seeds are reported and not silently replaced; held-out comparisons use seeds that pass in both arms.

The requested readout table compares lifts only, slides only and both, reporting mass and friction R² and mean absolute error. The six-condition one-step shove errors provide a separate check of whether inferred beliefs predict held-out motion. This follow-up does not assume that more informative slip observations will also repair the previously inaccurate long rollouts.

The same-task check fits each belief on episodes 0–4 and predicts the other five episodes (5–9) of the same task. It compares the training mean, this cube's same-task belief, another cube's same-task belief and the combined belief. Scores cover the active phase, with active moving rows reported separately. Physical XYZ prediction error includes the vertical motion needed to assess lifts. The prediction episodes never enter latent fitting or model selection.

## Pilot before collection

The initial pilot used a constant cap throughout pickup and motion. It covered masses 40 g and 500 g, cube friction parameters 0.4 and 1.0, eight grip caps from 0.25 to 30 N, and three matched reset seeds, for 96 lifts and 96 slow pushes. Forty-two lifts failed to establish an airborne grasp. Two apparent acquired-grasp slip events already involved substantial rotation at event onset. This pilot established pickup/hold thresholds, but not the intended clean slip signal.

The second pilot kept a firm grasp until step 45 before applying the same cap sweep. Under the final audited measurement and classification definitions, all 96 lifts acquired the object; 35 had translational slip after acquisition, 52 held, and nine had rotation-involved losses. Rotation through slip onset and the following control step is checked separately from tumbling later in a fall.

| Matched comparison | Grip cap after pickup | Result across three reset seeds |
| --- | ---: | --- |
| 40 g versus 500 g, at either friction value | 1 N per finger | Light cube held; heavy cube slipped in all three pairs |
| Friction 0.4 versus 1.0, 500 g cube | 8 N per finger | Low friction: two translational slips and one rotating loss; high friction: three holds |
| Friction 0.4 versus 1.0, 40 g cube | 0.5 N per finger | Low friction: two translational slips and one rotating loss; high friction: three holds |

These are controlled pilot examples demonstrating a usable signal, not estimates of generalization accuracy. The full readout comparison uses the separately sampled training and test cubes. Slow pushes never established bilateral finger grasps in the first pilot; their response to changing finger cap must not be described as pinch slip.

[Constant-cap pilot](../../runs/lift_slide_belief/slip_probe/pilot/summary.json) · [Firm-pickup then weaker-grip pilot](../../runs/lift_slide_belief/slip_probe/pilot_step45/summary.json)

![Matched mass and friction pilot traces](../../runs/lift_slide_belief/slip_probe/pilot_checks.png)

## Measurement checks before training

The new gap-velocity feature is the causal backward difference of consecutive TCP-frame cube positions, divided by the 0.05 second control interval. Its initial value is zero. It uses no future observations. The analytic instantaneous relative velocity is retained in separate diagnostic fields.

This definition followed a pilot check: the Jacobian-derived gripper velocity agreed with the native link velocity, but the native cube velocity sampled at the control boundary disagreed with observed control-interval cube motion during a stable hold. That instantaneous quantity therefore could not be treated as a direct measurement of the interval's slip. The mathematical rotation correction was checked independently. The pilot side fields were recomputed from saved poses, preserving the simulated states and actions.

An initial finger-target shape error was also corrected before collection. Each finger target is now scalar, and its target-minus-position error has exactly two coordinates. Acquisition requires three consecutive airborne, bilateral-contact observations with less than 1 mm relative-position change per control step. A slip label additionally requires at least 5 mm downward gap change with interval downward relative speed above 10 mm/s; rotation through onset and the next step must remain within 20 degrees. Rotating losses and failed pickups remain distinct outcomes.

The earlier provisional count of 42 slips was revised to 35 after these measurement and rotation checks. These annotations do not enter model inputs, latent fitting or readout targets. [Measurement revision and preserved-physics audit](../../runs/lift_slide_belief/slip_probe/pilot_step45/measurement_revision.json) · [Independent pilot audit](../../runs/lift_slide_belief/slip_probe/independent_pilot_audit.json)

## Full collection

Collection completed 200 training and 30 test cubes in each arm, with 6,900 episodes and 966,000 transitions per arm. Sharing unchanged shoves reduced the number of distinct simulator episodes to 11,500. Each arm contains ten lifts, ten slow pushes and ten shoves per cube.

| Lift outcome | Fixed grip | Varied grip |
| --- | ---: | ---: |
| Held after acquisition | 2,280 | 1,084 |
| Clean slip after acquisition | 0 | 980 |
| Rotation-involved loss | 20 | 236 |
| Failed to acquire | 0 | 0 |

All lifts established a grasp before the grip switch: the latest acquisition was step 39, before step 45. In the first five varied-grip learning lifts, 195/200 training cubes and 28/30 test cubes had both a hold and a clean slip. Test cube 203 had four holds and one rotating loss; cube 224 had three clean slips and two rotating losses. Both remain included in all evaluation denominators.

The two independent audits verified every paired cube. All original fixed-arm fields match the previous study exactly; shared shove observations and all pre-switch observations match between arms. Gap features are aligned, causal and finite, with the declared force-bin coverage and actual drive-cap settings. All 2,300 shared shoves reached rest.

[Collection audit](../../runs/lift_slide_belief/slip_probe/paired/data_audit.json) · [Independent full-data audit](../../runs/lift_slide_belief/slip_probe/paired/independent_data_audit.json)

## Limits of the interpretation

The force settings are sampled in advance. This is randomized probing; a policy that actively chooses the next action to reduce belief uncertainty is a separate experiment. Finger-contact friction and table-contact friction are linked through one shared cube material parameter in this simulator. Inferring that parameter from a grasp does not establish that arbitrary real-world surface pairs have the same friction.

## Reproduce

From the repository root, use a fresh output directory when changing the declared protocol:

```bash
source scripts/activate_grip.sh
probe=runs/lift_slide_belief/slip_probe
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    python -m experiments.lift_slide_belief.slip_data "$probe/paired" \
    --collect-pair --workers 8
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    python -m experiments.lift_slide_belief.slip_study \
    "$probe/paired/fixed/data" "$probe/paired/random/data" \
    "$probe/paired/study" --device cuda
```

Collection resumes missing cube pairs. Training resumes checkpoints only under the same saved dataset, source and optimization fingerprints. The study retains per-seed validation outcomes, final cube-ID belief banks, retrieval checks, physical-property predictions, one-step shove scores and same-task held-out predictions. Same-task absolute errors are interpreted within each acquisition arm because their lift and slide trajectories differ. The shared shove trajectories and physical-property labels support comparisons between arms.

[Previous belief study](lift_slide_shove_belief.md) · [Previous diagnostic report](lift_slide_shove_diagnostics.md)
