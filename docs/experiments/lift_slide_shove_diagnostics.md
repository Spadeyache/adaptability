# Diagnosing shove transfer

These checks examine the completed five-seed study without changing its trained models, stored beliefs, original targets, controller, or reported results. The original target distances were 5, 10 and 15 mm, with a 2.5 mm XY landing tolerance. Each push lasts 0.30 seconds. Candidate force commands range from 1 to 4 N in 0.25 N increments.

The evidence identifies inaccurate long rollouts as a major limitation. The belief improves one-step cube prediction, and force selection correctly minimizes the model's own predicted error. A real-simulator best-force search reaches many more targets than the model does. Some targets remain unreachable on this force grid, and the claimed force cap does not bound measured contact force.

## Task: best force in the real simulator

The search evaluated all 13 candidate forces on each of the 150 original setups: 1,950 distinct simulator trajectories. It reused the original 1,538 trajectories and generated the missing 412 in a separate diagnostic cache. Every trajectory matched its original measured starting state exactly, followed the specified commands, and reached rest. Original results and source fingerprints were unchanged.

The best force achieved **2.47 mm mean XY error and 62.7% success**, compared with **8.77 mm and 10.1%** for the combined-belief planner. This is a best-case reference using the true simulator after seeing every candidate's outcome, not a deployable policy.

| Original target | Setups | Best-force mean error | Best-force success |
| --- | ---: | ---: | ---: |
| 5 mm | 30 | 1.83 mm | 83.3% |
| 10 mm | 60 | 2.15 mm | 70.0% |
| 15 mm | 60 | 3.10 mm | 45.0% |
| Original mix | 150 | 2.47 mm | 62.7% |

There is substantial room for improvement through better action predictions. However, 56 of 150 setups have no successful candidate within the 2.5 mm tolerance. The best-case reference applies to this discrete grid; a denser or different action set could change it.

Moving targets to 20–60 mm would make this controller's task harder. An exploratory calculation using the completed simulator grid gives 36.7% best-force success at 20 mm, 3.3% at 40 mm, and 0% at 60 mm. Their mean errors are 4.04, 12.61 and 29.56 mm. These are offline reachability checks over all 150 setups, not new model-planned trials. No original targets were replaced.

[Oracle summary](../../runs/lift_slide_belief/three_task/diagnostics/oracle/summary.json) · [Per-setup results](../../runs/lift_slide_belief/three_task/diagnostics/oracle/oracle_cases.csv) · [Complete force grid](../../runs/lift_slide_belief/three_task/diagnostics/oracle/force_grid.csv)

![Simulator force response](../../runs/lift_slide_belief/three_task/diagnostics/oracle/force_response.png)

## Belief: one-step prediction

Every condition below predicts the same 839 moving shove transitions from the 30 held-out cubes. Scores average each cube's error, then average the five model seeds. The MSE is for normalized changes in all 43 observed state dimensions; it is not an error measured in millimeters.

| Belief | One-step shove MSE | Change against training average | Actual landing error | Landing success |
| --- | ---: | ---: | ---: | ---: |
| Training average | 1.0631 | Baseline | 9.00 mm | 9.1% |
| Five lifts | 1.0055 | 5.42% lower | 8.67 mm | 9.9% |
| Five slow slides | 1.0587 | 0.41% lower | 8.83 mm | 10.1% |
| Both | 1.0125 | 4.76% lower | 8.77 mm | 10.1% |
| Another cube's both | 1.1240 | 5.73% higher | 9.50 mm | 8.4% |
| Other five shoves | 1.0032 | 5.63% lower | 8.66 mm | 9.3% |

The combined belief improves one-step prediction against the training average and another cube's belief. Lifts alone have a slightly better mean score than both tasks together. This supports a modest cube-specific prediction benefit, while leaving the added contribution of slow slides unestablished.

A fresh frozen-model evaluation also separates cube motion from the robot joints and effort terms in the full-state loss:

| Belief | Normalized cube XY MSE | Normalized cube velocity MSE | Cube XY RMSE |
| --- | ---: | ---: | ---: |
| Training average | 1.0762 | 1.4306 | 0.461 mm |
| Five lifts | 0.7887 | 1.2538 | 0.396 mm |
| Five slow slides | 0.9119 | 1.4045 | 0.423 mm |
| Both | 0.7976 | 1.2731 | 0.398 mm |
| Another cube's both | 1.2264 | 1.5397 | 0.493 mm |
| Other five shoves | 0.7587 | 1.2414 | 0.389 mm |

Combining the tasks reduces normalized cube XY MSE by **25.9%** and velocity MSE by **11.0%** against the training average. Paired 95% cube-bootstrap intervals for the absolute MSE reductions are 0.176–0.388 and 0.066–0.282, respectively. The full-state MSE reduction is 0.0506, with an interval of 0.0159–0.0869. These descriptive intervals average the five seed scores within each cube, then resample the 30 whole cubes 10,000 times using seed 20261114. They reflect cube sampling uncertainty conditional on these five fitted models.

The physical XY RMSE is computed over the two coordinates and moving rows within each cube, then averaged over cubes and seeds. The fresh CPU calculation reproduced the original GPU full-state MSE within the recorded tolerance; its exact numerical discrepancy is retained in the diagnostic summary.

The linear physical-property readout is strongest for mass after lifting: mean test R² is 0.753 for lifts, 0.332 for slides and 0.728 for both. Friction readout R² is 0.137, −0.088 and 0.116 respectively. These are training-table-fitted linear readouts evaluated on inferred test beliefs. Weak linear readout does not establish that the latent contains no nonlinear information.

The shove-fitted condition and another cube's belief were present in the completed run and results CSV. They were omitted from the earlier short chat response. The shove-fitted condition is an adaptation reference, not a mathematical performance bound: it optimizes one-step prediction with the same frozen model and does not optimize target landing directly. Its poor landing score alone cannot show that no belief could help.

## Rollout: predicted distance versus actual rest

All 4,500 executed-push records are plotted below, including all six conditions and five seeds. Each prediction is paired with the actual execution of that same force from the same setup. The plotted distances are projections along the requested shove direction from the measured force-onset position. They do not include lateral error.

For the combined belief, signed-distance prediction has **r = 0.301**, **9.17 mm mean absolute error**, and a −1.48 mm mean bias. Including both horizontal coordinates gives 12.44 mm mean Euclidean disagreement between predicted and actual endpoints. The shove-fitted condition has r = 0.218 and 8.98 mm signed mean absolute error. Both plots have large scatter around the identity line.

All actual cubes reached rest. In contrast, **703 of 750 combined-belief forecasts (93.7%)** still predicted speed above 10 mm/s at the final horizon; median predicted terminal speed was 27.2 mm/s. These are fixed-horizon endpoint predictions, not reliable predictions of a stopping event. The earlier 716-of-750 count in the original study refers to the original recorded prove pushes; this diagnostic uses the planner's selected pushes.

These records are conditioned on the model's selected actions. Their correlation does not describe predictions over the entire candidate force grid. Repeated seeds and conditions share physical replays, so 4,500 records are not 4,500 independent experiments.

![Predicted versus actual shove distances](../../runs/lift_slide_belief/three_task/diagnostics/rollout/predicted_vs_actual_distance.png)

[Rollout and one-step summary](../../runs/lift_slide_belief/three_task/diagnostics/rollout/summary.json) · [Joined trial data](../../runs/lift_slide_belief/three_task/diagnostics/rollout/signed_distance_trials.npz) · [Vector figure](../../runs/lift_slide_belief/three_task/diagnostics/rollout/predicted_vs_actual_distance.pdf)

## Planner: choosing among predicted outcomes

Every one of the 4,500 chosen force indices equals the minimum of the model's 13 stored candidate errors. The selection code passes this check. For the combined belief, the chosen force matches the actual simulator-best force in only 12.3% of trials, and its average landing error exceeds the simulator reference by 6.30 mm. The corresponding gaps are 6.53 mm for the training average, 6.20 mm for lifts, 6.36 mm for slides, 7.03 mm for another cube's belief and 6.20 mm for the shove-fitted condition.

This distinguishes correct selection from accurate predictions: the planner chooses what its model says is best, but those rankings lead to poor physical outcomes. [All force-choice checks](../../runs/lift_slide_belief/three_task/diagnostics/oracle/chosen_force_audit.csv)

## Force cap: command limit versus contact measurement

The current controller bounds the requested horizontal hand wrench at 4 N and maps that wrench into joint torques. It has no measured-contact-force feedback. Vertical and orientation feedback also contribute joint torques. Arm position-drive stiffness and damping are zero during the pulse, then return to their native values during withdrawal. Finger drives remain active with a 30.06 N drive limit.

The saved contact channel needs a more precise label than the earlier report used: the installed CPU PhysX/SAPIEN/ManiSkill path reports the vector sum of **normal contact impulses divided by the 0.01 second physics step**. It excludes tangential friction impulses. Thus the quoted 7.08 N is the horizontal magnitude of that normal-contact signal, not a complete net contact-force measurement. A hard 4 N bound on actual total contact force was neither implemented nor verified.

Among the 1,538 original selected trajectories, 57 exceeded 4 N in this signal during the pulse, with 89 individual 10 ms samples exceeding the limit. Ten approach trajectories and twelve withdrawal trajectories also exceeded 4 N. The worst pulse reading was 7.084 N with a 3.75 N command, 30 ms after pulse onset. The largest withdrawal reading was 6.842 N. The readings after withdrawal were zero.

Isolated replays show that several effects contribute. In the worst pulse case, removing orientation feedback lowered the peak from 7.084 to 6.087 N; zeroing arm velocity at pulse entry lowered it to 6.504 N. Removing vertical feedback barely changed it (7.080 N), as did reducing the finger drive cap (7.017 N). Softer withdrawal drives reduced the worst withdrawal peak from 6.842 to 3.688 N. These are interventions on two selected examples, not a validated replacement controller or an additive decomposition of force.

The diagnostic verified the Jacobian uses the actual TCP link, with matching forward kinematics, and that gravity is disabled on all robot links. The issue is not explained by an incorrectly chosen Jacobian link or duplicated gravity compensation. The evidence instead shows that bounding one commanded wrench does not constrain all contact impulses generated by robot dynamics and the other drives.

[Force diagnostic summary](../../runs/lift_slide_belief/three_task/diagnostics/force_cap/summary.json) · [Archived phase measurements](../../runs/lift_slide_belief/three_task/diagnostics/force_cap/archived_forces.json) · [Controlled component replays](../../runs/lift_slide_belief/three_task/diagnostics/force_cap/component_interventions.json)

## Next experiment

A direct endpoint predictor trained only on training cubes is a reasonable next test: it can take the measured starting state, shove direction, force and inferred z, and predict the final displacement without recursively predicting dozens of intermediate robot states. Multi-step world-model training is another option. Neither repair has been trained in this diagnostic run.

For a physical force limit, first specify and measure the intended contact quantity, then calibrate a contact-feedback controller with slower contact entry and controlled withdrawal. A hard instantaneous guarantee needs simulator-level constraint enforcement or another independently verified method; feedback after a completed physics step alone cannot undo an impact. Changing the controller changes the transition distribution, so results for the revised force-limited task require fresh collection and evaluation. Clipping the logged force values would not fix the physics.

Larger targets are not supported by the present reachability results. Future task calibration should use training cubes or separate pilot cubes before committing a new held-out target protocol.

## What the checks establish

A real-simulator best-force search tests the available action choices. Its minimum error is a reference for this discrete force grid, pulse duration, controller and starting state; a continuous-force search or a different controller may do better.

For the rollout check, the model's endpoint at the recorded final horizon is compared with the actual resting location. Predicted terminal velocity must also be inspected before calling a forecast a predicted stop. Each comparison uses the force actually selected and executed for that same cube, setup, seed and belief condition.

Force selection is checked separately from model accuracy: a planner can correctly minimize the errors its model predicts while choosing a poor action in the simulator.

## Reproduce the checks

From the repository root:

```bash
source scripts/activate_grip.sh
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    python -m experiments.lift_slide_belief.oracle_diagnostics --workers 8
python -m experiments.lift_slide_belief.rollout_diagnostics \
    runs/lift_slide_belief/three_task --device cpu
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    python -m experiments.lift_slide_belief.force_cap_diagnostics \
    runs/lift_slide_belief/three_task
```

The oracle resumes only missing trajectories in its separate cache and verifies the original study fingerprints. The rollout diagnostic reloads the frozen checkpoints and saved beliefs, checks all trial joins, and recomputes the cube-only one-step metrics. The force diagnostic rechecks archived phase measurements and repeats the isolated controller interventions. The two focused oracle tests verify that lateral error affects the best-force choice and that passing through a target without resting cannot count as success.

[Original study and protocol](lift_slide_shove_belief.md) · [Original results CSV](../../runs/lift_slide_belief/three_task/study/results.csv)
