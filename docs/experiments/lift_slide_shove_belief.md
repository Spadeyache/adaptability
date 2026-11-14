# Learning a cube belief from lifts and slow slides

This study combines two sources of interaction data before predicting a third task. Scripted lifts vary speed and height. Scripted slow slides vary speed and direction. A shove issues a bounded horizontal robot-force command for a fixed duration, withdraws the hand, and allows the cube to coast to rest. The mass and friction ranges, simulator, 200 training cubes, 30 test cubes, and five model seeds follow the earlier lift-to-slide study.

The protocol tests whether the combined observations identify a useful cube-specific belief. Slow-slide resistance depends on both mass and friction, so the lift-only, slide-only and combined conditions measure what each experience actually contributes.

The [follow-up diagnostic report](lift_slide_shove_diagnostics.md) includes the real-simulator best-force reference, cube-only one-step errors, predicted-versus-actual distance plots, planner checks and a detailed force-cap audit. It also corrects the contact measurement's interpretation: the logged signal is derived from normal contact impulses and excludes tangential friction.

## Results

All five model seeds completed. They passed the requested own-z versus shuffled-z check with 7.92–12.41% lower moving validation error and positive paired confidence bounds. Only two seeds had a clear advantage over the training-average vector on validation.

The desired-distance test executed each model’s selected push in the simulator. There were 4,500 cube/episode/seed/condition outcomes, supplied by 1,538 distinct simulator replays when different conditions selected identical actions. Every actual replay started at its matched recorded state and reached rest. All candidate model rollouts remained finite, so the fallback force was unused. An independent audit verified every selected command sequence, starting state, endpoint, success flag and aggregate landing score. [Target-control audit](../../runs/lift_slide_belief/three_task/study/target_control/audit.json)

| Belief | One-step shove MSE | Forecast endpoint error | Actual target error | Landings within 2.5 mm |
| --- | ---: | ---: | ---: | ---: |
| Training average | 1.0631 | 33.77 mm | 9.00 mm | 9.1% |
| Five lifts | 1.0055 | 33.41 mm | 8.67 mm | 9.9% |
| Five slow slides | 1.0587 | 34.09 mm | 8.83 mm | 10.1% |
| Five lifts and five slow slides | 1.0125 | 32.64 mm | 8.77 mm | 10.1% |
| Another cube’s lifts and slides | 1.1240 | 33.81 mm | 9.50 mm | 8.4% |
| Other five shoves | 1.0032 | 37.35 mm | 8.66 mm | 9.3% |

The combined belief reduced one-step prediction error by 4.76% against the training average. Its paired mean reduction was 0.0506, with a 95% interval of 0.0157–0.0867. It also beat the other cube’s belief on that metric. Five lifts alone gave slightly lower mean error than the combined belief.

Actual target error fell by 0.23 mm against the training average, with a 95% interval of −0.20 to 0.63 mm. The success-rate gain was 1.07 percentage points, with an interval of −1.33 to 3.47 points. The combined belief’s actual target error was 0.10 mm higher than lifts alone, with the paired interval spanning zero. Its advantage over another cube’s combined belief was 0.73 mm, with an interval of 0.24–1.24 mm. These results show a small cube-specific effect while leaving the benefit of combining tasks for target landing unestablished.

Long forecasts remained inaccurate. The combined belief’s 1.13 mm endpoint improvement over the training average had a 95% interval of −3.30 to 5.71 mm. In 716 of 750 combined-belief forecasts, the model still predicted cube speed above 0.01 m/s at the time the observed cube was resting. The reported forecast endpoint is the predicted position at that fixed horizon.

| Experience used for z | Mass readout R² | Mass MAE | Friction readout R² | Friction MAE |
| --- | ---: | ---: | ---: | ---: |
| Five lifts | 0.753 | 52.1 g | 0.137 | 0.145 |
| Five slow slides | 0.332 | 82.2 g | −0.088 | 0.164 |
| Both | 0.728 | 55.1 g | 0.116 | 0.146 |

Mass information was strongest after lifts. The slow-slide belief’s friction readout was weak, and adding slides to lifts gave no clear improvement. The formation curve shows modest one-step prediction gains as lift observations accumulate, with small oscillations as slow slides are interleaved.

The confidence intervals above are descriptive post-run paired bootstraps. They first average the five seed scores within each cube, then use 10,000 shared resamples of the 30 whole cubes with random seed 5801. Figure error bars show one standard error across the five model means. All 3,450 result rows and all attempted landings are retained.

A fresh GPU process reloaded all five world models and the saved beliefs, using only prove shoves. All 450 cube/seed/learning-condition combinations reproduced both prediction metrics within the recorded tolerance. [Retrieval verification](../../runs/lift_slide_belief/three_task/study/retrieval_check.json)

[Results CSV](../../runs/lift_slide_belief/three_task/study/results.csv) · [Transfer figure](../../runs/lift_slide_belief/three_task/study/transfer.png) · [Formation figure](../../runs/lift_slide_belief/three_task/study/formation.png) · [Meaning figure](../../runs/lift_slide_belief/three_task/study/meaning.png) · [Detailed landing trials](../../runs/lift_slide_belief/three_task/study/target_control/trials.npz) · [Validation checks](../../runs/lift_slide_belief/three_task/study/validation.json)

## Collection checks

Collection completed 6,900 episodes and 966,000 transitions. Lifts and slow slides each last 120 control steps (6 seconds), and shoves last 180 steps (9 seconds, including approach and rest). Every lift rose over 2 cm and every slow slide moved over 2 cm. All 2,300 shoves made robot contact and satisfied the final rest criterion for 0.5 seconds. The rest check requires low linear and angular speed and negligible robot contact.

The matched-force check applied the same 4 N horizontal force command for 0.30 seconds to 40 g and 500 g cubes at friction 0.7. Across three matched directions and reset seeds, the light cube’s signed travel exceeded the heavy cube’s by 3.17 cm on average. Repeated light-cube endpoints matched exactly. Across the full collection, the maximum measured horizontal contact force during a pulse was 6.21 N. All post-withdrawal contact forces were zero. The median post-pulse coast distance was 0.55 mm, so most travel occurred during contact.

The 150 test prove shoves contain 839 moving pulse-and-coast rows, with 7–35 per cube. No test cube was excluded for remaining stationary. [Data and replay audit](../../runs/lift_slide_belief/three_task/data_audit.json) · [Matched-force confirmation](../../runs/lift_slide_belief/three_task/force_confirmation.json) · [Dataset schema](../../runs/lift_slide_belief/three_task/data/dataset.json)

## Protocol

Each cube has ten complete episodes of each task. Its mass and friction are sampled once and remain fixed for all 30 episodes. Training uses all three tasks, with two complete episodes per task per training cube held out for checkpoint selection and the own-z comparison. The five model seeds share the same cube samples, observations, and validation split.

The world model retains three hidden layers of 256 units. Its inputs are normalized observed state, actual issued action, and a four-dimensional cube vector. Its output is the normalized change in state. The world model and the 200-row latent table are trained together from prediction error, with the original Adam settings and latent-size penalty. Only fitting episodes determine normalization. Physical labels enter the final linear probes.

The validation gate requires the correct cube’s vector to reduce moving-step MSE by at least 1% against a shuffled cube vector, with a positive lower bound on the paired 95% cube-bootstrap interval. Comparison with the training average is also reported. Before test evaluation, an inherited extra requirement to beat the training average was removed to match the updated specification’s own-z versus shuffled-z check. This correction changes validation admission only. The training objective, selected checkpoints, numerical threshold and bootstrap procedure are the same.

For a new cube, inference starts at the mean of the 200 training vectors. It minimizes prediction MSE plus 0.0001 times the mean squared distance from that mean, divided by each latent coordinate’s training variance. A variance floor of 0.0001 avoids division by a nearly zero spread.

Each test cube supplies five lifts and five slow slides. The combined sequence alternates lift then slide, producing ten saved belief updates. Each cumulative prefix is fitted from the training mean with 100 Adam updates. This gives each prefix and each final condition the same optimization budget. The two conditions using one task each receive five episodes. The combined condition receives ten.

The first five shoves are the prove episodes. The other five fit a separate ceiling vector. Six conditions predict exactly the same prove transitions.

| Condition | Observations used to fit z |
| --- | --- |
| Training mean | None |
| Lifts only | Five lifts from this cube |
| Slow slides only | Five slow slides from this cube |
| Both | Five lifts and five slow slides from this cube |
| Another cube’s both | Five lifts and five slow slides from a different test cube |
| Shove ceiling | This cube’s other five shoves |

The one-step metric is normalized state-change MSE on moving steps from force onset through the recorded resting horizon. The stopping-position metric starts at the observed state at force onset, then feeds the model’s entire predicted state back into itself with the recorded commands. Its final cube XY position is compared with the observed resting position in meters. The rollout has no further observed-state resets. Predicted terminal speed is reported separately. Every finite endpoint error is retained. Nonfinite trajectories have explicit failure counts.

The training mean, coordinate spreads, and all inferred snapshots are saved. Retrieval reloads the cube ID’s vector and reruns the prove predictions. Each seed’s linear decoder fits mass and friction from its training latent table, then predicts those labels from the lift-only, slide-only and combined test vectors.

## Target-distance control

The landing test gives each belief condition the same measured state at force onset, push direction and target location. Across each cube’s five prove setups, target distances are 5, 10, 15, 10 and 15 mm along the shove direction. The planner predicts the complete outcome of 13 candidate forces, from 1 to 4 N in 0.25 N increments, each applied for 0.30 seconds. It chooses the finite prediction closest to the target. The withdrawal and holding commands are fixed before prediction.

After selection, fresh simulator episodes reproduce the original pre-pulse state and execute the chosen force. The resulting resting position supplies the actual target error. A successful landing requires the cube to stop within 2.5 mm of the target. All six conditions receive the same target and candidate force options. Selected physical trajectories shared across model seeds or conditions are simulated once and reused by their exact cube, setup and force key.

The planner receives observable initial state and candidate commands. Mass and friction are used to recreate the physical simulator for evaluation. If every candidate rollout becomes nonfinite, the controller executes the predefined 2.5 N fallback and records the planning failure. Finite predictions retain their complete errors, including implausible values. Success rates count failed, missed and unreachable outcomes in the denominator.

The 4 N cap applies to the commanded horizontal force at the robot’s hand. The reported horizontal normal-contact signal includes inertial impacts and can exceed that command: the peak during the selected target-control pushes was 7.08 N, compared with 6.21 N in the training and test collection. Its post-withdrawal values were zero. This channel excludes tangential friction impulses, and the controller does not enforce a measured-contact-force cap; see the follow-up audit linked above. The dataset varies shove magnitude uniformly between 1 and 4 N to provide training examples over the planner’s action range.

## Reproduce

From the repository root, activate the project environment and use a fresh output root for changed settings.

```bash
source scripts/activate_grip.sh
run=runs/lift_slide_belief/three_task
OPENBLAS_NUM_THREADS=1 python -m experiments.lift_slide_belief.collect "$run/data" --workers 8
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    python -m experiments.lift_slide_belief "$run/data" "$run/study" --device cuda
```

The collector resumes at missing cube files. Study checkpoints resume under the same recorded protocol. The aggregate results CSV contains per-seed, per-cube, per-condition and per-prefix scores. Three figures show the six-condition comparison, belief formation, and physical-property readouts.
