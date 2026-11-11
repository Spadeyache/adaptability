# Inferring hidden physics from Panda motion

[Experiment index](../experiments.md)

## Question

Can a world model trained on manipulation trajectories use a small context vector to predict how a particular robot and object will respond? The proposed deployment experiment asks whether repeated prediction errors can distinguish stable body properties from object properties that change at every reset.

## Repository and simulator audit

The frozen historical policy has 45 inputs and five actions. The 600-episode sweep passed it a fixed 40 g weight guess while recording actual cube mass, friction, COM, commands, robot/cube state, contacts and outcomes. Its single-factor and paired settings also vary visible geometry in some episodes. The new training set uses fresh trajectories and a fixed cube geometry. [Frozen checkpoint](../../runs/learning/20261009T045047-c852bfc9/final_ckpt.pt) · [Sweep manifest](../../runs/failure_collection/varied_fixed40g/manifest.json) · [Historical recorder](../../robot_adaptation/collection/varied.py)

The repository's collection wrapper builds an independent-physics PickCube scene. Its weighted variant inserts the reported weight and measured finger forces into the actor's input. The installed PushCube and PullCube tasks supply the two tabletop manipulation scenes. Starting poses use the tasks' seeded resets independently of the hidden constants. [Physics wrapper](../../robot_adaptation/simulation/environment.py) · [Weighted Pick wrapper](../../robot_adaptation/simulation/collection_lift.py) · [Frozen actor](../../robot_adaptation/policy/frozen_v1.py)

| Hidden property | Per-episode implementation verified in CPU PhysX |
| --- | --- |
| Cube mass | Set the body mass and scale uniform-cube inertia accordingly. Sample log-uniformly from 0.02–1.0 kg. |
| Cube friction | Set and read both material coefficients to the same draw in 0.1–1.5. |
| Table friction | Set and read both coefficients on table collision shapes in 0.2–1.2. |
| Gripper force scale | Scale both commanded native finger drive caps by 0.3–1.5×, relative to the policy's 0.25–40 N range. |
| Arm damping | Scale the seven Panda joint drive damping coefficients by 0.5–2.0×. The default velocity-feedback coefficient is 100. |
| Arm stiffness | Scale the seven drive stiffnesses by 0.5–1.5×. The default proportional gain is 1,000. |
| Action delay | Delay each issued five-channel command by 0–3 control steps. The queue is cleared at reset. |
| Cube COM | Set the inertial-frame origin within ±1 cm along each of three object-frame axes. This represents synthetic independent mass properties. |

All eight groups were available. The encoder receives ten scalars because COM contributes three. A 25-step mass estimator and a scalar filter exist from the earlier fixed-cube study. This experiment trains a new dynamics model with its own splits and loss. [Earlier mass study](mass_belief.md) · [Physics and control implementation](../../robot_adaptation/simulation/context_dynamics.py) · [ManiSkill controller conventions](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html)

## Data and model

The audited dataset contains two million transitions across 40,000 independent 50-step episodes. Each of Pick, Push, Pull and exploratory poke contributes 10,000 episodes. Exactly 36,000 episodes train the model and 4,000 are held out. Random command segments occurred in 10,002 episodes. The cube moved over 1 cm in 37,915 episodes. The observable state contains joint positions and velocities, TCP pose, gripper width, cube pose, linear velocity and angular velocity. Only the issued command enters the dynamics model. Field names, ranges, validation membership and command-queue checks are recorded with the data. [Audit](../../runs/context_dynamics/nov13/data/audit.json) · [Dataset schema](../../runs/context_dynamics/nov13/data/dataset.json)

The encoder and four-layer dynamics MLP train jointly using only multi-step state prediction. The initial model uses eight context coordinates. Its best five-step loss on the 2,000 checkpoint-selection episodes was 0.530 against 1.332 for state persistence. The remaining validation episodes are reserved for collapse checks. [Training history](../../runs/context_dynamics/nov13/z8/training.jsonl) · [Reproduction instructions](../../experiments/context_dynamics/README.md)

The initial observable state alone had held-out linear R² between −0.0020 and +0.0007 across the ten constants. A matched intervention also changed all physical settings while preserving the observable start in the controller regression. These checks address direct leakage at reset. [State audit](../../runs/context_dynamics/nov13/data/audit.json) · [Regression checks](../../tests/test_context_dynamics.py)

## Results for the Nov 13 review

After the eight-coordinate baseline, two revisions used low-force grasp probes and then matched physical pairs. The final four-coordinate model used 48,600 training episodes and ten-step forecasts. All architectures trained only on future-state error. A preliminary supplement used an inconsistent scene selector. The corrected Pick-scene supplement passed start-state replay checks. Its predecessor and the model trained on it are excluded from the reported results.

| Confirmation on 1,000 fresh episodes | Result |
| --- | ---: |
| Normalized ten-step loss with encoded constants | 0.367 |
| Loss with context from another same-task episode | 0.747 (+103.3%) |
| Loss with training-mean context | 0.589 (+60.4%) |
| Loss with zero context | 1.833 (+399.0%) |
| Passing axes on designated fresh Pick grasp | 3 of 4 |
| Episodes passing every axis | 545 of 1,000 |
| Pick episodes passing every axis | 112 of 250 |

Across the confirmation set, the average loss curve for each coordinate had its minimum at the encoder value. On the designated grasp, the third coordinate's minimum was 0.75 training SD away. The preregistered tolerance was 0.5 SD. The matched-pair repair did recover all four coordinates on the earlier failed grasp. [Full comparison, probe R² for each constant and physical replays](../../runs/context_dynamics/nov13/SUMMARY.md) · [Supervisor figures](../../runs/context_dynamics/nov13/context_diagnostics.pdf)

The final linear probes gave R² of 0.983 for action delay, 0.727 for mass, 0.607 for arm damping and 0.092 for gripper strength. Direct physical replay held the start and all 50 commands fixed. Changing only the finger cap from 0.3× to 1.5× produced a median peak cube-position separation of 93.0 mm across 48 policy grasps. Repeating the identical setting in a fresh scene reproduced the cube trajectory exactly across these 48 starts. This distinguishes a physically influential property from one the model makes easy to infer. [Matched measurements](../../runs/context_dynamics/nov13/fresh_physics/paired_physics.csv)

## Decision and open question

The designated fresh sweep failed Step 4. Deployment gradient inference, Kalman tracking, and the 50-trial comparison remain unrun. The experiment supports a forecasting benefit from context. It has yet to establish confidence about stable body properties across grasps.

A targeted next diagnostic would replay the same Pick commands with zero and three steps of delay. The designated grasp had a three-step delay. Within the fresh Pick episodes, 33 of 59 zero-delay grasps and 19 of 69 three-step-delay grasps passed all coordinate sweeps. These are different episodes, so the subgroup counts alone do not isolate an effect of delay.

## PickCube GPU collection

A subsequent ManiSkill collection produced 2,000,000 transitions from 40,000 randomized PickCube episodes on the RTX 3050. It provides 20,000 noisy-policy episodes, 10,000 scripted pushes and 10,000 random pokes. All three action sources appear across all 64 coarse mass, cube-friction and gripper-strength combinations in both the training and held-out episodes. The earlier reported model scores use the four-task dataset described above. [PickCube data and quality checks](../../runs/context_dynamics/pick_gpu_2m/DATASET.md) · [Loading and reproduction](../../experiments/context_dynamics/README.md#gpu-pickcube-trajectories)

On these PickCube episodes, a five-step model with eight context dimensions reduced mean held-out MSE from 1.3964 to 1.3069 against a separately trained zero-context baseline. Swapping contexts raised error by 45.6%. The training mean raised it by 25.3%. Five of six preselected episodes minimized interpolation error at their own encoded context. The remaining scripted push minimized error 10% toward its donor. Training stopped at the original model pair. Both fits and the baseline comparison ran on the RTX 3050. The frozen collapse checks ran on CPU. [Results, figures and proposed follow-up](../../runs/context_dynamics/pick_gpu_2m/world_model_z8/RESULTS.md)
