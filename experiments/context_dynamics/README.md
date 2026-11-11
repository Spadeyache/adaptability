# Learning hidden physical context from prediction errors

The unit of a split, intervention or bootstrap draw is a complete episode. The supervised targets are future states. Only the training encoder receives the raw simulator constants. The actor supplies motion and has frozen weights.

## GPU PickCube trajectories

The PickCube-only collection uses the frozen five-command Panda policy and ManiSkill CUDA PhysX. Each GPU batch builds 250 scenes with independently sampled cube mass, cube and table friction, gripper force scale, arm damping and stiffness, command delay, and three cube center-of-mass offsets. The collector reads the native properties again after reset. It spaces scenes 64 m apart to keep cubes that slide off a table clear of other environments.

Half the episodes use the policy with an independent log-uniform mass guess and Gaussian command noise. One quarter use a scripted push. The remaining quarter alternate reaching toward the cube and applying open-loop pokes with a sampled direction and magnitude. Each trajectory covers 50 control steps. The stock PickCube goal is randomized. The holdout contains complete episodes from the same physics and action distributions as training.

From the repo root, use the project environment. On WSL, expose the host CUDA libraries before collection.

```bash
source scripts/activate_grip.sh
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
run=runs/context_dynamics/pick_gpu_2m
python -m experiments.context_dynamics.collect_gpu "$run" --pilot
python -m experiments.context_dynamics.collect_gpu "$run"
python -m experiments.context_dynamics.verify_gpu "$run/data"
python -m experiments.context_dynamics.gpu_data_report "$run"
```

The pilot reopens its ten saved episodes and checks both simulator readbacks and cube travel. Each full-run shard contains 250 episodes. The training and validation masks select whole rows. Running the collector again with the same protocol verifies completed shards and resumes from the first missing batch.

```python
from pathlib import Path
import numpy as np

folder = Path("runs/context_dynamics/pick_gpu_2m/data")
with np.load(folder / "shard_0000.npz", allow_pickle=False) as shard:
    train = ~shard["is_validation"]
    states = shard["state"][train]
    state_now = states[:, :-1]           # (episodes, 50, 39)
    action = shard["command"][train]     # (episodes, 50, 5)
    state_next = states[:, 1:]           # (episodes, 50, 39)
    constants = shard["constants"][train]  # (episodes, 10), shared across the 50 steps
```

The 39 state values contain nine joint positions, nine joint velocities, a tool pose, gripper width, cube pose, and cube linear and angular velocity. Poses use scene-local positions in meters and wxyz quaternions. The first seven joints are revolute and the last two are prismatic. The five normalized commands control end-effector translation, finger opening and finger strength. Use `command` for the action issued at each step. The companion `applied` records the command received by the motors after the episode's 0–3 step delay. Both finger drive caps are logged in newtons on every step.

[Dataset guide](../../runs/context_dynamics/pick_gpu_2m/DATASET.md) · [Field order](../../runs/context_dynamics/pick_gpu_2m/data/dataset.json) · [Episode index](../../runs/context_dynamics/pick_gpu_2m/data/episodes.csv) · [Actual transition](../../runs/context_dynamics/pick_gpu_2m/transition.json) · [Collection figure](../../runs/context_dynamics/pick_gpu_2m/collection.png)

### Five-step context and zero-context comparison

The paired study fits an eight-dimensional encoder and a four-layer, 512-unit dynamics MLP. Its independently trained baseline uses the same initial shared weights, minibatches and update count. The baseline's context inputs are fixed at zero. The objective averages normalized prediction error equally over horizons 1–5. Both checkpoints use the fixed final update. All 4,000 held-out episodes remain available for the forecast comparison and frozen collapse checks.

```bash
run=runs/context_dynamics/pick_gpu_2m
study="$run/world_model_z8"
python -m experiments.context_dynamics.pick_train "$run/data" "$study"
python -m experiments.context_dynamics.pick_evaluate compare "$run/data" "$study"
python -m experiments.context_dynamics.pick_evaluate checks "$run/data" "$study"
```

Every comparison uses the same 46 starts per episode. Bootstrap draws resample complete episodes. Swaps use a seeded, distinct held-out donor of the same action type. The mean z comes from the training episodes. For interpolation, the first two held-out episodes per action source are evaluated along the straight line to their donors. The linear probe fits raw physics labels on training encodings and scores the held-out encodings. [Results and figures](../../runs/context_dynamics/pick_gpu_2m/world_model_z8/RESULTS.md)

## Collection

Activate the project environment with `source scripts/activate_grip.sh`. Use a new output root for a different specification.

```bash
python -m experiments.context_dynamics.collect runs/context_dynamics/nov13/data --workers 9
python -m experiments.context_dynamics.audit runs/context_dynamics/nov13/data
```

The four equally sampled motion sources are the repo's weighted PickCube-v1 variant, the installed PushCube-v1 and PullCube-v1 tasks, and a poke/grasp/shake script in the pick scene. All have the same Panda, cube dimensions, appearance and five normalized commands. Pick uses an independent random log-uniform mass guess with additive Gaussian command noise. Push and pull use noisy position waypoints. A random subset adds piecewise-random commands. The collector uses the stock CPU Pinocchio IK and PhysX drives, skips reward and rendering, and checks per-episode physical readbacks. A regression compares a physical control step to the original ManiSkill controller. Here arm damping is the velocity term of the joint drives. Finger strength scales the command-dependent per-finger force limit. Delay starts each episode with a zero-filled command queue.

Each atomic shard stores episode IDs, task IDs, reset seeds, the ten raw constants, policy guesses, random segment bounds, 51 states and 50 issued and applied actions. Transition t is (state[t], command[t], state[t+1]); episode constants, IDs and task are joined using the same shard row. The world model reads only the state and issued commands. The delay queue is unobserved, so a memoryless model cannot exactly reconstruct the delayed action at every possible transition. The cube and TCP poses use world-frame position and wxyz quaternion. Training converts orientation to two rotation matrix columns to remove quaternion sign jumps. Linear/angular cube velocities are in world axes.

## Training and checks

```bash
python -m experiments.context_dynamics.train runs/context_dynamics/nov13/data runs/context_dynamics/nov13/z8
python -m experiments.context_dynamics.check runs/context_dynamics/nov13/data runs/context_dynamics/nov13/z8 --fold 0
```

The encoder maps ten scalar constants (eight groups, with three COM coordinates) to eight learned values. Mass is log-transformed before training-only standardization. The world model is a four-hidden-layer MLP, width 256. It predicts normalized state increments from the current state, issued action and context. Five-step rollouts use observed commands and feed each predicted state back. The loss averages squared state errors in training delta-SD units, weighted by inverse forecast horizon. Episode ID, task, weight guess, actual delayed action and true constants are excluded from its inputs.

Of the 10% held-out episodes, half select training checkpoints. The remainder form two disjoint collapse-check folds. Within each fold, swap contexts only among episodes of the same task and use identical windows for all variants. Before running the checks, the following criteria were set:

- The encoded-context model must beat persistence. Swapping, replacing with the training mean and replacing with zero must each increase held-out rollout loss by at least 1%, with a positive lower bound on the paired 95% episode-bootstrap difference.
- For the first held-out Pick episode in the fold, sweep each coordinate individually from −2 to +2 training standard deviations in quarter-SD increments. At the encoded value the loss must undercut both ±1 SD neighbors and a grid minimum must lie within half an SD of the encoded value for **every dimension**. Also report the population-average sweep. This individual-episode criterion can fail even if context improves average forecasts.
- Fit linear readout probes on training encodings, score each physical constant on held-out encodings and report all scores. These measure linear information in the privileged encoding. A low score alone cannot establish that motion contains no information about that property.
- Advance to deployment inference only with passing checks. If revised, document the revision and evaluate on fresh held-out episodes. Interpreting confidence per coordinate also depends on whether learned coordinates separate fixed and varying factors.

## Repair and physical replay

The eight-coordinate result motivated separate checks of the motion signal and the model's encoding. The added dataset uses 6,000 fresh exploratory grasps with changes to the commanded force cap during lift and shake. A four-coordinate encoder with a wider dynamics model predicts ten steps. The original checkpoint selection split is retained. The second collapse-check fold was held back for this attempt. All commands below run from the repository root with the project environment active. For CUDA on WSL, make the host CUDA libraries visible via `LD_LIBRARY_PATH=/usr/lib/wsl/lib`.

```bash
python -m experiments.context_dynamics.collect runs/context_dynamics/nov13/grip_pulses --config configs/context_grip_probe.json --workers 9
python -m experiments.context_dynamics.audit runs/context_dynamics/nov13/grip_pulses
python -m experiments.context_dynamics.train runs/context_dynamics/nov13/data runs/context_dynamics/nov13/z4_grip --extra-training runs/context_dynamics/nov13/grip_pulses --latent 4 --width 384 --horizon 10 --updates 18000 --seed 9102
python -m experiments.context_dynamics.check runs/context_dynamics/nov13/data runs/context_dynamics/nov13/z4_grip --fold 1
python -m experiments.context_dynamics.physics_check runs/context_dynamics/nov13/data runs/context_dynamics/nov13/grip_pulses runs/context_dynamics/nov13/fresh_physics
python -m experiments.context_dynamics.report runs/context_dynamics/nov13 runs/context_dynamics/nov13/z8/checks_0 runs/context_dynamics/nov13/z4_grip/checks_1
```

In the matched physical replay, each setting gets an independently built scene. Each sequence replays identical commands twice at the low setting and once at the high setting. This checks the effect of changing the hidden property against same-setting simulator repeatability.

## Matched-pair repair and confirmation

The four-coordinate model recovered three coordinates on the designated Pick episode; the remaining sweep minimum was 0.75 SD from the encoder value. For the final repair, 4,000 new command sequences each produce two fresh-scene trajectories from the same initial state with one changed physical constant. Both members of each pair always belong to the same split. Training adds 7,200 paired episodes. Forecast loss gives equal weight to nine observed state fields so gripper width contributes comparably to joint velocities. All selection and collapse checks retain the original unweighted forecast metric. The model has four coordinates, width 512, a ten-step horizon and 24,000 optimizer updates. We retest the previously failed case (156) as a regression check. Because that case informed the repair, the first Pick episode of **1,000 fresh episodes using seed 930761** is the untouched case for the requested single-episode criterion. The same numerical thresholds apply, and both results are reported. No confirmation trajectory is used for fitting or checkpoint selection.

```bash
python -m experiments.context_dynamics.collect runs/context_dynamics/nov13/matched_train --config configs/context_matched_train.json --workers 8
python -m experiments.context_dynamics.audit runs/context_dynamics/nov13/matched_train
python -m experiments.context_dynamics.collect runs/context_dynamics/nov13/confirmation --config configs/context_confirmation.json --workers 2
python -m experiments.context_dynamics.train runs/context_dynamics/nov13/data runs/context_dynamics/nov13/z4_matched --extra-training runs/context_dynamics/nov13/grip_pulses --extra-training runs/context_dynamics/nov13/matched_train --latent 4 --width 512 --horizon 10 --updates 24000 --seed 9103 --balance-state
python -m experiments.context_dynamics.check runs/context_dynamics/nov13/data runs/context_dynamics/nov13/z4_matched --fold 1
python -m experiments.context_dynamics.check runs/context_dynamics/nov13/data runs/context_dynamics/nov13/z4_matched --external runs/context_dynamics/nov13/confirmation
```

Physics replays and matched training pairs construct a fresh PhysX scene for each replayed setting. Ordinary exploration reuses scenes across resets for throughput. Seeds reproduce the visible starting state. Exact numerical replication of subsequent contact dynamics uses the archived trajectories. The run manifest fingerprints the arrays used for the reported metrics.

[Completed study and decisions](../../docs/experiments/context_dynamics.md) · [Saved supervisor packet](../../runs/context_dynamics/nov13/context_diagnostics.pdf)

To regenerate the final cross-model figures, pass both matched-model evaluations as the last two arguments:

```bash
python -m experiments.context_dynamics.report runs/context_dynamics/nov13 runs/context_dynamics/nov13/z8/checks_0 runs/context_dynamics/nov13/z4_grip/checks_1 runs/context_dynamics/nov13/z4_matched/checks_1 runs/context_dynamics/nov13/z4_matched/checks_external
```
