# Grip-aware PPO: current setup

The active v2 policy is `model.py`; the separately preserved v1 collection model
is `frozen_v1.py`. Training tasks live in `../simulation/`. New runs save flat,
standalone source snapshots beside their checkpoints; old run sources stay unchanged.

Prepare the joint movement/strength run without starting training:

```bash
cd /home/ubuntu/faliure
source scripts/activate_grip.sh
python -m robot_adaptation.policy.train --prepare-only --iterations 3500
```

This writes validated settings, copied initialization weights, frozen source and
runtime files, and a manifest with `status="prepared"`. It creates no simulator
and starts no training subprocess. The default source is the latest completed
compatible checkpoint; use `--checkpoint /absolute/path/final_ckpt.pt` to choose
one explicitly. Preparation copies source weights without updating them.

When ready to train, use the prepared directory printed by that command:

```bash
python -m robot_adaptation.policy.train --from-prepared runs/policy/PREPARED_RUN
```

`PREPARED_RUN` is a placeholder. This command creates a new run using the saved
settings and hash-checked `initialization.pt` snapshot in the prepared directory.
Later changes to the latest checkpoint or its original source path do not change
those prepared starting weights. The new run copies the current code; this option
reuses saved settings and weights. Running `python -m robot_adaptation.policy.train` without `--prepare-only`
starts training immediately. Inspection remains:

```bash
python -m robot_adaptation.policy.inspect
```

Edit `SETTINGS` in `train.py` or `inspect.py`. New training writes to `runs/policy/`; latest-checkpoint lookup also searches historical `runs/learning/` without modifying it. Training saves under
`runs/policy/<run>/`; inspection saves under `runs/inspect/<run>/`.
Check artifacts stay under `runs/checks/`.

## What the files do

- `../simulation/training_lift.py`: cube mass/inertia, floating goal, measurements and reward.
- `../simulation/grip_controller.py`: native Panda finger opening plus a chosen motor force limit.
- `model.py`: one combined policy with independent movement and strength branches; joint or strength-only optimization.
- `../simulation/training_hold.py`: real held-object practice; arm stays still while strength is learned.
- `train.py`: warm-start the latest model and run the pinned ManiSkill PPO loop.
- `inspect.py`: run without learning, log signals and compare weight estimates.
- `checks/`: optional checks of controllers, rewards, PPO updates and loading.
- `examples/`: simulator starter, original ManiSkill references and earlier experiments.

## One policy, two branches

The movement/opening actor and independent strength branch retain their learned
weights. The copies share no parameters; the movement branch supplies XYZ and
finger opening, and the strength branch supplies the fifth action. Loading an
existing branched checkpoint retains both branches. Converting a shared
45-input/five-action checkpoint preserves its deterministic outputs before
training. Original checkpoint files are never overwritten.

`train.py` defaults to `training_stage="carry_joint"`. PPO samples all five
actions and updates both actor branches, both used exploration distributions,
and the critic on the full picking/carrying task. Likelihood, entropy and KL
calculations include all five actions. This allows approach and finger opening
to learn alongside grip strength; it does not guarantee that a trained policy
will outperform its initialization. `Agent(optimization_mode="joint")` selects
this behavior. Preparation only enables the mode and copies weights.

The two previous stages remain available with
`Agent(optimization_mode="strength_only")`. Their movement/opening network and
saved exploration values stay frozen; only strength is sampled and contributes
to PPO likelihood, entropy and KL. Set `training_stage="hold_strength"` for
pre-grasped practice. Each episode begins with the cube physically between
closed fingers and a brief strong-grip
settling phase. No attachment or lock is used. These preparation steps are not
PPO experience. The episode then holds the arm at its initial position and lets
the strength action determine whether the cube stays held. Warm-up affects the
whole simulator, so this stage requires synchronized full resets and
`partial_reset=False`, including evaluation.

`training_stage="carry_strength"` uses the original floating-goal carrying
task, with movement/opening still frozen and PPO still optimizing only strength.
Preserved weights do not guarantee identical physical trajectories: new strength
changes forces, the observations, and therefore later actions. This stage does
not repair an approach policy that misses the cube.

Inspection defaults to `inspection_stage="carry"` to test the combined policy.
Set `inspection_stage="hold_strength"` to view the strength practice task instead.
The same checkpoint supports both. Training evaluation measures its selected
training stage; hold success is not evidence of successful reaching/carrying.
Both optimization modes use compatible checkpoint tensor layouts and identical
deterministic inference. Manifests record the selected mode, optimized actions
and frozen actions; old manifests without a mode retain strength-only loading.

## Inputs: retain the previous 43 and append two forces

| Measurement | Numbers |
|---|---:|
| Joint positions | 9 |
| Joint velocities | 9 |
| Grasp contact flag | 1 |
| Hand pose | 7 |
| Floating goal position | 3 |
| Cube pose | 7 |
| Hand-to-cube displacement | 3 |
| Cube-to-goal displacement | 3 |
| True object weight, newtons | 1 |
| Measured left finger compression, newtons | 1 |
| Measured right finger compression, newtons | 1 |
| Total | 45 |

Object weight remains at index 42. The new forces are at indices 43 and 44.
Force observations are measured contact forces projected along each finger's
opening direction, with negative compression clamped to zero. They are neither
requested motor limits nor the grasp detector's binary flags. Immediately after
a carry-task reset, contact buffers can still describe the previous episode.
Reset robots now report zero force and not grasped until fresh physics data is
available; active robots keep their readings. Holding practice steps real physics
before returning its first observation, so its initial measured forces are real.

## Actions: retain the previous four and append strength

The five normalized actions are XYZ hand movement, finger opening, and
per-finger motor force limit. The arm settings and native opening mapping stay
unchanged. Opening maps from -1 closed to +1 open. Strength maps from -1 to
0.25 N and +1 to 40 N **per finger**, controlled by `grip_force_limits_n`.

Strength caps how much the native position-drive motors may push. It does not
ask the simulator to fabricate an exact contact force. Measured forces can also
contain collision impacts. Both fingers use the same requested limit, and each
parallel robot can choose its own limit. The cap applies to both closing and
opening effort; a higher limit also lets the fingers open more firmly. Native implicit position-drive
integration is retained on CPU and GPU.

The real Franka gripper also exposes width and grasp force, but its API and
force convention are not identical to our simulated per-finger motor cap.
The custom action is a project extension, not ManiSkill's original baseline.

## Objective: carry and hold with only the force needed

The green target is a floating point for the **cube centre**, at 10–30 cm
upright bottom clearance, retaining the original randomized XY.
The task keeps PickCube's reaching, grasping, carrying and stillness terms.
Success additionally requires finger contact while at the target with a still arm.

Each step subtracts:

```text
grip_force_cost × clamp(mean measured finger compression / maximum motor limit, 0, 1)
```

The environment default coefficient is 0.05; the current training settings use
0.10, so the training cost is at most 0.10 per step. Successful holding remains
the dominant reward. This encourages lower measured
squeezing while still holding. It does not prescribe a force from a weight formula
or guarantee that the learned policy will use weight. All reward parts and the
total are logged, including after success.

Training now continues through a complete 50-step attempt after reaching the
goal. That gives the policy experience holding the cube, rather than immediately
resetting on the first success. Each attempt is 2.5 simulated seconds.

## Fine-tuning the latest trained policy

By default `checkpoint="latest"` chooses the newest completed main training
checkpoint compatible with either the previous weighted policy or this extension.
Checks never participate in automatic selection. A specific path pins a model.
If no compatible model exists, selection stops with an explanation.
To start from the published model instead, set `checkpoint=None` and leave
`initialize_from_pretrained=True`. Set both to None/False for random initialization.

A previous 43-input/four-action model is converted by copying every learned
existing weight, zeroing the two added input columns, and preserving the first
four action outputs and critic predictions. The added strength output initially
requests the maximum allowed strength, with exploration. It must learn during PPO.
The new motor cap and reward change physical behavior, even if copied actions match.

The official 42-input/four-action model is also convertible; its added weight
and force columns start at zero. Existing shared 45-input/five-action models are copied into the two branches;
branched models resume both branches directly.
Old custom 21-input/seven-action policies are incompatible.

Continuation loads model weights but restarts Adam and simulation. It is
fine-tuning, not exact optimizer-state resumption. Runs freeze the controller,
task, model, launcher, pinned PPO source and license and record the initialization
checkpoint/hash plus source settings. Earlier files and artifacts remain intact.

## Starting training settings

The mass range is 0.040, 0.064, 0.100, 0.250 and 0.500 kg. Shape remains
unchanged and physical inertia scales with mass. The current hidden friction
profile also reduces cube friction as mass increases, as detailed below. Sixty
GPU environments provide twelve robots per mass, with ten evaluation robots
(two per mass).
GPU masses remain assigned across resets. CPU one-robot checks cycle masses.

One rollout contains 60 × 50 = 3,000 interactions.
The default fine-tuning budget is 10,500,000 interactions: 3,500 collect/learn
cycles. `--iterations N` sets that number of cycles, each containing 3,000
interactions under the default environment/rollout sizes.
Each cycle has up to four PPO epochs, ten minibatches per pass, and target-KL
early stopping. The terminal's Epoch counter means collect/learn cycles.
Learning rate is 0.0001; other PPO loss/advantage settings retain the reference.
These are starting choices, not tuned or proven settings.

PPO remains the pinned ManiSkill baseline at commit
`a4a4f9272ad64b1564035874b605ceb687b63ed8`: three 256-unit tanh layers per network,
Gaussian exploration of all five actions in `carry_joint` (strength only in the
other stages), clipping actions before physics, and finite-horizon GAE. The
source baseline is preserved unchanged; the runtime replaces the model and task
integration. Published checkpoint metadata does not establish the exact
original training seed, budget or hyperparameter overrides.

## Inspect whether grip strength changes with weight

Inspection runs ten matched starts per actual mass with both true and fixed
40 g reported-weight inputs. Only index 42 changes in the nominal condition;
measured finger-force feedback stays available. The model weights are unchanged.

Outputs include:

- `summary.json`: success, grasp loss after lifting, average measured compression
  and requested strength during lifted contact.
- `success_rates.png`: success rates by physical mass and reported weight.
- `grip_strength_by_mass.png`: measured force and requested limit by mass.
- `steps.jsonl`: actual measurements, fed inputs, policy actions, applied actions
  and reward parts. During hold practice the fixed controller supplies movement/
  opening, so its applied action differs from the four ignored policy outputs.
- Two camera GIFs and initial/final/closest-held-goal PNGs for the first attempt
  in **both** conditions at each mass.
- Per-attempt plots of rewards, forces, requested limits and cube position relative
  to the fingers.

Force averages include only grasped steps with cube bottom clearance above 3 cm.
Missing averages mean no qualifying contact; compare success rates alongside
force plots. Loss of a grasp flag can mean opening or dropping, not confirmed
slipping. Relative cube motion and recordings help diagnose it.
Friction follows the saved experiment settings. No extra impulses or scripted
failure events are applied.

A successful experiment should demonstrate secure holds across masses, lower
force usage for light cubes where feasible, and repeatable comparisons against
fixed strength/incorrect weight. True-weight versus nominal input alone does not
prove adaptation from past failures. The policy is feedforward, receives true
weight during training and has no learned weight estimator or persistent memory.

The original pretrained runner uses the preserved v1 weighted task in
`examples/maniskill_pick_cube/weighted_lift_v1.py`, with its original controller
and reward. It is a separate reference, not an evaluation of this new controller.

## Historical verification of the original extension

CPU reward/controller checks and a 150-step PPO run passed. A 6,000-step GPU
run with 60 robots passed, including independent limits and partial-reset isolation.
Both added force columns and the strength output changed during PPO updates,
and the final model reloaded exactly. Six inspection attempts exercised raw logs,
new force plots, rewards, and both cameras. These are integration checks; the
early short runs did not establish learned adaptation. Subsequent full runs and
the diagnosis still did not establish a benefit from true weight input.

A separate fixed-position hold diagnostic used one matched starting seed.
At a 1 N per-finger motor limit, a 40 g cube stayed lifted while a 500 g cube
dropped. At 5 N, both stayed lifted. This establishes a physical force/weight
tradeoff without changing friction, rather than proving that PPO learned it.
The inherited grasp detector requires 0.5 N contact on both fingers; very gentle
physically stable holds can fall below that binary detector threshold.

Artifacts: `runs/checks/grip_extension/calibration/20261008T031936-anchored/`,
`runs/checks/grip_extension/gpu_training/20261008T031725-a918879c/`, and
`runs/checks/grip_extension/inspection/20261008T031931-9f92e11a/`.


## Recovering a failed hold preparation

Holding practice must establish real airborne finger contact before returning
an episode. Preparation now tries ten settling steps first, allows ten more if
contact is missing, and then repositions only the failed cubes for bounded
retries. It still aborts with the failed robot indices if no valid grasp can be
prepared after 50 steps. These transitions do not enter PPO's rollout or the
50-step episode limit; cubes are never attached to the hand.


The run `20261008T183327-ffbcabfe` saved `ckpt_126.pt` at 375,000 transitions
before preparation failed. `checkpoint="latest"` selects completed runs, so
use this checkpoint explicitly to preserve that progress. To continue for the
remaining 627,000 transitions of the original 1,002,000-transition budget:

```bash
python - <<'PYTHON'
from robot_adaptation.policy.train import main
main({
    "checkpoint": "runs/learning/20261008T183327-ffbcabfe/ckpt_126.pt",
    "total_timesteps": 627_000,
})
PYTHON
```

This loads the saved movement/strength/critic weights into a new run using the
repaired source. Adam's state, random generators, simulation, and displayed
update counter restart; model progress is retained. Do not execute the old
run's frozen `ppo_runtime.py`, which retains its original reset implementation.

## Hidden mass-dependent friction experiment

New training deliberately changes **both physical mass and surface friction**.
The cube's static and dynamic friction coefficients decrease linearly from
`1.0` at 40 g to `0.4` at 500 g, clamped to those bounds outside that mass range.
Both Panda finger coefficients are set to `0.2` instead of the native `2.0`.
The cube schedule uses actual physical mass; falsely reporting 40 g never
changes the physical material. These coefficients are hidden from the policy:
the model still has 45 inputs and five actions, with movement/opening frozen.

| Actual cube mass | Cube static/dynamic coefficient | Finger static/dynamic coefficient |
|---|---|---|
| 40 g | 1.000 | 0.200 |
| 64 g | 0.969 | 0.200 |
| 100 g | 0.922 | 0.200 |
| 250 g | 0.726 | 0.200 |
| 500 g | 0.400 | 0.200 |

This is an artificial combined weight/friction challenge. Success does not
isolate adaptation to weight alone, and existing pre-grasp pickup failures may
remain. Collision geometry, masses/inertias, gravity and motor strength bounds
retain their previous definitions. Static/dynamic material coefficients are
both logged; they are not the same quantity as measured contact force.

`train.py` retains the `0.10` force cost, starts from the latest completed
checkpoint by default, and writes the friction configuration, mass
mapping and experiment note into each new run's manifest. GPU materials are
assigned before physics initialization, separately for each robot's fixed mass.
CPU inspection updates cube material when actual mass changes at reset.

Inspection normally matches the training manifest's friction. Old checkpoints
without these fields retain their original native materials. To evaluate an
old model under the new challenge, set `friction_override` to the friction
configuration explicitly; the inspection manifest flags that physics differs
from training. The per-step logs, episode/group summaries, GIF/PNG overlays and
aggregate plots all identify the changing friction. The original baseline
inspection artifacts and saved model weights are not rewritten.

## Explicit mass and friction conditioning

The historical 45-input task and runtime remain available. The new explicit mode
appends object friction at index 45, after the original weight at index 42 and the
two measured force inputs. Its manifest uses
`independent_movement_strength_friction_v3` and records 46 observations. Loading
an older branched policy into this model copies both actors and critic, adding
zero columns for friction so the initial deterministic behavior is preserved.

```bash
python -m robot_adaptation.policy.train --condition-on-friction --prepare-only --iterations 200
```

This mode independently crosses the five physical masses with object friction
coefficients 0.2, 0.5, 1.0. Static and dynamic coefficients are equal; finger friction
is fixed at 0.2 and the native table material is fixed. Sixty training environments
cover four robots per condition, and 15 evaluation environments cover every
condition. This is separate from the older mass-correlated hidden-friction test.
The full pick/carry/hold task and all five PPO actions remain trainable.

Use a short diagnostic run before a long run. The paired evaluator rebuilds the
physics scene for every supplied-input condition, checks identical reset-state
fingerprints, and compares true inputs against fixed and deliberately wrong
mass/friction. It reports final success, continuous success over the final ten
steps, grasp loss after lifting, force usage, and descriptive seed-block bootstrap
intervals. Action sensitivity is secondary evidence, not a success claim.

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
python -m experiments.physics_conditioning.evaluate CHECKPOINT --seed-start 81000 --seed-count 5
```

Reserve 910xx seeds for the final assessment. The optional
`experiments.physics_conditioning.run_long` supervisor runs 50,000 rollout/update
cycles from a reviewed pilot, saves checkpoints every 500 cycles, then performs
untouched final evaluation automatically. It writes separate training/evaluation
status to its output directory. 50,000 cycles at 60 × 50 transitions require 150M
interactions; the internal PPO minibatch epoch count remains four. Long-run
continuation loads model weights and restarts Adam and simulation.
