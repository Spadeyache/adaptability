# Movement capability

[Experiment index](../experiments.md)

**Status:** Foundation. Training and inspection are implemented; performance
depends on the selected saved checkpoint.

## Question

Can a pick-and-lift policy provide the movement capability needed to study
adaptation through physical understanding?

## Method

Use a Panda/ManiSkill pick-and-lift task with state, a supplied object-weight
input, and finger-force feedback. The active v2 learner has independent
movement/opening and grip-strength branches. Collection explicitly uses the
preserved frozen v1 interface associated with its historical data.

## Result

Saved policies, training commands and inspection tools are available. They supply
the existing skill used by collection and estimator integration. The existence
of the v2 learner does not establish that a new joint-training run has completed.

## Limitation

Training and collection use different task variants and friction assumptions.
Checkpoint versions are not interchangeable just because input/output sizes
match. This foundation alone does not demonstrate deployment-time learning.

## Next step

Select and record the policy checkpoint for each adaptation experiment.
Before training an updater, test whether changing the supplied estimate changes
that policy's behavior or task performance.

## Code and evidence

- [Policy guide and commands](../../robot_adaptation/policy/README.md)
- [Current model](../../robot_adaptation/policy/model.py) and [frozen v1](../../robot_adaptation/policy/frozen_v1.py)
- [Simulation guide](../../robot_adaptation/simulation/README.md)
- [Historical training runs](../../runs/learning/) and [inspection runs](../../runs/inspect/)
