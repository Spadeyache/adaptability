# RMA policy and history adaptation

This is an independent implementation of the two-stage RMA/RMA² method, integrated
with this repository's pinned ManiSkill PPO and existing grip-aware ManiSkill3 task.
Reference: https://github.com/yichao-liang/rma4rma and https://arxiv.org/abs/2312.04670.
It does not copy the external Stable-Baselines3 implementation or install its forks.

`model.py` keeps training and inference together. The environment encoder maps mass
and friction into eight latent values. The actor and critic receive the remaining
44 state measurements plus that latent. During deployment `action_from_latent`
accepts only those measurements and an adapter prediction. Weight (index 42) and
friction (45) are excluded from state/history. Contact and object pose are ideal
simulator sensor proxies, not demonstrated real-robot perception.

Phase 1 jointly trains encoder, actor, and critic with PPO. Phase 2 freezes all
three and regresses the history adapter to encoder outputs on adapter-driven
rollouts. Each history entry contains current state, previous clipped/applied
action, and a padding-validity flag. Each 50-step episode resets history and
previous action. The first prediction uses the reset observation and zero previous
action. Inference needs no true physical parameters. Only training labels and
oracle diagnostics use privileged parameters.

Differences from RMA²: fixed cube, no depth CNN or identity dictionaries, a small
masked temporal CNN, a five-action force-controlling task, and the existing PPO
implementation. This is a state-based adaptation baseline, not a reproduction of
the paper's results. There is no memory across trials or novelty classifier yet.

See [experiment commands](../../../experiments/rma/README.md).
