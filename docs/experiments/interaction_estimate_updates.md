# Interaction estimate updates

[Experiment index](../experiments.md)

**Status:** Proposed. No complete interaction-driven learning loop is implemented.

## Question

Can a robot improve its physical understanding through deployment interaction
and use that understanding to perform better on later actions or attempts?

## Method under consideration

Start within one pick-and-lift task and one object type. Use an initial visual
estimate, then investigate whether motion, contact and outcome evidence can
correct it. A candidate first study holds the policy and friction fixed while
varying mass independently of appearance.

Before training an updater, check whether accurate supplied mass helps the chosen
policy and whether the available interactions contain useful mass information.
Possible matched comparisons are a fixed initial estimate, an interaction-updated
estimate and true mass as a reference, with equal interaction budgets.

The updater's form, the representation of physical understanding and the rules
for retaining learning across attempts remain open.

## Result

No adaptation result is claimed. Current work establishes
[initial estimation](initial_visual_mass.md),
[simulation integration](estimator_policy_integration.md), and exploratory
[failure analysis](failure_behavior_grouping.md).

## Limitation

An unfamiliar outcome does not automatically imply that a new movement is
required. It may reflect an inaccurate estimate, a missing physical factor,
insufficient movement capability, or several of these. Updating one estimate,
improving a shared estimator and learning a movement are different possible
changes.

## Next step

1. Read the three papers below.
2. Manually watch saved failure replays. Describe and compare what happens;
   record possible causes and the evidence still needed.
3. Choose a focused update experiment from that review. Keep observations,
   hypotheses and experimentally supported explanations distinct.

## Reading

- [Xie & Finn — Lifelong Robotic Reinforcement Learning by Retaining Experiences](https://arxiv.org/abs/2109.09180):
  reuse past data and policies when learning successive tasks.
- [REFLECT — Summarizing Robot Experiences for Failure Explanation and Correction](https://arxiv.org/abs/2306.15724):
  use multisensory experience summaries for LLM failure explanations.
- [AHA — A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation](https://arxiv.org/abs/2410.00371):
  train a vision-language model on generated failures to detect and explain them.

## Starting points

- [Failure replays and grouping](failure_behavior_grouping.md)
- [Parameter sweeps](physical_parameter_sweeps.md)
- [Earlier experiment proposal](../../presentations/interaction_adaptation_experiment.md) — design notes, not completed results
- [Runtime guide](../../robot_adaptation/runtime/README.md)
