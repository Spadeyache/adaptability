# RMA adaptation baseline

[Experiment index](../experiments.md)

**Status:** Integration checks passed; GPU training runs in `runs/rma/baseline_20261106_persistent/` under the user systemd service `rma-baseline-20261106.service`. The earlier detached run was interrupted at update 3; only its pre-learning checkpoint was saved. Learned task performance is not yet established.

## Question

Where does an RMA-style policy succeed or fail when object mass and friction move
beyond its training range? This establishes a baseline for the subsequent question
of whether the robot can recognize unfamiliar conditions from its own observations.
An unseen episode can be in-distribution; out-of-distribution episodes shift the
physical distribution. Familiar failures and unfamiliar successes are both possible.

## Method

Jointly train a privileged environment encoder and PPO policy, then freeze both
and learn an adapter from recent state/action history. Use the existing carry-and-
hold task with fresh weights, a 20–120 g / 0.3–1.0 crossed training grid, and separate
training, development, and final-test seed blocks. True mass/friction are excluded
from deployment state/history. Object poses/contact forces remain simulator proxies.

This is an independent, state-based implementation informed by
[RMA²](https://github.com/yichao-liang/rma4rma), integrated with the existing pinned
PPO and ManiSkill3 task. It omits depth vision and object identity dictionaries;
its success rates are not directly comparable to the paper's tasks.

## Evidence so far

GPU Phase 1 check: two PPO updates, 6,000 transitions. GPU Phase 2 check: two
60-environment episode batches, 6,000 transitions. The paired 70 g / 0.65 smoke
evaluation completed one oracle and one adapted episode; both failed. This checks
execution and logging, not learned competence. A visual report and simulator GIFs
are in `runs/rma/integration_evaluation/`. Fifteen focused policy/interface tests
passed, including privileged-input exclusion and frozen-policy adapter training.

## Outputs and next step

[Commands and evaluation contract](../../experiments/rma/README.md).
Each evaluation writes raw CSV/NPZ records, plots, replay GIFs when requested,
an HTML report, and `RESULTS.md`. Establish in-range competence first; then collect
reserved OOD results. Novelty signals and cross-trial memory remain later work.
