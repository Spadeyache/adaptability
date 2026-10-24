# Failure behavior grouping

[Experiment index](../experiments.md)

**Status:** Exploratory result; causal failure types are unvalidated.

## Question

Can groups of recorded robot behavior recover useful distinctions between failures?

## Method

Analyze 230 saved attempts, including 154 failures. Represent object/gripper
movement, relative position and rotation, finger width, finger-force proxies,
actions and goal distance as 60 time samples. Standardize, apply PCA and cluster
the failures with k-means. Physical parameter values and old failure-type labels
are excluded from fitting. Held-out seeds test consistency on new starts.

## Result

Two broad candidate groups contain 115 and 39 failures. The first has a median
peak object rise of 19.4 cm; the second has 1.3 cm. These are behavior differences,
with overlap, rather than verified distinct physical causes.

Representative recorded-pose replays and group plots are available.

## Limitation

The groups did not establish the causal distinctions we wanted. The features
included forces and actions as well as motion; this was not a motion-only study.
Consistent grouping does not demonstrate correct failure names, known/unknown
recognition, or a diagnosis of whether understanding or movement capability
should change. Replay rendering reconstructs saved poses rather than rerunning
physics.

## Next step

Manually review representative failures, describe the sequence of events and
compare candidate explanations. Determine which observations or further
interactions could distinguish those explanations.

## Code and evidence

- [Method guide](../../robot_adaptation/analysis/failure_discovery/README.md)
- [Result report](../../failure_discovery/results/mass_friction_discovery/index.html)
- [Scores and feature definitions](../../failure_discovery/results/mass_friction_discovery/scores.json)
- [Identification notes](../../failure_discovery/results/mass_friction_discovery/IDENTIFICATION.md)
- [Group plot](../../failure_discovery/results/mass_friction_discovery/groups.png)
- [Representatives](../../failure_discovery/results/mass_friction_discovery/representative_episodes.csv)
- Replays: [group 1](../../failure_discovery/results/mass_friction_discovery/examples/group_1_01.gif), [group 2](../../failure_discovery/results/mass_friction_discovery/examples/group_2_01.gif)
