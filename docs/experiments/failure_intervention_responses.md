# Failure intervention responses

[Experiment index](../experiments.md)

**Status:** Exploratory matched-intervention result.

## Question

Do failed attempts respond differently to tested changes in mass and friction?

## Method

Reuse 90 saved attempts: a complete 3 × 3 mass/friction grid across ten matched
starts. Compare recorded outcomes under mass-only, friction-only and joint
physical changes. Group the 51 failed source cases by their measured
intervention-response vectors. The policy's supplied mass estimate remains 40 g.

## Result

The failed cases form seven observed response profiles. Among the tested levels,
five failed cases had no successful single-variable change but did have a
successful joint change.

## Limitation

These are response profiles, not verified distinct physical mechanisms.
Equal responses need not imply equal causes; different responses need not imply
different causes. Trials reuse seeds and comparison targets. The joint-change
finding is bounded by the tested levels.

These interventions change simulator physics, not estimator inputs, grip commands
or policy weights. Deployment recognition of the profiles has not been validated.

## Next step

Inspect replay examples, refine intervention levels and evaluate additional
held-out starts. Separately test whether correcting an estimated policy input
improves the same attempt.

## Code and evidence

- [Method and result](../../failure_discovery/results/response_mass_friction/METHOD_AND_RESULTS.md)
- [Result report](../../failure_discovery/results/response_mass_friction/index.html)
- [Summary](../../failure_discovery/results/response_mass_friction/summary.json)
- [Observed grid](../../failure_discovery/results/response_mass_friction/observed_grid.csv)
- [Matched evidence](../../failure_discovery/results/response_mass_friction/matched_intervention_evidence.csv)
- [Response analysis code](../../robot_adaptation/analysis/failure_discovery/response_groups.py)
