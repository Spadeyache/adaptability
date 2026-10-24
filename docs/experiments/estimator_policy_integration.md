# Estimator–policy integration

[Experiment index](../experiments.md)

**Status:** Completed simulation integration. Policy-benefit comparison pending.

## Question

Can the initial image estimate be supplied to a saved policy during a complete
pick, lift and hold attempt?

## Method

Capture one initial RGB image, predict mass, convert it to gravitational weight
in newtons, and supply that value to the policy. Keep model weights and the
estimate fixed during the episode. Run the complete 50-action horizon rather
than stopping at the first success signal.

## Result

The full three-case run completed all 50 actions and ended grasped at the goal.

| Cube side | Actual mass | Estimated mass | Final success | Final continuous success |
| --- | ---: | ---: | --- | ---: |
| 4 cm | 64.0 g | 62.32 g | Yes | 2.10 s |
| 5 cm | 125.0 g | 121.24 g | Yes | 2.10 s |
| 6 cm | 216.0 g | 212.45 g | Yes | 2.05 s |

MAE across these three cases was 3.00 g. Recorded runner time was 10.10 seconds,
excluding Python/import startup.

An earlier full-horizon 4 cm run also completed 50 actions / 51 observations.
It predicted 62.319 g for 64.000 g, rose 105.5 mm, and maintained the goal-and-grasp
condition for the final 42 samples (about 2.1 seconds). Final goal error was
7.85 mm; runner time was 3.98 seconds, excluding startup.

## Limitation

These are configured examples with one starting seed per size, not a broad
reliability estimate. The runtime uses simulator state observations. It does not
implement a real-robot sensing stack, estimate revision, or learning across
attempts. No matched comparison yet establishes benefit over a fixed estimate.

Earlier nine-step clips stopped at first success; they do not establish
continued holding. Their saved artifacts remain preserved.

## Next step

Compare initial estimated mass, a fixed estimate and true mass on matched starts.
Use this as a prerequisite to the proposed
[interaction-update study](interaction_estimate_updates.md).

## Code and evidence

- [Runtime guide](../../robot_adaptation/runtime/README.md)
- [Deployment configuration](../../configs/visual_mass_deployment.json)
- [Three-case summary](../../runs/deployment/20261022_014201_067220/summary.json) and [recordings](../../runs/deployment/20261022_014201_067220/)
- [Earlier full-horizon summary](../../runs/deployment/20261021_235109_413496/summary.json) and [replay](../../runs/deployment/20261021_235109_413496/episode_000/rollout.gif)
