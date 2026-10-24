# CREST context selection

[Experiment index](../experiments.md)

**Status:** Experiment implementation available. A completed context-selection
result is not established by this overview.

## Question

Which environmental context factors affect a parameterized lift skill, and
which controller parameters can repair the observed effect?

## Method

Use the shared simulator with a waypoint/strength skill. Optimize a baseline,
change one declared context factor, attempt a repair, and search subsets of
the discovered controller changes. Keep experiment choices under
`experiments/crest/`.

## Result

Planning, bounded search, journaling, resume and context-to-controller
representations are implemented. Consult a completed run manifest/report before
claiming a selected set of factors or a trained upper policy.

## Limitation

This experiment is separate from the visual estimator and the frozen RL-policy
failure analyses. Context values are privileged simulator inputs. A discovered
repair is conditional on tested contexts, controller and search budget; it does
not establish universal relevance or solve deployment estimation.

## Next step

If context selection is needed for the next study, start with a bounded,
explicitly recorded subset. The immediate project focus remains reading and
manual failure review.

## Code and configuration

- [Implementation and detailed guide](../../experiments/crest/README.md)
- [Configuration](../../configs/crest_lift.json)
- Side-effect-free preview: `python -m experiments.crest --plan-only`
