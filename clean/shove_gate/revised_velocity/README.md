# Horizontal-velocity gate on fresh test cubes

**Completed: the revised gate failed.** Oracle's moving-coast horizontal
velocity-change error was 19.15% worse than Plain, with separation favoring
Plain beyond seed spread. See [RESULTS.md](RESULTS.md) for all metrics,
validation curves, physics comparisons, and the retained unstable stopping
rollout. No belief model or test-based tuning followed.

This revision follows the [written protocol](REVISED_PROTOCOL.md). The previous
gate remains reported in [the parent results](../RESULTS_DIRECT.md); its test
cubes 200–229 are retired. Existing training cubes 0–199 are reused without
changing their saved bytes. Fresh test cubes 230–259 use seed 2026111517.

The new primary metric is moving-coast horizontal velocity-change NMSE, scaled
by fitting-only change energy. The [physics sanity report](PHYSICS_FLOOR_REPORT.md)
records the analytic reference before model fitting. Architecture, training
loss, allowed inputs, trimming, and all previous secondary scores remain intact.
Validation selects the new metric and follows the preregistered budget extension
rule. No belief model is part of this revision.

The [final report](RESULTS.md) was generated after all ten models were frozen
and the single held-out evaluation completed. Validation curves and test
figures are generated solely from saved results.

Commands from `/home/ubuntu/faliure`:

```bash
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove_gate.revised_velocity.collection collect
PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.shove_gate.revised_velocity.collection audit
PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.shove_gate.revised_velocity.collection probe
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.shove_gate.revised_velocity.models_gate run --device cuda
PYTHONPATH=clean/vendor:. MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove_gate.revised_velocity.report
```

Do not edit sealed protocol, source, data, or checkpoints when resuming. Collection
never overwrites completed traces. Predictive test scoring starts only after
all ten checkpoints are sealed; model training is forbidden afterward. Report
generation does not simulate, train, or evaluate predictions.
