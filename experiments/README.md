# Experiment drivers

[Research index](../docs/experiments.md) · [Development guide](../docs/development.md)

This folder contains executable experiment-specific choices. Reusable policy,
simulation, estimation, collection and analysis code lives in
`robot_adaptation/`. Each study's question, result and evidence are documented
under `docs/experiments/`.

The current driver is [CREST](crest/README.md), which composes the shared simulator
with context selection and controller-repair search. Most existing studies use
their component's command directly; they do not need duplicate wrapper packages.

For a new experiment, start with a study page and explicit status. Add a driver
here when it needs its own comparisons or orchestration. Save generated outputs
under `runs/` and link them from the study page. Do not create empty directories
to imply that proposed functionality is implemented.
