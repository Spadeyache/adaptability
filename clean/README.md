# Isolated laws/constants research loop

The subsequent **shove sensitivity check**, with no model training, is isolated
in [shove/README.md](shove/README.md). Its runs and results do not modify the
sealed experiment described below.

The active protocol is [REQUEST.txt](REQUEST.txt). It authorizes the ten-training-cube pilot, full collection, five-seed training, validation-only tuning for at most twenty iterations, and one final evaluation. Earlier scaffold restrictions are archived in `results/prior_scaffold/`.

Everything implemented for this run lives under `clean/`. No old project module is imported or modified. Installed simulator assets are read-only dependencies. NumPy 1.26.4 is isolated under `clean/vendor/` because the installed mplib 0.1.1 constructor crashes with NumPy 2.2.6; see the saved diagnostic logs and vendor README. ManiSkill 3.0.1, SAPIEN 3.0.3, and PyTorch 2.14.0 are installed.

## Data and provenance

`data/split.json` fixes 200 training cubes, 30 test cubes, and 30 heavier out-of-range cubes. The split, property table, and precise metric protocol are SHA-256 sealed once. True per-cube mass/friction values occur only in `data/cubes.json`. Episode arrays have only state, action, next state, cube/episode identifiers, and step indices; identifiers select train embeddings and never become numerical input features.

`script.py` adapts the reach, approach, close, and lift sequence from the installed `mani_skill/examples/motionplanning/panda/solutions/pick_cube.py` and its planner setup. Grasp symmetry follows its bounding-box axis-selection rule. The script sees initial cube pose and randomized settings only. The complete open-loop plan is generated before execution. Ten independently randomized settings are shared across all cubes, exactly balancing actions across object properties but limiting action diversity.

`native.py:set_cube_properties()` is the only cube-property setter. `schema.py:observe()` constructs the explicit physical whitelist in [STATE_COLUMNS.md](STATE_COLUMNS.md). Joint-wrench conventions and CPU body-property API were reimplemented from read-only references `experiments/lift_slide_belief/collect.py:measure` and `robot_adaptation/simulation/context_dynamics.py:set_episode`, as recorded in the original scaffold. No old RL policy is used.

## Splits and interpretation

Training uses episodes 0–7, validation uses episodes 8–9. Fresh validation z starts from the mean training embedding and fits only episodes 0–4 with all shared weights frozen. Validation cubes are familiar identities; this is not an unseen-cube validation set. Final test and out-of-range evaluation fit episodes 0–4 and score episodes 5–9. Held-out motion reads are blocked outside the logged final invocation.

All four model families share training transitions, optimizer update counts, batch sizes, and seeds. Model-specific parameter counts and runtimes are recorded. Identical baseline checkpoints may be reused when a candidate changes only latent-specific or fitting settings; their original training budgets remain identical. History uses at most twenty state/action pairs and starts fresh each episode. Ten-step prediction uses predicted intermediate states with recorded future actions.

One-step error and R² are scored only on predefined moving steps. Ten-step error, normalization, cube weighting, five seeds, and success arithmetic are locked in `data/protocol.json`. No alternate metric selects a model. A supervised one-episode GRU probe is an empirical recoverability benchmark, not a rigorous upper bound; the z probe uses five episodes and familiar training identities, so these comparisons have limits.

## Commands

Run from `/home/ubuntu/faliure`. All commands need the isolated NumPy import path. CPU simulation requires no renderer. GPU training needs host GPU access when the command sandbox hides the device.

```bash
export PYTHONPATH=clean/vendor:.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/clean-mpl
.venv/bin/python -m clean.protocol
.venv/bin/python -m clean.collect --stage pilot
.venv/bin/python -m clean.audit --stage pilot
.venv/bin/python -m clean.collect --stage full
.venv/bin/python -m clean.audit --stage full
.venv/bin/python -m unittest clean.test_models clean.test_train clean.test_evaluate clean.test_audit -v
.venv/bin/python -m clean.run candidate --iteration 1 --device cuda
```

Completed episodes and the immutable split are never regenerated. Candidate configurations and exact later commands are saved in `configs/` and `LOOP_LOG.md`. Final evaluation refuses a second invocation. To reproduce final metrics without violating this experiment's one-use ledger, make a separate copy of the complete `clean/` experiment, retain its immutable data and checkpoint seals, and explicitly label the copy a reproduction rather than a new tuning candidate.
