# Isolated laws/constants experiment

Only the ten-cube pilot is authorized before the next user approval. Full collection, world models, and checks A–E are pending. No old project package is imported. The installed external packages and existing simulator assets are read-only dependencies.

Verified installed versions: ManiSkill3.0.1, SAPIEN3.0.3, mplib0.1.1, PyTorch2.14.0. The original ranges read from `runs/lift_slide_belief/slip_probe/paired/fixed/data/dataset.json` are mass0.04–0.50kg and cube friction0.4–1.0. Fresh samples use seed2026111507; cube labels exist only in `data/cubes.json`.

## Sources and changes

- Installed `mani_skill/examples/motionplanning/panda/solutions/pick_cube.py`: reach above cube, approach, close, lift sequence adapted in `script.py`.
- Installed `mani_skill/examples/motionplanning/base_motionplanner/motionplanner.py`: mplib Panda planner setup and screw planning pattern adapted in `script.py`. `Panda.build_grasp_pose` supplies the equivalent orthogonal grasp-frame construction.
- Read-only old references `experiments/lift_slide_belief/collect.py:measure` and `robot_adaptation/simulation/context_dynamics.py:set_episode`: joint wrench axis conventions and CPU body-property API. The necessary logic is reimplemented locally in `schema.py` and `native.py`; no imports from these files.
- All other collection/probe/report code is new under `clean/`.

The script takes only initial cube pose and random settings. It plans the entire command sequence from fixed nominal robot joints before execution, using planned endpoints instead of observed tracking feedback. Grip cap is an issued action. Cube mass/friction/inertia are installed only by `set_cube_properties()`. Simulator observations/rewards/done flags do not influence commands or episode duration.

Ten independently randomized episode settings are shared across cubes. This matched design randomizes approach/lift speed, height, grip cap, and grasp offset while balancing the actions exactly across physical labels. Same episode index also shares initial cube pose and robot start. No settings depend on physical properties. Pilot train IDs0–7 and test IDs200–201 preserve the eventual200/30 identity split. Scores from two test cubes will be preliminary.

The state whitelist is documented in `data/STATE_COLUMNS.md`. Mean/std/min/max correlation flags will be inspected by provenance, not automatically rejected. A strong physical sensor correlation is allowed. The old numeric-column blacklist is not applied to this new explicitly named schema.

Pilot command from repo root:

```bash
PYTHONPATH=. .venv/bin/python -m clean.collect
```

Collection refuses to overwrite existing labels or episodes. A failed collection is saved and must be reviewed before retrying. No training or collection resumes automatically after a failed scientific check.
