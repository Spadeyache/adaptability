# RMA baseline: integration, training, and limits

Reusable networks live in `robot_adaptation/policy/rma/`. These drivers compose
them with the shared simulator. Settings live in `configs/rma_baseline.json`;
outputs and source snapshots live under `runs/rma/`. Existing runs are never
reused as output directories. Training uses fresh RMA weights, as approved.

From the repository root:

```bash
source scripts/activate_grip.sh
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
python -m experiments.rma.train --output runs/rma/base_001
python -m experiments.rma.adapt --checkpoint runs/rma/base_001/final_ckpt.pt --output runs/rma/adapter_001 --episodes 1000
python -m experiments.rma.evaluate --checkpoint runs/rma/base_001/final_ckpt.pt --adapter runs/rma/adapter_001/final_adapter.pt --output runs/rma/eval_001 --seeds 34 --record
```

Commands run in foreground. GPU execution requires access to the PC GPU. Phase 1
supports `--prepare-only`, `--iterations N`, and a one-environment `--cpu` check.
Phase 2 supports `--cpu`. Evaluation uses CPU physics so every episode can select
its exact mass/friction; training uses GPU physics. Backend differences must be
considered when comparing training and evaluation success.

## Contract

Carry-and-hold with the existing custom task success flag at the final step.
Also log success at all final ten sampled endpoints. Keep 50-step attempts even
if success occurs earlier. Cube geometry stays fixed; physical inertia scales
with mass. Finger friction is 0.2; object static and dynamic coefficients are equal.

Training crosses six masses (20–120 g) with five friction values (0.3–1.0), with
two GPU environments per cell. These are discrete grid samples, not continuously
sampled episode parameters. The CPU diagnostic cycles the same grid. Normalization
uses fixed training-range scales and never clips out-of-range parameters.

Evaluation holds starting seeds and parameters fixed between oracle and adapted
modes and verifies equal initial observations. The default 36-cell grid includes
unseen in-range interpolation and mass-only, friction-only, combined OOD settings.
At 34 starts per cell there are at least 102 episodes per condition per mode.
The test seed block starts at 95000; training evaluation starts at 83000. Do not
use final OOD scores to select checkpoints. A fixed-grid success rate does not
establish performance over all possible deployment conditions.

## Visual evidence

Evaluation writes `episodes.csv`, one compressed step trace per episode, optional
camera GIFs, `success_maps.png`, `condition_success.png`, and per-episode diagnostic
plots. Open `index.html` to browse them. Each plot is regenerated from saved records:

```bash
python -m experiments.rma.plot runs/rma/eval_001
```

`summary.json` and `RESULTS.md` report actual success counts/rates. Intervals
bootstrap starting-seed blocks, retaining physical cells; one seed has no reported
interval. `--smoke --seeds 1` is explicitly labeled as integration-only evidence.
Phase 1 also writes TensorBoard events and `console.log`; Phase 2 writes
`learning.jsonl`. Neither low latent error nor passing software tests proves task
success. The oracle/adapted comparison helps distinguish policy and inference
limitations but does not prove a unique failure cause.

## Current limitations

Paper success percentages concern different objects, tasks, sensing, and physics.
They are not a stopping target for this cube experiment. Assess in-range competence
before interpreting OOD failures. A novelty detector and the historical scalar-mass
vision baseline comparison remain subsequent work. No successful trained RMA result
is claimed by the integration checks.

## Bounded unattended run

```bash
python -m experiments.rma.run --output runs/rma/baseline_20261106
```

The runner allows up to three 1,000-update base-policy blocks, then up to three
1,000-episode-batch adapter blocks. Each continuation loads only compatible RMA
weights and restarts the optimizer, with a fresh training seed/episode range.
After each block, 108 in-range CPU episodes measure development performance.
Operational gates are 90% oracle success, then 80% adapted success with at most a
10 percentage point gap. These point-estimate gates are engineering criteria,
not statistical proof or paper-reproduction targets. Repeated validation is
explicit development data. A budget exhaustion records `needs_review` and stops
without claiming completion; crashes record `failed`.

Only after both gates pass does the runner perform the reserved full evaluation.
Final test seeds start at 95000 (the earlier smoke check used 93000, which is
retired from final testing). Validation starts at 84000. No final OOD score is
used to choose or retrain a model.

The root `index.html` and `training_progress.png` update about once a minute while
subprocesses run. `manifest.json` records the current step and status. Task limits
and dataset splits are fixed in `config.json`. The PC must remain awake.
