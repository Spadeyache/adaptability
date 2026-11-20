# Iteration 002 — STOP at the requested input guard

No probes ran. There is still no leakage-controlled per-task episode-to-property R² table. Existing models, z tables, datasets, and reported numbers are preserved.

## First: main-run audit

[MAIN_RUN_AUDIT.md](/home/ubuntu/faliure/results/loop/002/MAIN_RUN_AUDIT.md) has a separate row for every retained main checkpoint, its dataset file, input code, and column meanings. Fifteen three-task/fixed-grip/varied-grip runs have matching recorded training source hashes and no explicit true mass/friction-derived state/action inputs. These include all five fixed-grip runs behind mass R² 0.755 and the 11.6% same-task gain. Their linear decoder uses physics as readout targets only, after fitting z from transitions.

Five original nov13 two-task runs have clean saved raw-state/action schemas but cannot be fully certified: recorded model.py SHA256 `f6bb4ac53a11660538e245eafd730874ca935ffa3f0cb2e8df0ac1a47914528d` was not recovered from retained revisions or Git blobs. `recovery.json` records the search. Claims about those five runs' historical z-training path remain unverified, including the unconditional z-path fields in the initial machine-readable audit. The source gap is not evidence that contamination occurred.

In the main schema, state42 is measured right-finger joint force, not supplied weight. State45 exists only in the extended state and is TCP-relative cube Z position, not friction. Raw 43D state has no column45; column45 of concatenated state+action would be vertical action.

## Guard implemented and triggered

The new permanent probe loader, `experiments/property_probe/data.py`, has no skip-guard option. It drops original state42 and extended state45 from both current/next observations and drops extended state48, the backward-difference descendant of relative Z. Labels, IDs, old learned z, and decoder outputs are not feature inputs. This removal uses source-column identity before reindexing, not arbitrary positions after concatenation. Actions have fewer than 42 columns.

The guard uses absolute Pearson correlation across equally weighted cubes, separately for each task and split, checking each retained input column's per-cube mean/std/min/max. Threshold remains strictly greater than 0.9. Three unit checks passed: negative correlation fails, constant columns are handled, and both state blocks plus the derivative are removed.

The first actual check failed: fixed-grip **train** lifts, 200 cubes. Saved values from `guard.json`:

| Retained feature | Target | Pearson r |
|---|---|---:|
| Mean state37: measured arm joint effort | Mass | 0.9975248645 |
| Mean state35: measured arm joint effort | Mass | -0.9927817042 |
| Mean vertical action | Mass | 0.9402066733 |

Additional violations are saved in full. These are input-correlation checks, not learned-probe test R². Correlation does not establish explicit label injection: measured effort or feedback actions can carry physical information. The literal guard blocks these inputs regardless of origin. No further feature removal, data collection, probe fitting, or threshold changes were made after failure.

## Reviews and decision

Leakage: direct source audit passes for the fifteen matched runs; five historical sources remain unverified; shuffled control unrun. Confounds: requested correlation guard **fails**, including an action channel; actions-only learned control unrun. Splits: checked loader uses original train/test cube identities separately; no normalizer fitted. Robustness: probes/seeds unrun. Skeptical PI: the audit distinguishes injected answers from measured correlations and cannot yet establish identification. Physics sanity: effort-related mass signal is a hypothesis consistent with this observation, not yet a held-out probe result.

The remaining decision is whether to retain the literal guard and reject these inputs, or explicitly revise the screening rule to distinguish observed physical signals from injected labels. Revising the rule changes what counts as success and requires user approval. The missing original training source is a separate unresolved limitation. Earlier privileged results are labeled in the additive `PRIVILEGED_INPUT_COMPARISONS.md`; originals are untouched.

Reproduce the guard from the repo root (failure is expected):

```bash
PYTHONPATH=. .venv/bin/python -m unittest tests.test_property_probe_guard
PYTHONPATH=. .venv/bin/python -c 'from experiments.property_probe.data import load_task; load_task("runs/lift_slide_belief/slip_probe/paired/fixed/data", "train", 0)'
```

The audit script and exact matched source copies are retained here. Its output files are exclusive-create; do not rerun it over prior output. No final all-lenses-pass claim is made.
