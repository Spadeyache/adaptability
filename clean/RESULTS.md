# Laws/constants experiment results

**All predeclared one-step test success criteria passed in the selected script regime.**

Completed 20 of the 20 allowed validation candidates. The iteration budget ended without validation success; the selected saved candidate received the single final evaluation.
The final candidate was iteration 19 (`iteration_19_39534e75942a`), using a script grip-force upper bound of 8 N. The claim here is limited to that data regime. Out-of-range performance is reported separately and is not required to pass.

This is limited prediction evidence. The Oracle advantage over Plain is small relative to its seed variation, and the per-seed gap fraction is undefined for one seed. Fitted z did not improve ten-step rollout error over Plain or History. Its frozen linear property decoder generalized poorly to unseen cubes despite the strong one-episode GRU probe. Out-of-range own z did not clearly beat shuffled z. The experiment therefore does not establish interpretable physical constants, robust extrapolation, or improved long-horizon prediction.

After candidate 10, reference selection was corrected to require all failure checks whenever a passing candidate existed, then minimize the same fitted-z validation NMSE within that set. Earlier selection had retained lower-error F4-failing references despite candidate 6 passing the checks. The unused initial plan for candidate 11 was archived before training; it is not an extra scientific iteration. No score, threshold, or completed result was changed.

The fixed sampling design used independent mass and friction draws: training/test mass 0.04–0.50 kg, out-of-range mass 0.55–0.90 kg, and friction 0.4–1.0. Property values and cube memberships were generated once.

All error and R² entries below are mean ± sample standard deviation across the five saved seeds. One-step scores include only the locked cube-moving transitions. NMSE averages normalized errors within each cube, then weights cubes equally; R² pools moving transitions. Rollouts recursively predict ten states from eligible moving starts using recorded actions. Normalization uses training data only.

| Group | Model / latent condition | One-step NMSE ↓ | R² ↑ | Ten-step rollout error ↓ |
|---|---|---:|---:|---:|
| Test | Plain | 0.28126 ± 0.0056511 | 0.787467 ± 0.00513935 | 0.469449 ± 0.0282471 |
| Test | History | 0.25734 ± 0.00581311 | 0.80686 ± 0.00462711 | 0.295698 ± 0.0353611 |
| Test | Oracle (ceiling only) | 0.276688 ± 0.0252419 | 0.793427 ± 0.0204743 | 0.467657 ± 0.0377866 |
| Test | Two-part: fitted z | 0.273758 ± 0.00594845 | 0.79584 ± 0.00406531 | 0.487546 ± 0.0392227 |
| Test | Two-part: mean z | 0.28828 ± 0.00741623 | 0.783438 ± 0.00457837 | 0.476387 ± 0.0365591 |
| Test | Two-part: shuffled z | 0.316887 ± 0.0121704 | 0.761416 ± 0.00853443 | 0.572687 ± 0.049805 |
| Test | Two-part: another cube's z | 0.302483 ± 0.00439226 | 0.773855 ± 0.00282766 | 0.505218 ± 0.0473316 |
| Out-of-range | Plain | 1.01341 ± 0.0204501 | 0.591496 ± 0.00858832 | 1.49983 ± 0.063651 |
| Out-of-range | History | 0.871005 ± 0.0345779 | 0.671118 ± 0.0100667 | 1.27096 ± 0.0165698 |
| Out-of-range | Oracle (ceiling only) | 0.985526 ± 0.0274953 | 0.603163 ± 0.0119309 | 1.68709 ± 0.0751275 |
| Out-of-range | Two-part: fitted z | 0.96118 ± 0.0118303 | 0.605581 ± 0.00459354 | 1.58005 ± 0.142336 |
| Out-of-range | Two-part: mean z | 1.13166 ± 0.0438332 | 0.543152 ± 0.0193978 | 1.69332 ± 0.231347 |
| Out-of-range | Two-part: shuffled z | 0.964199 ± 0.0160911 | 0.604909 ± 0.00403516 | 1.59648 ± 0.14696 |
| Out-of-range | Two-part: another cube's z | 1.16653 ± 0.0423039 | 0.522272 ± 0.0172808 | 1.76599 ± 0.240627 |

Undefined values remain undefined; no seed or cube is dropped to improve an aggregate.

## Success criteria

| Criterion | Locked rule | Test | Out-of-range (informational) |
|---|---|---|---|
| S1 | Recover at least half of a positive Plain→Oracle NMSE gap | PASS | PASS |
| S2 | Fitted mean + std < shuffled mean − std | PASS | FAIL |
| S3 | Fitted NMSE ≤ 1.1 × History NMSE | PASS | FAIL |
| S4 | All failure checks pass | PASS | PASS |

**Test:** Plain→Oracle gap recovery, as the ratio of aggregate means used for S1: 164.093%. Paired per-seed gap recovery: undefined. Paired gain over mean z (mean-z NMSE minus fitted-z NMSE): 0.0145229 ± 0.00457605.
Paired gap recovery is undefined because the Plain→Oracle denominator is nonpositive or undefined for seed(s) 211; those seeds are retained, so the aggregate paired statistic is undefined.

**Out-of-range:** Plain→Oracle gap recovery, as the ratio of aggregate means used for S1: 187.305%. Paired per-seed gap recovery: undefined. Paired gain over mean z (mean-z NMSE minus fitted-z NMSE): 0.170477 ± 0.0399083.
Paired gap recovery is undefined because the Plain→Oracle denominator is nonpositive or undefined for seed(s) 211; those seeds are retained, so the aggregate paired statistic is undefined.

The ratio of aggregate means and the mean of paired seed ratios answer different questions. S1 uses the former exactly as locked. A nonpositive aggregate Plain→Oracle gap makes S1 fail rather than yielding a favorable recovery percentage.

## Failure checks

| Check | Requirement | Final test | Final out-of-range |
|---|---|---|---|
| F1 | Per-column provenance and high-correlation classification | PASS | PASS |
| F2 | Actions-only held-out-cube probe below the locked threshold in every seed | PASS | PASS |
| F3 | No held-out test/out-of-range motion used for training or tuning | PASS | PASS |
| F4 | Own fitted z wins against shuffled z and mean z on validation | PASS | PASS |
| F5 | No detected dimension-growth memorization pattern on validation | PASS | PASS |
| F6 | Mean-z and Plain–Oracle comparisons reported | PASS | PASS |
| F7 | Five seeds; finite metrics; sample standard deviations | PASS | PASS |
| F8 | One-episode signal probe and fitted-z property probe reported | PASS | PASS |
| F9 | Locked metric used for selection | PASS | PASS |

F5 was conditional: the selected candidate has no otherwise-matched lower-dimension comparison. Its saved pass means no dimension-growth failure was detected; it does not establish that all larger latents avoid memorization.

F4 and F5 are validation checks carried into the final report. Finite five-seed metrics and frozen-weight hashes are checked again during final evaluation; each group's S4 uses its saved final failure-check record. Test S1–S3 use the final test scores.

## Property recoverability

| Diagnostic / evaluation cubes | Mass R² | Friction R² |
|---|---:|---:|
| Training-cube folds: one-episode actions-only GRU (F2) | -0.0111655 ± 0.0133962 | -0.0225546 ± 0.0293216 |
| Training-cube folds: one-episode state/action GRU (F8) | 0.959209 ± 0.0159058 | 0.79768 ± 0.136415 |
| Training-cube folds: ridge probe of fitted z (five fitting episodes) | 0.863371 ± 0.0369757 | 0.238521 ± 0.136836 |
| Test: one-episode state/action GRU | 0.968957 ± 0.025436 | 0.899054 ± 0.0658269 |
| Test: ridge probe of fitted z, five episodes | -1.18363 ± 3.32644 | -0.206263 ± 0.53603 |
| Out-of-range: one-episode state/action GRU | -6.6411 ± 2.38941 | 0.621969 ± 0.294252 |
| Out-of-range: ridge probe of fitted z, five episodes | -8.23119 ± 0.981076 | -0.430337 ± 0.560527 |

Training-cube-fold rows were evaluated entirely within the training identities. There, the z probe uses embeddings fitted with a dynamics model that already trained on those identities. Any Test and Out-of-range rows above are saved final diagnostics on unseen cubes, using probe weights fitted only on training cubes. The GRU uses one episode while z uses five. These scores have different evidence budgets; the GRU is not a rigorous information ceiling. A nonlinear decoder could also recover information that the fixed ridge probe misses.

## Data-regime change

- Grip-force upper bound 30 N: iterations 1, 2.
- Grip-force upper bound 8 N: iterations 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20.

The script's grip-force range was narrowed after validation failure. The corresponding trajectories and train-only normalization were regenerated under the same fixed cube identities and properties. Normalized errors across these data versions are not a same-distribution model-improvement comparison. Cross-version transitions are excluded from the ranking below. The archived earlier test/out-of-range trajectories were collected but never analyzed; final evaluation used only the selected regime.
The original 30 N episodes, plans, normalization, audits, and generator source remain under [/home/ubuntu/faliure/clean/data/versions/grip_max_30](/home/ubuntu/faliure/clean/data/versions/grip_max_30).

## Three changes that mattered most

1. **Grip-force upper bound: 30 N to 8 N.** The original validation failure was dominated by rapid alternating finger motion, as recorded in the training-cube diagnostics. Narrowing this range was intended to reduce that behavior while retaining randomized lifts. It yielded the narrower regime in which later models passed the specificity checks. The new normalization and trajectories prevent interpreting the lower NMSE as a same-data model gain or isolating the change's causal benefit.

2. **Latent dimension: 4 to 16.** The larger latent made predictions depend more clearly on which cube supplied z. This was the first dimension candidate to pass F4 and the non-overlapping-standard-deviation specificity criterion. It helped satisfy the object-specificity requirement, even though its overall NMSE was higher than the four-dimensional candidate. Extra latent capacity is a plausible reason, not proof that the coordinates became physical parameters. At 16 dimensions, fitted-z validation NMSE was 0.345667 ± 0.0111626, versus 0.377068 ± 0.0113453 with shuffled z.

3. **Latent L2 penalty: 0.0001 to 0.01.** This retained the specificity checks and gave the selected candidate the lowest validation NMSE among candidates passing all failure checks. The intended mechanism was to discourage unnecessarily large object embeddings. The observed effect was tiny and mixed across seeds, so a practical regularization benefit is not established: 0.345667 ± 0.0111626 before, 0.345539 ± 0.0113682 after. Paired relative change: -0.039% ± 0.103%.

## Largest measured regressions

Changes below are ranked by the absolute relative change in mean fitted-z validation NMSE from the recorded base candidate (or the preceding candidate when no base was recorded), only when the generator regime and audited dataset match. This describes observed effects, including regressions; it does not prove a causal mechanism. Every displayed error and paired change uses the five saved seeds.

- **Iteration 16, compared with base iteration 6 in the 8 N regime: `moving_weight` changed from `1.0` to `10.0`.** Fitted-z validation NMSE increased: 0.345667 ± 0.0111626 → 0.399 ± 0.00691585. Paired relative change (new minus previous, divided by previous): 15.486% ± 2.572%. The recorded explanatory hypothesis was: Oracle has not established a positive Plain-to-Oracle error gap. Giving moving transitions more loss weight may focus fitting on the transitions used by the locked metric. This is a hypothesis, not an established causal explanation.

- **Iteration 15, compared with base iteration 6 in the 8 N regime: `lr` changed from `0.001` to `0.003`.** Fitted-z validation NMSE increased: 0.345667 ± 0.0111626 → 0.398711 ± 0.0213301. Paired relative change (new minus previous, divided by previous): 15.597% ± 9.822%. The recorded explanatory hypothesis was: Oracle has not established a positive Plain-to-Oracle error gap. A larger training learning rate may improve convergence within the unchanged update budget. This is a hypothesis, not an established causal explanation.

- **Iteration 17, compared with base iteration 6 in the 8 N regime: `prediction_horizon` changed from `1` to `3`.** Fitted-z validation NMSE increased: 0.345667 ± 0.0111626 → 0.383286 ± 0.0216029. Paired relative change (new minus previous, divided by previous): 10.879% ± 5.177%. The recorded explanatory hypothesis was: Oracle has not established a positive Plain-to-Oracle error gap. A short recursive training loss may favor shared dynamics that remain useful beyond one step. This is a hypothesis, not an established causal explanation.

## Suspicious observations and limits

- Ten preplanned randomized scripts are shared across all cubes. This exactly balances action commands against labels and helps F2, but covers limited action diversity. Validation holds out only two of those trajectories.
- Validation cube identities also occur in weight training. Fitting a fresh z tests adaptation on held-out episodes of familiar cubes; only the final evaluation tests unseen identities.
- Final scoring uses episodes 5–9, including three shared script settings (5–7) already seen during weight training on other cubes. Validation uses only settings 8–9. Thus the final test introduces new objects but includes more familiar trajectories; its pass after validation failed is not evidence of stronger generalization to new motions.
- The state includes cube world orientation and relative position, but omits explicit cube orientation relative to the gripper. Relative pose therefore is not fully represented by a dedicated feature.
- Joint effort is transmitted reaction wrench in the child-joint frame, sampled once per control step without averaging. The URDF joint-axis convention and PhysX force-then-torque layout support the chosen X projections, but the installed Python stubs do not expose the six-column getter implementation, so that component packing was not independently proved. This is not a calibrated hardware motor-torque measurement; no mass/friction leakage path was found.
- The saved reset diagnostic reports identical reset states but small differences between repeated CPU trajectories. This infrastructure diagnostic was not a five-seed scientific result; it shows that bitwise-identical complete trajectories should not be assumed.
- The narrowed-grip pilot reused one scene across cubes, while parallel full collection created a fresh environment per cube. Both used the same physical setup and commands. Exact equivalence of contact-cache state and numerical trajectories across those environment lifecycles was not established.
- The pilot probe had only two held-out training cubes per fold and produced unstable, sometimes strongly negative R². This small-fold instability is distinct from the five-seed dynamics-model uncertainty. The full audit uses all training identities and substantially larger held-out-cube folds.
- The installed planner crashed with NumPy 2.2.6 and ran with NumPy 1.26.4 isolated under clean/vendor. This is a saved local compatibility observation, not proof of a general library defect. Reproduction needs that runtime path.
- Oracle is a supervised reference under the same optimization budget, not a guaranteed attainable error lower bound. If it fails to beat Plain, the locked gap-recovery claim cannot pass.
- F2 detects property information in the sampled commands under the tested probe; matched scripts make the commands identical across cubes. Passing it does not certify every possible future controller.
- The one-episode GRU is an empirical signal benchmark. It does not establish the maximum information any estimator could extract.
- Recorded selected-model parameter counts are 30,128 for Plain and 118,320 for History. The matched budget is 2,000 optimizer updates with the same minibatch size and data; it does not equalize parameter count, floating-point operations, or wall-clock time.

## Exact reproduction commands

To replay the complete recorded candidate sequence, retrain every diagnostic probe and model, and run one final evaluation in a new copy, use the helper below. It restores the matching 30 N and 8 N data regimes automatically. The original split, properties, results, and final ledger are preserved. The helper was checked in plan-only mode; no extra scientific run was performed after the budget ended.

```bash
cd /home/ubuntu/faliure
PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.replay --plan-only
replay_parent=$(mktemp -d /tmp/clean-all-XXXXXXXX)
PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.replay --destination "$replay_parent/experiment" --device cuda
```

Add `--recollect` to the last command to also regenerate all motion episodes in that separate copy, keeping the existing cube identities and true properties. Without that flag, every training/probe/final computation is repeated on the saved trajectories. The replay stops early if a reproduced validation candidate meets all criteria.

The original setup commands are recorded below for provenance. Run collection and audits only in a separate reproduction, rather than replacing the completed original audit. Existing sealed split/property files and completed episodes must be retained; collection checks them and skips completed work.

```bash
cd /home/ubuntu/faliure
export PYTHONPATH=clean/vendor:.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/clean-mpl
.venv/bin/python -m clean.protocol
.venv/bin/python -m clean.collect --stage pilot
.venv/bin/python -m clean.audit --stage pilot
.venv/bin/python -m clean.parallel_collect --workers 4
.venv/bin/python -m clean.audit --stage full
.venv/bin/python -m unittest clean.test_models clean.test_train clean.test_evaluate clean.test_audit clean.test_report clean.test_search clean.test_parallel_collect -v
```

The original final-evaluation ledger is single-use. The following commands retrain only the selected final candidate on its matching saved dataset in a separately named copy, then run final evaluation once in that copy. Earlier candidate records remain historical evidence of the iteration budget; they are not retrained on a different generator regime. The copy is a reproduction, not an additional tuning result. GPU access is required for the recorded CUDA device.

```bash
cd /home/ubuntu/faliure
reproduction_dir=$(mktemp -d /tmp/clean-reproduction-XXXXXXXX)
cp -a clean "$reproduction_dir/clean"
cd "$reproduction_dir"
mv clean/runs clean/source_runs
mv clean/results clean/source_results
mkdir clean/runs clean/results
cp clean/source_results/full_audit.json clean/results/full_audit.json
export PYTHONPATH=clean/vendor:.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/clean-mpl
/home/ubuntu/faliure/.venv/bin/python - <<'PY'
import hashlib, json, shutil, subprocess, sys
from pathlib import Path
selected = 19
for directory in sorted(Path('clean/source_runs').glob('iteration_*')):
    if directory.name != ('iteration_%02d' % selected):
        shutil.copytree(directory, Path('clean/runs') / directory.name)
saved = Path('clean/source_runs') / ('iteration_%02d' % selected) / 'result.json'
result = json.loads(saved.read_text())
assert result['audit_sha256'] == hashlib.sha256(Path('clean/results/full_audit.json').read_bytes()).hexdigest()
configuration = Path('clean/configs') / ('reproduction_%02d.json' % selected)
configuration.write_text(json.dumps(result['config'], indent=2) + '\n')
command = [sys.executable, '-m', 'clean.run', 'candidate',
    '--iteration', str(selected), '--config', str(configuration),
    '--hypothesis', result['hypothesis'], '--change', result['change'],
    '--device', 'cuda']
if result.get('base_iteration') is not None:
    command += ['--base-iteration', str(result['base_iteration'])]
subprocess.run(command, check=True)
subprocess.run([sys.executable, '-m', 'clean.run', 'final', '--iteration', '19', '--device', 'cuda'], check=True)
subprocess.run([sys.executable, '-m', 'clean.report'], check=True)
PY
```

To regenerate this Markdown from the original saved results without rerunning any evaluation:

```bash
cd /home/ubuntu/faliure
PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.report
```

Earlier iterations used a different dataset. To rerun only the archived 30 N validation candidates, start a second fresh copy and restore its matching episodes, normalization, generator source, and audits. The immutable cube identities, property table, and protocol are retained. No final evaluation is invoked for this old regime.

```bash
cd /home/ubuntu/faliure
old_reproduction_dir=$(mktemp -d /tmp/clean-grip30-reproduction-XXXXXXXX)
cp -a clean "$old_reproduction_dir/clean"
cd "$old_reproduction_dir"
mv clean/runs clean/source_runs
mv clean/results clean/source_results
mkdir clean/runs clean/results
export PYTHONPATH=clean/vendor:.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/clean-mpl
/home/ubuntu/faliure/.venv/bin/python - <<'PY'
import hashlib, json, shutil, subprocess, sys
from pathlib import Path
root = Path('clean'); data = root / 'data'
archive = data / 'versions/grip_max_30'
backup = root / 'source_active_regime'; backup.mkdir()
for name in ['train', 'test', 'out_of_range', 'plans', 'episode_settings.json', 'pilot_manifest.json', 'full_manifest.json', 'normalizer.json', 'generator.json', 'dataversion.json']:
    target = data / name
    if target.exists(): shutil.move(str(target), str(backup / name))
    source = archive / name
    if source.is_dir(): shutil.copytree(source, target)
    elif source.exists(): shutil.copy2(source, target)
for name in ['collect.py', 'native.py', 'schema.py', 'script.py']:
    shutil.copy2(root / name, backup / name)
    shutil.copy2(archive / name, root / name)
for name in ['pilot_audit.json', 'full_audit.json']:
    shutil.copy2(archive / name, root / 'results' / name)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
generator = dict(version='grip_max_30', grip_force_max=30.0,
    source_sha256={name: sha(root / name) for name in ['script.py', 'native.py', 'schema.py']},
    split_sha256=sha(data / 'split.json'))
(data / 'generator.json').write_text(json.dumps(generator, indent=2) + '\n')
for saved in sorted((root / 'source_runs').glob('iteration_*/result.json')):
    result = json.loads(saved.read_text())
    if result['config'].get('grip_force_max', 30.0) != 30.0: continue
    assert result['audit_sha256'] == sha(root / 'results/full_audit.json')
    config = root / 'configs' / ('reproduction_grip30_%02d.json' % result['iteration'])
    config.write_text(json.dumps(result['config'], indent=2) + '\n')
    command = [sys.executable, '-m', 'clean.run', 'candidate', '--iteration', str(result['iteration']),
        '--config', str(config), '--hypothesis', result['hypothesis'], '--change', result['change'], '--device', 'cuda']
    if result.get('base_iteration') is not None:
        command += ['--base-iteration', str(result['base_iteration'])]
    subprocess.run(command, check=True)
PY
```

## Saved evidence

- Final scores: [/home/ubuntu/faliure/clean/results/final/result.json](/home/ubuntu/faliure/clean/results/final/result.json)
- Selected validation result: [/home/ubuntu/faliure/clean/runs/iteration_19/result.json](/home/ubuntu/faliure/clean/runs/iteration_19/result.json)
- Data audit: [/home/ubuntu/faliure/clean/results/full_audit.json](/home/ubuntu/faliure/clean/results/full_audit.json)
- Independent final integrity audit: [/home/ubuntu/faliure/clean/results/final_integrity_audit.json](/home/ubuntu/faliure/clean/results/final_integrity_audit.json)
- Complete reproduction helper: [/home/ubuntu/faliure/clean/replay.py](/home/ubuntu/faliure/clean/replay.py)
- Locked definitions: [/home/ubuntu/faliure/clean/data/protocol.json](/home/ubuntu/faliure/clean/data/protocol.json)
