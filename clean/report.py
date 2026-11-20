"""Render the completed saved experiment; never open simulator episode data.

This report is deliberately unavailable before final/result.json exists. Its
only inputs are saved final and validation results, the full audit, and the
sealed protocol. Formatting cannot trigger training, fitting, or evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_ROWS = (
    ("plain", "Plain"), ("history", "History"), ("oracle", "Oracle (ceiling only)"),
    ("fitted_z", "Two-part: fitted z"), ("mean_z", "Two-part: mean z"),
    ("shuffled_z", "Two-part: shuffled z"), ("another_cube_z", "Two-part: another cube's z"),
)
CHECK_LABELS = {
    "F1": "Per-column provenance and high-correlation classification",
    "F2": "Actions-only held-out-cube probe below the locked threshold in every seed",
    "F3": "No held-out test/out-of-range motion used for training or tuning",
    "F4": "Own fitted z wins against shuffled z and mean z on validation",
    "F5": "No detected dimension-growth memorization pattern on validation",
    "F6": "Mean-z and Plain–Oracle comparisons reported",
    "F7": "Five seeds; finite metrics; sample standard deviations",
    "F8": "One-episode signal probe and fitted-z property probe reported",
    "F9": "Locked metric used for selection",
}


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def number(value, *, percent=False):
    if not finite(value):
        return "undefined"
    return f"{value * 100:.3f}%" if percent else f"{value:.6g}"


def mean_std(summary, *, percent=False):
    if not isinstance(summary, dict) or not finite(summary.get("mean")) or not finite(summary.get("std")):
        return "undefined"
    return f"{number(summary['mean'], percent=percent)} ± {number(summary['std'], percent=percent)}"


def status(value):
    return "PASS" if value is True else "FAIL" if value is False else "NOT ESTABLISHED"


def _escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _paired_relative_change(previous, current):
    before = {run["seed"]: run["metrics"]["fitted_z"]["one_step_nmse"] for run in previous["runs"]}
    values = []
    for run in current["runs"]:
        old = before.get(run["seed"])
        new = run["metrics"]["fitted_z"]["one_step_nmse"]
        if not finite(old) or old <= 0 or not finite(new):
            return {"mean": None, "std": None}
        values.append((new - old) / old)
    if len(values) != 5:
        return {"mean": None, "std": None}
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def ranked_changes(iterations):
    """Rank observed one-change transitions, using saved validation NMSE only."""
    changes = []
    ordered = sorted(iterations, key=lambda result: result["iteration"])
    by_iteration = {result["iteration"]: result for result in ordered}
    for current in ordered:
        base = current.get("base_iteration")
        if base is None:
            base = current["iteration"] - 1
        previous = by_iteration.get(base)
        if previous is None:
            continue
        old_config = {"grip_force_max": 30.0, **previous["config"]}
        new_config = {"grip_force_max": 30.0, **current["config"]}
        if old_config["grip_force_max"] != new_config["grip_force_max"]:
            continue
        if previous.get("audit_sha256") != current.get("audit_sha256"):
            continue
        keys = set(old_config) | set(new_config)
        changed = [key for key in sorted(keys) if old_config.get(key) != new_config.get(key)]
        if len(changed) != 1:
            continue
        before = previous["summary"]["fitted_z"]["one_step_nmse"]["mean"]
        after = current["summary"]["fitted_z"]["one_step_nmse"]["mean"]
        if not finite(before) or before <= 0 or not finite(after):
            continue
        relative = (after - before) / before
        changes.append({"previous": previous, "current": current, "key": changed[0],
            "relative_change_of_means": relative,
            "paired_relative_change": _paired_relative_change(previous, current)})
    return sorted(changes, key=lambda item: abs(item["relative_change_of_means"]), reverse=True)


def _reproduction_commands(root: Path, selected_iteration: int):
    """Commands use a separate output ledger and retain the immutable dataset."""
    project = root.parent
    python = project / ".venv/bin/python"
    # Paths are represented as shell-safe single-quoted strings, including spaces.
    import shlex
    q = shlex.quote
    return [
        "The original setup commands are recorded below for provenance. Run collection and audits only in a separate reproduction, rather than replacing the completed original audit. Existing sealed split/property files and completed episodes must be retained; collection checks them and skips completed work.",
        "", "```bash", f"cd {q(str(project))}",
        "export PYTHONPATH=clean/vendor:.", "export OMP_NUM_THREADS=2", "export OPENBLAS_NUM_THREADS=1",
        "export MPLCONFIGDIR=/tmp/clean-mpl", ".venv/bin/python -m clean.protocol",
        ".venv/bin/python -m clean.collect --stage pilot", ".venv/bin/python -m clean.audit --stage pilot",
        ".venv/bin/python -m clean.parallel_collect --workers 4", ".venv/bin/python -m clean.audit --stage full",
        ".venv/bin/python -m unittest clean.test_models clean.test_train clean.test_evaluate clean.test_audit clean.test_report clean.test_search clean.test_parallel_collect -v",
        "```", "",
        "The original final-evaluation ledger is single-use. The following commands retrain only the selected final candidate on its matching saved dataset in a separately named copy, then run final evaluation once in that copy. Earlier candidate records remain historical evidence of the iteration budget; they are not retrained on a different generator regime. The copy is a reproduction, not an additional tuning result. GPU access is required for the recorded CUDA device.",
        "", "```bash", f"cd {q(str(project))}",
        'reproduction_dir=$(mktemp -d /tmp/clean-reproduction-XXXXXXXX)',
        'cp -a clean "$reproduction_dir/clean"', 'cd "$reproduction_dir"',
        "mv clean/runs clean/source_runs", "mv clean/results clean/source_results",
        "mkdir clean/runs clean/results", "cp clean/source_results/full_audit.json clean/results/full_audit.json",
        "export PYTHONPATH=clean/vendor:.", "export OMP_NUM_THREADS=2", "export OPENBLAS_NUM_THREADS=1",
        "export MPLCONFIGDIR=/tmp/clean-mpl", f"{q(str(python))} - <<'PY'",
        "import hashlib, json, shutil, subprocess, sys", "from pathlib import Path",
        f"selected = {selected_iteration}",
        "for directory in sorted(Path('clean/source_runs').glob('iteration_*')):",
        "    if directory.name != ('iteration_%02d' % selected):",
        "        shutil.copytree(directory, Path('clean/runs') / directory.name)",
        "saved = Path('clean/source_runs') / ('iteration_%02d' % selected) / 'result.json'",
        "result = json.loads(saved.read_text())",
        "assert result['audit_sha256'] == hashlib.sha256(Path('clean/results/full_audit.json').read_bytes()).hexdigest()",
        "configuration = Path('clean/configs') / ('reproduction_%02d.json' % selected)",
        "configuration.write_text(json.dumps(result['config'], indent=2) + '\\n')",
        "command = [sys.executable, '-m', 'clean.run', 'candidate',",
        "    '--iteration', str(selected), '--config', str(configuration),",
        "    '--hypothesis', result['hypothesis'], '--change', result['change'],",
        "    '--device', 'cuda']",
        "if result.get('base_iteration') is not None:",
        "    command += ['--base-iteration', str(result['base_iteration'])]",
        "subprocess.run(command, check=True)",
        f"subprocess.run([sys.executable, '-m', 'clean.run', 'final', '--iteration', '{selected_iteration}', '--device', 'cuda'], check=True)",
        "subprocess.run([sys.executable, '-m', 'clean.report'], check=True)", "PY", "```", "",
        "To regenerate this Markdown from the original saved results without rerunning any evaluation:",
        "", "```bash", f"cd {q(str(project))}",
        "PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.report", "```",
    ]


def _old_regime_commands(root: Path):
    """Restore the archived 30 N regime only in a fresh reproduction copy."""
    import shlex
    q = shlex.quote
    return ["", "Earlier iterations used a different dataset. To rerun only the archived 30 N validation candidates, start a second fresh copy and restore its matching episodes, normalization, generator source, and audits. The immutable cube identities, property table, and protocol are retained. No final evaluation is invoked for this old regime.",
        "", "```bash", f"cd {q(str(root.parent))}",
        'old_reproduction_dir=$(mktemp -d /tmp/clean-grip30-reproduction-XXXXXXXX)',
        'cp -a clean "$old_reproduction_dir/clean"', 'cd "$old_reproduction_dir"',
        "mv clean/runs clean/source_runs", "mv clean/results clean/source_results",
        "mkdir clean/runs clean/results", "export PYTHONPATH=clean/vendor:.",
        "export OMP_NUM_THREADS=2", "export OPENBLAS_NUM_THREADS=1", "export MPLCONFIGDIR=/tmp/clean-mpl",
        f"{q(str(root.parent / '.venv/bin/python'))} - <<'PY'",
        "import hashlib, json, shutil, subprocess, sys", "from pathlib import Path",
        "root = Path('clean'); data = root / 'data'", "archive = data / 'versions/grip_max_30'",
        "backup = root / 'source_active_regime'; backup.mkdir()",
        "for name in ['train', 'test', 'out_of_range', 'plans', 'episode_settings.json', 'pilot_manifest.json', 'full_manifest.json', 'normalizer.json', 'generator.json', 'dataversion.json']:",
        "    target = data / name",
        "    if target.exists(): shutil.move(str(target), str(backup / name))",
        "    source = archive / name",
        "    if source.is_dir(): shutil.copytree(source, target)",
        "    elif source.exists(): shutil.copy2(source, target)",
        "for name in ['collect.py', 'native.py', 'schema.py', 'script.py']:",
        "    shutil.copy2(root / name, backup / name)",
        "    shutil.copy2(archive / name, root / name)",
        "for name in ['pilot_audit.json', 'full_audit.json']:",
        "    shutil.copy2(archive / name, root / 'results' / name)",
        "sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()",
        "generator = dict(version='grip_max_30', grip_force_max=30.0,",
        "    source_sha256={name: sha(root / name) for name in ['script.py', 'native.py', 'schema.py']},",
        "    split_sha256=sha(data / 'split.json'))",
        "(data / 'generator.json').write_text(json.dumps(generator, indent=2) + '\\n')",
        "for saved in sorted((root / 'source_runs').glob('iteration_*/result.json')):",
        "    result = json.loads(saved.read_text())",
        "    if result['config'].get('grip_force_max', 30.0) != 30.0: continue",
        "    assert result['audit_sha256'] == sha(root / 'results/full_audit.json')",
        "    config = root / 'configs' / ('reproduction_grip30_%02d.json' % result['iteration'])",
        "    config.write_text(json.dumps(result['config'], indent=2) + '\\n')",
        "    command = [sys.executable, '-m', 'clean.run', 'candidate', '--iteration', str(result['iteration']),",
        "        '--config', str(config), '--hypothesis', result['hypothesis'], '--change', result['change'], '--device', 'cuda']",
        "    if result.get('base_iteration') is not None:",
        "        command += ['--base-iteration', str(result['base_iteration'])]",
        "    subprocess.run(command, check=True)", "PY", "```"]


def render_report(final: dict, iterations: list[dict], audit: dict, protocol: dict, root: Path = ROOT) -> str:
    selected = next((item for item in iterations if item["iteration"] == final["iteration"]), None)
    if selected is None:
        raise ValueError("The selected final candidate's saved validation result is missing")
    groups = final["groups"]
    if set(groups) != {"test", "out_of_range"}:
        raise ValueError("Final report requires both test and out-of-range groups")
    for group in groups.values():
        if len(group["runs"]) != len(protocol["seeds"]):
            raise ValueError("Final report requires all five saved seed runs")
    test_success = groups["test"]["success"]
    passed = all(test_success.get(f"S{i}") is True for i in range(1, 5))
    validation_passed = all(selected["success"].get(f"S{i}") is True for i in range(1, 5))
    regimes = {}
    for result in iterations:
        cap = result["config"].get("grip_force_max", 30.0)
        regimes.setdefault(cap, []).append(result["iteration"])
    selected_cap = selected["config"].get("grip_force_max", 30.0)
    lines = ["# Laws/constants experiment results", "",
        "**All predeclared one-step test success criteria passed in the selected script regime.**" if passed else
        "**The claim did not pass all predeclared test success criteria.**", "",
        f"Completed {len(iterations)} of the {protocol['iteration_budget']} allowed validation candidates. " +
        ("Validation met the stopping rule, followed by the single final evaluation." if validation_passed else
         "The iteration budget ended without validation success; the selected saved candidate received the single final evaluation."),
        f"The final candidate was iteration {final['iteration']} (`{_escape(final['candidate'])}`), using a script grip-force upper bound of {number(selected_cap)} N. The claim here is limited to that data regime. Out-of-range performance is reported separately and is not required to pass.", "",
        "This is limited prediction evidence. The Oracle advantage over Plain is small relative to its seed variation, and the per-seed gap fraction is undefined for one seed. Fitted z did not improve ten-step rollout error over Plain or History. Its frozen linear property decoder generalized poorly to unseen cubes despite the strong one-episode GRU probe. Out-of-range own z did not clearly beat shuffled z. The experiment therefore does not establish interpretable physical constants, robust extrapolation, or improved long-horizon prediction.", "",
        "After candidate 10, reference selection was corrected to require all failure checks whenever a passing candidate existed, then minimize the same fitted-z validation NMSE within that set. Earlier selection had retained lower-error F4-failing references despite candidate 6 passing the checks. The unused initial plan for candidate 11 was archived before training; it is not an extra scientific iteration. No score, threshold, or completed result was changed.", "",
        "The fixed sampling design used independent mass and friction draws: training/test mass 0.04–0.50 kg, out-of-range mass 0.55–0.90 kg, and friction 0.4–1.0. Property values and cube memberships were generated once.", "",
        "All error and R² entries below are mean ± sample standard deviation across the five saved seeds. One-step scores include only the locked cube-moving transitions. NMSE averages normalized errors within each cube, then weights cubes equally; R² pools moving transitions. Rollouts recursively predict ten states from eligible moving starts using recorded actions. Normalization uses training data only.", "",
        "| Group | Model / latent condition | One-step NMSE ↓ | R² ↑ | Ten-step rollout error ↓ |",
        "|---|---|---:|---:|---:|"]
    for group_name, label in (("test", "Test"), ("out_of_range", "Out-of-range")):
        for name, model_label in MODEL_ROWS:
            score = groups[group_name]["summary"][name]
            lines.append(f"| {label} | {model_label} | {mean_std(score['one_step_nmse'])} | {mean_std(score['r2'])} | {mean_std(score['rollout_error'])} |")
    lines += ["", "Undefined values remain undefined; no seed or cube is dropped to improve an aggregate.", "",
        "## Success criteria", "",
        "| Criterion | Locked rule | Test | Out-of-range (informational) |", "|---|---|---|---|"]
    rules = {"S1": "Recover at least half of a positive Plain→Oracle NMSE gap",
        "S2": "Fitted mean + std < shuffled mean − std",
        "S3": "Fitted NMSE ≤ 1.1 × History NMSE", "S4": "All failure checks pass"}
    for name, rule in rules.items():
        lines.append(f"| {name} | {rule} | {status(groups['test']['success'].get(name))} | {status(groups['out_of_range']['success'].get(name))} |")
    for group_name, label in (("test", "Test"), ("out_of_range", "Out-of-range")):
        group = groups[group_name]
        recovery = group["success"].get("gap_recovery")
        lines += ["", f"**{label}:** Plain→Oracle gap recovery, as the ratio of aggregate means used for S1: {number(recovery, percent=True)}. "
            f"Paired per-seed gap recovery: {mean_std(group.get('paired_gap_recovery'), percent=True)}. "
            f"Paired gain over mean z (mean-z NMSE minus fitted-z NMSE): {mean_std(group.get('paired_mean_z_gain'))}."]
        invalid_seeds = []
        for run in group["runs"]:
            p = run["metrics"]["plain"]["one_step_nmse"]
            o = run["metrics"]["oracle"]["one_step_nmse"]
            if not finite(p) or not finite(o) or p <= o:
                invalid_seeds.append(run["seed"])
        if invalid_seeds:
            lines.append("Paired gap recovery is undefined because the Plain→Oracle denominator is nonpositive or undefined for seed(s) " +
                ", ".join(map(str, invalid_seeds)) + "; those seeds are retained, so the aggregate paired statistic is undefined.")
    lines += ["", "The ratio of aggregate means and the mean of paired seed ratios answer different questions. S1 uses the former exactly as locked. A nonpositive aggregate Plain→Oracle gap makes S1 fail rather than yielding a favorable recovery percentage.", "",
        "## Failure checks", "", "| Check | Requirement | Final test | Final out-of-range |", "|---|---|---|---|"]
    for check, description in CHECK_LABELS.items():
        test_checks = groups['test'].get('failure_checks', final['failure_checks'])
        oor_checks = groups['out_of_range'].get('failure_checks', final['failure_checks'])
        lines.append(f"| {check} | {description} | {status(test_checks.get(check))} | {status(oor_checks.get(check))} |")
    comparisons = selected.get("f5", {}).get("comparisons", [])
    if not comparisons:
        lines += ["", "F5 was conditional: the selected candidate has no otherwise-matched lower-dimension comparison. Its saved pass means no dimension-growth failure was detected; it does not establish that all larger latents avoid memorization."]
    else:
        lines += ["", "F5 compares learned-z validation error with fresh-fitted-z validation error as latent dimension grows. The saved comparisons use candidate(s) " +
            ", ".join(str(item["iteration"]) for item in comparisons) + ". This is a familiar-cube validation proxy; no held-out test result was used to choose dimension."]
    lines += ["", "F4 and F5 are validation checks carried into the final report. Finite five-seed metrics and frozen-weight hashes are checked again during final evaluation; each group's S4 uses its saved final failure-check record. Test S1–S3 use the final test scores.", "",
        "## Property recoverability", "",
        "| Diagnostic / evaluation cubes | Mass R² | Friction R² |", "|---|---:|---:|"]
    probes = audit["probes"]
    for key, label in (("actions_only", "Training-cube folds: one-episode actions-only GRU (F2)"), ("state_action", "Training-cube folds: one-episode state/action GRU (F8)")):
        scores = probes[key]["r2"]
        lines.append(f"| {label} | {mean_std(scores['mass_kg'])} | {mean_std(scores['friction'])} |")
    lines.append(f"| Training-cube folds: ridge probe of fitted z (five fitting episodes) | {mean_std(final['z_recovery']['mass_r2'])} | {mean_std(final['z_recovery']['friction_r2'])} |")
    for group_name, label in (("test", "Test"), ("out_of_range", "Out-of-range")):
        recovered = groups[group_name].get("property_recovery")
        if recovered is not None:
            for key, diagnostic in (("gru", "one-episode state/action GRU"), ("z", "ridge probe of fitted z, five episodes")):
                value = recovered.get(key, {})
                lines.append(f"| {label}: {diagnostic} | {mean_std(value.get('mass_kg'))} | {mean_std(value.get('friction'))} |")
    lines += ["", "Training-cube-fold rows were evaluated entirely within the training identities. There, the z probe uses embeddings fitted with a dynamics model that already trained on those identities. Any Test and Out-of-range rows above are saved final diagnostics on unseen cubes, using probe weights fitted only on training cubes. The GRU uses one episode while z uses five. These scores have different evidence budgets; the GRU is not a rigorous information ceiling. A nonlinear decoder could also recover information that the fixed ridge probe misses.", ""]
    if len(regimes) > 1:
        lines += ["## Data-regime change", ""]
        for cap, numbers in sorted(regimes.items(), reverse=True):
            lines.append(f"- Grip-force upper bound {number(cap)} N: iterations " + ", ".join(map(str, sorted(numbers))) + ".")
        lines += ["", "The script's grip-force range was narrowed after validation failure. The corresponding trajectories and train-only normalization were regenerated under the same fixed cube identities and properties. Normalized errors across these data versions are not a same-distribution model-improvement comparison. Cross-version transitions are excluded from the ranking below. The archived earlier test/out-of-range trajectories were collected but never analyzed; final evaluation used only the selected regime.",
            f"The original 30 N episodes, plans, normalization, audits, and generator source remain under [{root / 'data/versions/grip_max_30'}]({root / 'data/versions/grip_max_30'}).", ""]
    by_id={result['iteration']:result for result in iterations}
    if all(number in by_id for number in [3,4,6,19]):
        lower,larger,regularized=by_id[4],by_id[6],by_id[19]
        lines += ["## Three changes that mattered most", "",
            "1. **Grip-force upper bound: 30 N to 8 N.** The original validation failure was dominated by rapid alternating finger motion, as recorded in the training-cube diagnostics. Narrowing this range was intended to reduce that behavior while retaining randomized lifts. It yielded the narrower regime in which later models passed the specificity checks. The new normalization and trajectories prevent interpreting the lower NMSE as a same-data model gain or isolating the change's causal benefit.", "",
            "2. **Latent dimension: 4 to 16.** The larger latent made predictions depend more clearly on which cube supplied z. This was the first dimension candidate to pass F4 and the non-overlapping-standard-deviation specificity criterion. It helped satisfy the object-specificity requirement, even though its overall NMSE was higher than the four-dimensional candidate. Extra latent capacity is a plausible reason, not proof that the coordinates became physical parameters. "
            f"At 16 dimensions, fitted-z validation NMSE was {mean_std(larger['summary']['fitted_z']['one_step_nmse'])}, versus {mean_std(larger['summary']['shuffled_z']['one_step_nmse'])} with shuffled z.", "",
            "3. **Latent L2 penalty: 0.0001 to 0.01.** This retained the specificity checks and gave the selected candidate the lowest validation NMSE among candidates passing all failure checks. The intended mechanism was to discourage unnecessarily large object embeddings. The observed effect was tiny and mixed across seeds, so a practical regularization benefit is not established: "
            f"{mean_std(larger['summary']['fitted_z']['one_step_nmse'])} before, {mean_std(regularized['summary']['fitted_z']['one_step_nmse'])} after. "
            f"Paired relative change: {mean_std(_paired_relative_change(larger,regularized),percent=True)}.", ""]
    lines += ["## Largest measured regressions", "",
        "Changes below are ranked by the absolute relative change in mean fitted-z validation NMSE from the recorded base candidate (or the preceding candidate when no base was recorded), only when the generator regime and audited dataset match. This describes observed effects, including regressions; it does not prove a causal mechanism. Every displayed error and paired change uses the five saved seeds."]
    changes = ranked_changes(iterations)[:3]
    if not changes:
        lines += ["", "No completed one-change comparison is available; three influential changes cannot be claimed."]
    for item in changes:
        before, after, key = item["previous"], item["current"], item["key"]
        direction = "decreased" if item["relative_change_of_means"] < 0 else "increased" if item["relative_change_of_means"] > 0 else "was unchanged"
        lines += ["", f"- **Iteration {after['iteration']}, compared with base iteration {before['iteration']} in the {number(after['config'].get('grip_force_max', 30.0))} N regime: `{key}` changed from `{before['config'].get(key)}` to `{after['config'].get(key)}`.** "
            f"Fitted-z validation NMSE {direction}: {mean_std(before['summary']['fitted_z']['one_step_nmse'])} → {mean_std(after['summary']['fitted_z']['one_step_nmse'])}. "
            f"Paired relative change (new minus previous, divided by previous): {mean_std(item['paired_relative_change'], percent=True)}. "
            f"The recorded explanatory hypothesis was: {_escape(after['hypothesis'])} This is a hypothesis, not an established causal explanation."]
    if 0 < len(changes) < 3:
        lines += ["", f"Only {len(changes)} eligible one-change comparisons were completed; no additional changes are invented."]
    lines += ["", "## Suspicious observations and limits", "",
        "- Ten preplanned randomized scripts are shared across all cubes. This exactly balances action commands against labels and helps F2, but covers limited action diversity. Validation holds out only two of those trajectories.",
        "- Validation cube identities also occur in weight training. Fitting a fresh z tests adaptation on held-out episodes of familiar cubes; only the final evaluation tests unseen identities.",
        "- Final scoring uses episodes 5–9, including three shared script settings (5–7) already seen during weight training on other cubes. Validation uses only settings 8–9. Thus the final test introduces new objects but includes more familiar trajectories; its pass after validation failed is not evidence of stronger generalization to new motions.",
        "- The state includes cube world orientation and relative position, but omits explicit cube orientation relative to the gripper. Relative pose therefore is not fully represented by a dedicated feature.",
        "- Joint effort is transmitted reaction wrench in the child-joint frame, sampled once per control step without averaging. The URDF joint-axis convention and PhysX force-then-torque layout support the chosen X projections, but the installed Python stubs do not expose the six-column getter implementation, so that component packing was not independently proved. This is not a calibrated hardware motor-torque measurement; no mass/friction leakage path was found.",
        "- The saved reset diagnostic reports identical reset states but small differences between repeated CPU trajectories. This infrastructure diagnostic was not a five-seed scientific result; it shows that bitwise-identical complete trajectories should not be assumed.",
        "- The narrowed-grip pilot reused one scene across cubes, while parallel full collection created a fresh environment per cube. Both used the same physical setup and commands. Exact equivalence of contact-cache state and numerical trajectories across those environment lifecycles was not established.",
        "- The pilot probe had only two held-out training cubes per fold and produced unstable, sometimes strongly negative R². This small-fold instability is distinct from the five-seed dynamics-model uncertainty. The full audit uses all training identities and substantially larger held-out-cube folds.",
        "- The installed planner crashed with NumPy 2.2.6 and ran with NumPy 1.26.4 isolated under clean/vendor. This is a saved local compatibility observation, not proof of a general library defect. Reproduction needs that runtime path.",
        "- Oracle is a supervised reference under the same optimization budget, not a guaranteed attainable error lower bound. If it fails to beat Plain, the locked gap-recovery claim cannot pass.",
        "- F2 detects property information in the sampled commands under the tested probe; matched scripts make the commands identical across cubes. Passing it does not certify every possible future controller.",
        "- The one-episode GRU is an empirical signal benchmark. It does not establish the maximum information any estimator could extract."]
    training = selected["runs"][0].get("training", {})
    parameter_counts = {kind: training.get(kind, {}).get("log", {}).get("num_parameters") for kind in ("plain", "history")}
    if all(isinstance(value, int) for value in parameter_counts.values()):
        lines += [f"- Recorded selected-model parameter counts are {parameter_counts['plain']:,} for Plain and {parameter_counts['history']:,} for History. The matched budget is {protocol['train_updates']:,} optimizer updates with the same minibatch size and data; it does not equalize parameter count, floating-point operations, or wall-clock time."]
    else:
        lines += ["- The matched training budget equalizes optimizer updates, minibatch size, and data. It does not equalize model parameter count, floating-point operations, or wall-clock time."]
    lines += ["", "## Exact reproduction commands", ""]
    lines += ["To replay the complete recorded candidate sequence, retrain every diagnostic probe and model, and run one final evaluation in a new copy, use the helper below. It restores the matching 30 N and 8 N data regimes automatically. The original split, properties, results, and final ledger are preserved. The helper was checked in plan-only mode; no extra scientific run was performed after the budget ended.",
        "", "```bash", f"cd {root.parent}",
        "PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.replay --plan-only",
        'replay_parent=$(mktemp -d /tmp/clean-all-XXXXXXXX)',
        'PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m clean.replay --destination "$replay_parent/experiment" --device cuda',
        "```", "",
        "Add `--recollect` to the last command to also regenerate all motion episodes in that separate copy, keeping the existing cube identities and true properties. Without that flag, every training/probe/final computation is repeated on the saved trajectories. The replay stops early if a reproduced validation candidate meets all criteria.", ""]
    lines.extend(_reproduction_commands(root, final["iteration"]))
    if len(regimes) > 1 and 30.0 in regimes:
        lines.extend(_old_regime_commands(root))
    lines += ["", "## Saved evidence", "",
        f"- Final scores: [{root / 'results/final/result.json'}]({root / 'results/final/result.json'})",
        f"- Selected validation result: [{root / 'runs' / ('iteration_%02d' % final['iteration']) / 'result.json'}]({root / 'runs' / ('iteration_%02d' % final['iteration']) / 'result.json'})",
        f"- Data audit: [{root / 'results/full_audit.json'}]({root / 'results/full_audit.json'})",
        f"- Independent final integrity audit: [{root / 'results/final_integrity_audit.json'}]({root / 'results/final_integrity_audit.json'})",
        f"- Complete reproduction helper: [{root / 'replay.py'}]({root / 'replay.py'})",
        f"- Locked definitions: [{root / 'data/protocol.json'}]({root / 'data/protocol.json'})", ""]
    return "\n".join(lines)


def generate_report(root: Path = ROOT) -> Path:
    import hashlib
    root = Path(root).resolve()
    final_path = root / "results/final/result.json"
    if not final_path.exists():
        raise FileNotFoundError("Final evaluation is not complete; RESULTS.md will not be changed")
    final = json.loads(final_path.read_text())
    iterations = [json.loads(path.read_text()) for path in sorted((root / "runs").glob("iteration_*/result.json"))]
    audit_path = root / "results/full_audit.json"
    audit = json.loads(audit_path.read_text())
    selected = next((result for result in iterations if result["iteration"] == final["iteration"]), None)
    if selected is None or selected.get("audit_sha256") != hashlib.sha256(audit_path.read_bytes()).hexdigest():
        raise ValueError("Final candidate and current data audit have different provenance; report not written")
    protocol = json.loads((root / "data/protocol.json").read_text())
    rendered = render_report(final, iterations, audit, protocol, root)
    temporary = root / ".RESULTS.md.tmp"
    temporary.write_text(rendered)
    output = root / "RESULTS.md"
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    print(generate_report(arguments.root))


if __name__ == "__main__":
    main()
