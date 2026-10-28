"""Matched checkpoint diagnostics, with whole-starting-seed bootstrap intervals.

This report never changes training. Reusing these diagnostics to choose a model
or budget makes them validation evidence, not reserved final-test evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

MODES = ("true", "fixed_friction", "wrong_friction", "fixed_mass", "wrong_mass", "fixed_both")
BINARY_METRICS = ("final_success", "sustained_success", "grasp_lost_after_lift", "lifted_ever")
FORCE_METRICS = ("lifted_force_n", "lifted_motor_limit_n")
BOOTSTRAP_SEED = 73520
BOOTSTRAP_DRAWS = 10000


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_key(row):
    return (row["grid"], float(row["mass_kg"]), float(row["friction"]), int(row["seed"]))


def row_key(row):
    return (str(row["checkpoint_label"]), *physical_key(row), row["mode"])


def checkpoint_sort_key(label):
    """Numeric labels preserve update ordering; k labels are also accepted."""
    match = re.search(r"(\d+(?:[._]\d+)?)\s*(k)?", str(label), re.IGNORECASE)
    if match:
        value = float(match.group(1).replace("_", "."))
        return (0, value * (1000 if match.group(2) else 1), str(label))
    return (1, 0, str(label))


def normalize_plan(plan, labels=None):
    """Accept the parent experiment plan or a runner's explicit per-run plan."""
    grids = {}
    for name, value in plan["grids"].items():
        if isinstance(value, list):
            grids[name] = {(float(r["mass_kg"]), float(r["friction"])) for r in value}
        else:
            grids[name] = {(float(mass), float(friction)) for mass in value["masses_kg"] for friction in value["frictions"]}
    modes, seeds = set(plan["modes"]), set(map(int, plan["seeds"]))
    if modes != set(MODES) or not seeds or any(not values for values in grids.values()):
        raise ValueError("Declared plan requires all six modes, nonempty physics grids and starting seeds")
    expected = sum(len(values) for values in grids.values()) * len(seeds) * len(MODES)
    declared = plan.get("expected_rows", plan.get("episodes_per_checkpoint", expected))
    if declared != expected:
        raise ValueError("Declared row count disagrees with planned physics × seeds × interventions")
    return dict(grids=grids, seeds=seeds, modes=modes,
                labels=set(map(str, plan.get("checkpoints", labels or []))),
                horizon=int(plan.get("steps", plan.get("episode_steps", 50))),
                expected_per_checkpoint=expected)


def validate_manifest_consistency(sources, parent_plan=None):
    """Do not combine different simulator/metric implementations or mislabeled weights."""
    required = {"checkpoint_label", "checkpoint_sha256", "source_sha256", "task_semantics"}
    for source in sources:
        manifest = source["manifest"]
        if required - set(manifest) or not manifest["source_sha256"]:
            raise ValueError("Runner manifest lacks checkpoint/source/task provenance")
        if str(manifest["checkpoint_label"]) not in source["checkpoint_labels"]:
            raise ValueError("Runner manifest checkpoint label disagrees with its episode rows")
    for key in ("source_sha256", "task_semantics"):
        if any(source["manifest"][key] != sources[0]["manifest"][key] for source in sources[1:]):
            raise ValueError(f"Runner {key} differs across checkpoints")
    if parent_plan is not None:
        for source in sources:
            label = source["checkpoint_labels"][0]
            if label not in parent_plan.get("checkpoints", {}):
                raise ValueError("Runner checkpoint label is absent from the pinned parent plan")
            pinned, manifest = parent_plan["checkpoints"][label], source["manifest"]
            if "sha256" in pinned and manifest.get("checkpoint_sha256") != pinned["sha256"]:
                raise ValueError("Runner checkpoint bytes do not match the pinned parent plan")
            if "path" in pinned and Path(manifest.get("checkpoint", "")).resolve() != Path(pinned["path"]).resolve():
                raise ValueError("Runner checkpoint path disagrees with the pinned parent plan")
            if "completed_learning_cycles" in pinned and manifest.get("completed_rollout_cycles") != pinned["completed_learning_cycles"]:
                raise ValueError("Runner completed-cycle count disagrees with the pinned parent plan")


def validate_rows(rows, expected_plan=None):
    """Require a complete crossed matched design and identical physical resets."""
    if not rows:
        raise ValueError("No checkpoint rows supplied")
    indexed, fingerprints, physics_fingerprints = {}, {}, {}
    grids, labels = defaultdict(set), set()
    seeds_by_grid = defaultdict(set)
    for row in rows:
        required = {"checkpoint_label", "grid", "mass_kg", "friction", "seed", "mode", "episode_id",
                    "reset_fingerprint", "lifted_steps", "reward_sum", *BINARY_METRICS}
        missing = required - set(row)
        if missing:
            raise ValueError(f"Missing episode fields: {sorted(missing)}")
        if row["mode"] not in MODES:
            raise ValueError(f"Unsupported input intervention: {row['mode']}")
        if any(not np.isfinite(float(row[k])) or float(row[k]) <= 0 for k in ("mass_kg", "friction")):
            raise ValueError("Physical parameters must be finite and positive")
        if isinstance(row["seed"], bool) or int(row["seed"]) != row["seed"]:
            raise ValueError("Starting seed must be an integer")
        if (isinstance(row["lifted_steps"], bool) or int(row["lifted_steps"]) != row["lifted_steps"]
                or row["lifted_steps"] < 0 or not np.isfinite(float(row["reward_sum"]))):
            raise ValueError("Invalid lifted duration or reward")
        for metric in BINARY_METRICS:
            if row[metric] not in (False, True, 0, 1):
                raise ValueError(f"Invalid binary outcome: {metric}")
        if bool(row["sustained_success"]) and not bool(row["final_success"]):
            raise ValueError("Sustained success requires final success")
        if bool(row["lifted_ever"]) != (row["lifted_steps"] > 0):
            raise ValueError("Lifted flag disagrees with measured lifted duration")
        for metric in FORCE_METRICS:
            if row.get(metric) is not None and (not np.isfinite(float(row[metric])) or row[metric] < 0):
                raise ValueError(f"Invalid lifted-contact measurement: {metric}")
        key = row_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate checkpoint/condition/seed/intervention row: {key}")
        indexed[key] = row
        physical = physical_key(row)
        fingerprint = row["reset_fingerprint"]
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("A nonempty physical reset fingerprint is required")
        if fingerprints.setdefault(physical, fingerprint) != fingerprint:
            raise ValueError(f"Physical reset differs across checkpoints or input interventions: {physical}")
        if "physics_fingerprint" in row and physics_fingerprints.setdefault(physical, row["physics_fingerprint"]) != row["physics_fingerprint"]:
            raise ValueError(f"Physics/controller fingerprint differs across checkpoints or input interventions: {physical}")
        labels.add(str(row["checkpoint_label"]))
        grids[row["grid"]].add((float(row["mass_kg"]), float(row["friction"])))
        seeds_by_grid[row["grid"]].add(int(row["seed"]))
    if len(labels) < 2:
        raise ValueError("Need at least two checkpoint labels")
    if expected_plan is not None:
        expected = expected_plan
        if expected["labels"] and labels != expected["labels"]:
            raise ValueError("Observed checkpoint labels disagree with the declared experiment plan")
        if dict(grids) != expected["grids"]:
            raise ValueError("Observed physical grids disagree with the declared experiment plan")
        if any(seeds != expected["seeds"] for seeds in seeds_by_grid.values()):
            raise ValueError("Observed starting seeds disagree with the declared experiment plan")
        if any(sum(str(r["checkpoint_label"]) == label for r in rows) != expected["expected_per_checkpoint"] for label in labels):
            raise ValueError("Observed episode count disagrees with the declared experiment plan")
        for row in rows:
            if row["lifted_steps"] > expected["horizon"]:
                raise ValueError("Lifted duration exceeds the declared episode horizon")
            if row["lifted_steps"] > 0 and any(row.get(metric) is None for metric in FORCE_METRICS):
                raise ValueError("Qualifying lifted episodes require finite measured force and motor limit")
            if "force_step_count" in row and row["force_step_count"] != row["lifted_steps"]:
                raise ValueError("Force duration denominator disagrees with lifted steps")
            if "force_sample_count" in row and row["force_sample_count"] != 2 * row["lifted_steps"]:
                raise ValueError("Force sample denominator disagrees with two-finger lifted samples")
    # Every condition uses the same seed blocks, including across grids.
    all_seeds = set.union(*seeds_by_grid.values())
    if any(seeds != all_seeds for seeds in seeds_by_grid.values()):
        raise ValueError("Starting-seed blocks differ across physics grids")
    for label in labels:
        for grid, physics in grids.items():
            for mass, friction in physics:
                for seed in all_seeds:
                    for mode in MODES:
                        key = (label, grid, mass, friction, seed, mode)
                        if key not in indexed:
                            raise ValueError(f"Incomplete matched checkpoint coverage: {key}")
    return indexed, sorted(labels, key=checkpoint_sort_key), dict(grids), sorted(all_seeds)


def seed_block_difference(pairs, metric, *, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS):
    """Bootstrap matched differences, retaining all cases for each starting seed.

    Common-lift comparisons condition on both episodes having lifted contact.
    Eligible counts may differ by seed; the estimate is an equal-seed average,
    and the pooled episode average is reported separately.
    """
    differences, blocks = [], defaultdict(list)
    excluded = left_steps = right_steps = 0
    for left, right in pairs:
        if physical_key(left) != physical_key(right):
            raise ValueError("Paired comparison changed the physical case or starting seed")
        if metric in FORCE_METRICS and not (left["lifted_steps"] > 0 and right["lifted_steps"] > 0
                                            and left.get(metric) is not None and right.get(metric) is not None):
            excluded += 1
            continue
        difference = float(left[metric]) - float(right[metric])
        if metric in FORCE_METRICS:
            left_steps += left["lifted_steps"]
            right_steps += right["lifted_steps"]
        differences.append(difference)
        blocks[int(left["seed"])].append(difference)
    means = np.asarray([np.mean(blocks[s]) for s in sorted(blocks)])
    interval = None
    if len(means) >= 2:
        rng = np.random.default_rng(seed)
        resampled = means[rng.integers(len(means), size=(draws, len(means)))].mean(axis=1)
        interval = np.quantile(resampled, [.025, .975]).tolist()
    result = dict(metric=metric, direction="left minus right", pairs=len(differences),
                  excluded_pairs=excluded, seed_blocks=len(means),
                  seed_block_pair_counts={str(s): len(blocks[s]) for s in sorted(blocks)},
                  difference=float(means.mean()) if len(means) else None,
                  pooled_episode_difference=float(np.mean(differences)) if differences else None,
                  paired_seed_bootstrap_95_interval=interval,
                  bootstrap=dict(seed=seed, draws=draws, unit="whole starting-seed block"),
                  conditional_on_common_lift=metric in FORCE_METRICS,
                  interval_support="descriptive" if interval is not None else "insufficient seed blocks")
    if metric in ("final_success", "sustained_success", "lifted_ever"):
        result["left_better_supported"] = interval is not None and interval[0] > 0
        result["right_better_supported"] = interval is not None and interval[1] < 0
    if metric in FORCE_METRICS:
        result["left_qualifying_lifted_steps"] = left_steps
        result["right_qualifying_lifted_steps"] = right_steps
        result["left_two_finger_samples"] = 2 * left_steps
        result["right_two_finger_samples"] = 2 * right_steps
        result["limitation"] = ("Only matched episodes that lifted under BOTH alternatives contribute. "
                                 "This conditions on survivors and does not establish whole-task force efficiency. "
                                 "Qualifying contact durations can differ within those common episodes.")
    return result


def grouped_metrics(rows):
    result = dict(episodes=len(rows), physical_conditions=len({physical_key(r)[1:3] for r in rows}),
                  seed_blocks=len({r["seed"] for r in rows}),
                  **{metric: float(np.mean([r[metric] for r in rows])) for metric in BINARY_METRICS},
                  mean_lifted_steps=float(np.mean([r["lifted_steps"] for r in rows])),
                  reward_sum=float(np.mean([r["reward_sum"] for r in rows])))
    for metric in FORCE_METRICS:
        eligible = [r for r in rows if r["lifted_steps"] > 0 and r.get(metric) is not None]
        result[metric] = float(np.mean([r[metric] for r in eligible])) if eligible else None
        result[metric + "_episodes"] = len(eligible)
        result[metric + "_qualifying_steps"] = sum(r["lifted_steps"] for r in eligible)
    result["force_scope"] = "Descriptive episode means conditional on lifted contact; cross-alternative force differences use common-lift pairs separately."
    return result


def budget_interpretation(checkpoint_comparisons, labels, grids, meaningful_gain=.05):
    """A stated descriptive margin informs interpretation, never training actions."""
    latest, previous = labels[-1], labels[-2]
    states = {}
    for grid in sorted(grids):
        comparison = next(r for r in checkpoint_comparisons if r["left_checkpoint"] == latest
                          and r["right_checkpoint"] == previous and r["grid"] == grid)
        primary = comparison["metrics"]["final_success"]
        interval, difference = primary["paired_seed_bootstrap_95_interval"], primary["difference"]
        if interval is None:
            state = "insufficient independent starts"
        elif interval[1] < 0:
            state = "supported recent regression"
        elif interval[0] >= meaningful_gain:
            state = "supported recent practical improvement"
        elif interval[0] > 0:
            state = "supported recent improvement; practical magnitude uncertain"
        elif interval[1] < meaningful_gain:
            state = "recent practical improvement not supported on these cases"
        else:
            state = "recent trend uncertain"
        states[grid] = dict(state=state, final_success_difference=difference, interval=interval)
    if any("regression" in item["state"] for item in states.values()):
        verdict = "Recent checkpoint quality regressed on at least one tested grid; more updates alone are not established as useful."
    elif any("supported recent practical improvement" == item["state"] for item in states.values()):
        verdict = "Recent updates produced a practical gain on at least one tested grid; continued monitored training has some empirical support."
    elif any(item["state"] == "supported recent improvement; practical magnitude uncertain" for item in states.values()):
        verdict = "Recent updates improved tested success, but their practical magnitude and the value of a 50,000-cycle budget remain uncertain."
    elif all(item["state"] == "recent practical improvement not supported on these cases" for item in states.values()):
        verdict = "The recent interval provides no demonstrated practical success gain on either tested grid; the 50,000-cycle budget remains unvalidated."
    else:
        verdict = "The recent learning trend is uncertain with these starts; this comparison does not establish whether 50,000 cycles are worthwhile."
    return dict(latest_checkpoint=latest, previous_checkpoint=previous, grids=states, interpretation=verdict,
                practical_success_margin=meaningful_gain,
                margin_scope="Descriptive five-percentage-point margin stated for this report; not a preregistered final-test threshold.",
                interpretation_scope="Primary final success only; inspect sustained holding, grasp loss, conditioning and common-lift force evidence separately.",
                fifty_thousand_cycles_necessary_or_sufficient_demonstrated=False,
                training_action="none: reporting only; no automatic stop, restart or hyperparameter change")


def analyze(rows, expected_plan=None):
    indexed, labels, grids, seeds = validate_rows(rows, expected_plan)
    aggregates, interventions, checkpoint_comparisons = [], [], []
    for label in labels:
        for grid in sorted(grids):
            for mode in MODES:
                selected = [r for r in rows if str(r["checkpoint_label"]) == label and r["grid"] == grid and r["mode"] == mode]
                aggregates.append(dict(checkpoint_label=label, grid=grid, mode=mode, metrics=grouped_metrics(selected)))
            for altered in MODES[1:]:
                pairs = [(indexed[(label, grid, mass, friction, seed, "true")],
                          indexed[(label, grid, mass, friction, seed, altered)])
                         for mass, friction in sorted(grids[grid]) for seed in seeds]
                interventions.append(dict(checkpoint_label=label, grid=grid, left_mode="true", right_mode=altered,
                                          metrics={metric: seed_block_difference(pairs, metric)
                                                   for metric in (*BINARY_METRICS, *FORCE_METRICS)}))
    for later_index in range(1, len(labels)):
        for earlier in labels[:later_index]:
            later = labels[later_index]
            for grid in sorted(grids):
                pairs = [(indexed[(later, grid, mass, friction, seed, "true")],
                          indexed[(earlier, grid, mass, friction, seed, "true")])
                         for mass, friction in sorted(grids[grid]) for seed in seeds]
                checkpoint_comparisons.append(dict(left_checkpoint=later, right_checkpoint=earlier, grid=grid,
                                                   mode="true", metrics={metric: seed_block_difference(pairs, metric)
                                                         for metric in (*BINARY_METRICS, *FORCE_METRICS)}))
    return dict(checkpoints=labels, seed_blocks=seeds, rows=len(rows),
                grids={grid: dict(physical_conditions=len(values), settings=[list(v) for v in sorted(values)])
                       for grid, values in sorted(grids.items())},
                validation=dict(complete_pair_coverage=True, identical_physical_resets=True,
                                all_six_input_interventions=True,
                                declared_plan_coverage_verified=expected_plan is not None),
                aggregates=aggregates, checkpoint_comparisons=checkpoint_comparisons,
                true_vs_altered_inputs=interventions,
                budget_assessment=budget_interpretation(checkpoint_comparisons, labels, grids),
                limitations=["CPU simulation preserves the trained native carry task's randomized goal XY and floating-height rule; backend/RNG differences mean this is not GPU final-test evidence.",
                             "Results are conditional on tested physics and ten diagnostic starting seeds when the full planned probe is supplied.",
                             "Whole-seed bootstrap retains all physical conditions; the physics grid is not sampled as independent episodes.",
                             "These diagnostic seeds may inform model/budget selection; reserved final-assessment seeds remain separate.",
                             "Intervals are descriptive and are not adjusted for inspecting multiple checkpoints, physics grids and input interventions.",
                             "Grasp loss after lift describes behavior and does not establish slipping or a physical cause.",
                             "Reward, input sensitivity or lower force alone cannot establish useful physics conditioning.",
                             "No training process is stopped or changed by this report."])


def render_markdown(report):
    lines = ["# Checkpoint learning diagnostic", "", report["budget_assessment"]["interpretation"], "",
             "This diagnostic does not establish that 50,000 cycles are necessary or sufficient. Training was not changed.", "",
             "The budget heuristic uses final success only; inspect holding, grasp loss, input conditioning and common-lift force evidence separately.", "",
             "## True-input task outcomes", "",
             "| Checkpoint | Grid | Episodes | Final success | Sustained success | Grasp loss after lift | Lifted |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["aggregates"]:
        if row["mode"] != "true":
            continue
        m = row["metrics"]
        lines.append(f"| {row['checkpoint_label']} | {row['grid']} | {m['episodes']} | {m['final_success']:.1%} | {m['sustained_success']:.1%} | {m['grasp_lost_after_lift']:.1%} | {m['lifted_ever']:.1%} |")
    if report.get("checkpoint_metadata"):
        lines += ["", "Saved checkpoint counters and completed learning updates differ:", ""]
        for label in report["checkpoints"]:
            metadata = report["checkpoint_metadata"][label]
            completed = metadata.get("completed_learning_cycles")
            if completed is not None:
                lines.append(f"- {label}: {completed:,} completed learning cycles ({metadata.get('transitions', completed * 3000):,} transitions).")
    lines += ["", "## Matched checkpoint changes", "",
              "Differences are later minus earlier, with a 95% whole-starting-seed bootstrap interval.", "",
              "| Later − earlier | Grid | Final success change | 95% interval | Sustained change |",
              "|---|---|---:|---|---:|"]
    for row in report["checkpoint_comparisons"]:
        m, sustained = row["metrics"]["final_success"], row["metrics"]["sustained_success"]
        ci = m["paired_seed_bootstrap_95_interval"]
        ci_text = f"[{ci[0]:+.1%}, {ci[1]:+.1%}]" if ci is not None else "Insufficient starts"
        lines.append(f"| {row['left_checkpoint']} − {row['right_checkpoint']} | {row['grid']} | {m['difference']:+.1%} | {ci_text} | {sustained['difference']:+.1%} |")
    lines += ["", "## Does accurate supplied physics help?", "",
              "Each comparison changes only supplied policy inputs and pairs identical physical resets.", "",
              "| Checkpoint | Grid | True − altered inputs | Success change | 95% interval |",
              "|---|---|---|---:|---|"]
    for row in report["true_vs_altered_inputs"]:
        m = row["metrics"]["final_success"]
        ci = m["paired_seed_bootstrap_95_interval"]
        ci_text = f"[{ci[0]:+.1%}, {ci[1]:+.1%}]" if ci is not None else "Insufficient starts"
        lines.append(f"| {row['checkpoint_label']} | {row['grid']} | true − {row['right_mode']} | {m['difference']:+.1%} | {ci_text} |")
    lines += ["", "## Measured force", "",
              "Cross-checkpoint force differences include only pairs with qualifying lifted contact under both alternatives. Different survivor sets are never compared as evidence of improved force efficiency. The contact durations within common episodes can still differ; this is conditional evidence.", "",
              "| Later − earlier | Grid | Common-lift pairs | Seed blocks | Measured force change (N) | 95% interval (N) | Motor-limit change (N) |",
              "|---|---|---:|---:|---:|---|---:|"]
    for row in report["checkpoint_comparisons"]:
        force = row["metrics"]["lifted_force_n"]
        limit = row["metrics"]["lifted_motor_limit_n"]
        ci = force["paired_seed_bootstrap_95_interval"]
        ci_text = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci is not None else "Insufficient starts"
        force_text = f"{force['difference']:+.3f}" if force["difference"] is not None else "Unavailable"
        limit_text = f"{limit['difference']:+.3f}" if limit["difference"] is not None else "Unavailable"
        lines.append(f"| {row['left_checkpoint']} − {row['right_checkpoint']} | {row['grid']} | {force['pairs']} | {force['seed_blocks']} | {force_text} | {ci_text} | {limit_text} |")
    lines += ["", "All per-mode force summaries and qualifying step/sample denominators are in comparison.json. Lower conditional force alone does not establish a better task policy.", "",
              "## Limits", ""]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def plot_report(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(report["grids"]), figsize=(5 * len(report["grids"]), 3.7), squeeze=False)
    for ax, grid in zip(axes[0], report["grids"]):
        x = [report.get("checkpoint_metadata", {}).get(label, {}).get("completed_learning_cycles") for label in report["checkpoints"]]
        numerical_x = all(value is not None for value in x)
        for metric, name in (("final_success", "Final success"), ("sustained_success", "Sustained success")):
            values = [next(r["metrics"][metric] for r in report["aggregates"] if r["checkpoint_label"] == label
                           and r["grid"] == grid and r["mode"] == "true") for label in report["checkpoints"]]
            ax.plot(x if numerical_x else report["checkpoints"], values, marker="o", label=name)
        ax.set_title(grid)
        ax.set_xlabel("Completed learning cycles" if numerical_x else "Saved policy checkpoint")
        ax.set_ylabel("Fraction of matched episodes")
        ax.set_ylim(0, 1)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("CPU carry-task checkpoint diagnostic — true physics inputs")
    fig.tight_layout()
    fig.savefig(output / "checkpoint_learning.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="Parent experiment plan declaring all three checkpoints and complete grids/seeds")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if len(args.inputs) != 3:
        parser.error("This diagnostic requires exactly three checkpoint input directories")
    rows, sources = [], []
    runner_plans = []
    for directory in args.inputs:
        path = directory.resolve() / "rows.jsonl"
        loaded = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        labels = {str(r["checkpoint_label"]) for r in loaded}
        if len(labels) != 1:
            raise ValueError(f"Each input directory must contain exactly one checkpoint label: {directory}")
        if any(labels & set(source["checkpoint_labels"]) for source in sources):
            raise ValueError("Different input directories repeat a checkpoint label")
        source = dict(directory=str(directory.resolve()), rows_sha256=file_hash(path), checkpoint_labels=sorted(labels))
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Checkpoint probe lacks a manifest: {directory}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"Checkpoint probe is incomplete: {directory}")
        source["manifest_sha256"] = file_hash(manifest_path)
        source["manifest"] = manifest
        if "checkpoint_label" in manifest and str(manifest["checkpoint_label"]) not in labels:
            raise ValueError("Runner manifest checkpoint label disagrees with its episode rows")
        if "plan" in manifest:
            runner_plan = normalize_plan(manifest["plan"])
            if manifest.get("planned_rows", runner_plan["expected_per_checkpoint"]) != runner_plan["expected_per_checkpoint"]:
                raise ValueError("Runner planned_rows disagrees with its plan")
            if manifest.get("completed_rows", len(loaded)) != runner_plan["expected_per_checkpoint"]:
                raise ValueError("Runner completed_rows disagrees with its plan")
            runner_plans.append(runner_plan)
        sources.append(source)
        rows.extend(loaded)
    if args.plan:
        raw_plan = json.loads(args.plan.read_text())
        expected = normalize_plan(raw_plan)
        validate_manifest_consistency(sources, raw_plan)
    elif len(runner_plans) == 3:
        expected = dict(runner_plans[0], labels={source["checkpoint_labels"][0] for source in sources})
        validate_manifest_consistency(sources)
    else:
        raise ValueError("Pass --plan or supply complete declared plans in every runner manifest")
    for runner_plan in runner_plans:
        if any(runner_plan[key] != expected[key] for key in ("grids", "seeds", "modes", "horizon", "expected_per_checkpoint")):
            raise ValueError("Runner and parent experiment plans disagree")
    report = analyze(rows, expected)
    report["source_inputs"] = sources
    report["checkpoint_metadata"] = {source["checkpoint_labels"][0]: dict(
        checkpoint_sha256=source["manifest"].get("checkpoint_sha256"),
        completed_learning_cycles=source["manifest"].get("completed_rollout_cycles"),
        transitions=source["manifest"].get("completed_rollout_cycles", 0) * 3000
        if source["manifest"].get("completed_rollout_cycles") is not None else None)
        for source in sources}
    report["report_source_sha256"] = file_hash(__file__)
    if args.plan:
        report["declared_experiment_plan"] = dict(path=str(args.plan.resolve()), sha256=file_hash(args.plan), plan=raw_plan)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.output / "comparison.md").write_text(render_markdown(report))
    if not args.no_plot:
        plot_report(report, args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), budget_assessment=report["budget_assessment"]), indent=2))
    return report


if __name__ == "__main__":
    main()
