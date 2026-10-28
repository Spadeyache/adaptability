"""Audit completed work independently of positive or negative research results.

A green status alone does not prove 50,000 learning cycles or complete matched
evaluations. This auditor checks counters, model tensors, hashes, coverage and
raw-record alignment. It never starts training or evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np


POLICY_MODES = ("true", "fixed_friction", "wrong_friction", "fixed_mass", "wrong_mass", "fixed_both")
RETRY_MODES = ("fixed", "adaptive", "true")
CAPABILITY_STATUSES = {
    "supported_on_tested_cases", "supported_on_historical_held_out_conditions",
    "supported_on_tested_grid", "partially_supported", "not_demonstrated",
    "inconclusive", "unsupported",
}


@dataclass(frozen=True)
class AuditExpectations:
    minimum_iterations: int = 50000
    num_envs: int = 60
    rollout_steps: int = 50
    policy_seed_start: int = 91000
    policy_seed_count: int = 10
    retry_conditions: int = 9
    retry_seed_blocks: int = 10
    retry_attempts: int = 3
    retry_seed_start: int = 54000
    transfer_masses: int = 4
    transfer_frictions: int = 4
    transfer_attempts: int = 5
    transfer_seed_start: int = 64000
    transfer_mass_values: tuple[float, ...] = (.08, .16, .32, .48)
    transfer_friction_values: tuple[float, ...] = (.25, .45, .65, .85)
    transfer_supplied_mass: float = .1
    transfer_supplied_friction: float = .4
    require_guarded_updates: bool = False


class Incomplete(Exception):
    pass


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def physics_key(mass, friction):
    return (round(float(mass), 6), round(float(friction), 6))


def check_observable_contract(manifest):
    boundary = manifest.get("estimator_input_boundary", "")
    require("input__" in boundary and "timestamp" in boundary and "only" in boundary,
            "Estimator input boundary is missing its observable-only contract")


def validate_raw_episodes(directory, rows, *, retries=False):
    """Check saved outcomes, physical labels and matching fresh initial states."""
    identifiers = [row["episode_id"] for row in rows]
    require(len(set(identifiers)) == len(identifiers), "Duplicate episode IDs")
    expected_files = {identifier + ".npz" for identifier in identifiers}
    actual_files = {path.name for path in (directory / "episodes").glob("*.npz")}
    require(actual_files == expected_files, "Raw episode files do not exactly cover attempt records")
    initial_states = {}
    for row in rows:
        with np.load(directory / "episodes" / (row["episode_id"] + ".npz"), allow_pickle=False) as data:
            require(bool(data["meta__fresh_scene_reset"]), "A recorded attempt lacks a fresh physics reset")
            require(int(data["label__seed"]) == row["seed"], "Raw episode seed disagrees with its attempt row")
            require(bool(data["label__held_goal_success_final"]) == row["success"],
                    "Raw success disagrees with the reported outcome")
            require(np.allclose([float(data["label__mass_kg"]),
                                 float(data["label__object_static_friction"]),
                                 float(data["label__object_dynamic_friction"])],
                                [row["actual_mass_kg"], row["actual_friction"], row["actual_friction"]]),
                    "Raw physical labels disagree with recorded mass/friction")
            if retries:
                key = (row["condition_index"], row["sequence_index"], row["attempt"])
                fingerprint = str(data["label__reset_state_hash"])
                require(bool(fingerprint), "Empty fresh-state fingerprint")
                prior = initial_states.setdefault(key, fingerprint)
                require(prior == fingerprint, "Matched retry methods start from different physical states")
    return {"raw_episodes": len(rows), "matched_initial_states": len(initial_states)}


def validate_guarded_execution(directory, manifest):
    """Check post-update guard evidence for every actual rollout, streaming logs."""
    settings=manifest['settings']
    require(settings.get('guard_policy_updates') is True and manifest.get('trust_region',{}).get('enabled'),
            'The correction did not enable guarded optimizer updates')
    require(digest(Path(directory)/'trust_region.py')==manifest['trust_region_sha256'],
            'The executed guard source does not match its recorded identity')
    batch=settings['num_envs']*settings['num_steps']
    limit=settings['max_exact_kl']
    count,accepted,maximum=0,0,0.
    with (Path(directory)/'trust_region.jsonl').open() as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line);count+=1
            require(row['iteration']==count and row['global_step']==count*batch,
                    'Guard execution counters are missing, duplicated or out of order')
            require(row['rollout_observations']==batch and row['effective_actions']==5 and
                    row['optimization_semantics']==manifest['optimization_semantics'],
                    'Guard evidence does not cover the declared full five-action rollout')
            for key in ('max_accepted_exact_kl','rollout_exact_kl'):
                value=row[key]
                require(math.isfinite(value) and 0<=value<=limit+1e-8,
                        'A recorded accepted policy update exceeds the exact KL bound')
            accepted+=row['accepted_updates']
            maximum=max(maximum,row['max_accepted_exact_kl'])
    if count<manifest['planned_iterations']:
        if manifest.get('status')!='complete':
            raise Incomplete(f'Guard recorded {count} of {manifest["planned_iterations"]} rollouts')
        raise ValueError('Completed training lacks full guard execution coverage')
    require(count==manifest['planned_iterations'] and accepted>0,
            'Guard execution has an inconsistent count or no accepted learning updates')
    return dict(rollouts=count,accepted_updates=accepted,max_accepted_exact_kl=maximum,limit=limit)


def audit(pipeline_root, *, transfer=None, combined_assessment=None, expectations=None):
    """Return requirement-level proved/incomplete/contradicted evidence.

    Optional transfer/combined-assessment paths add required checks when supplied.
    A negative capability result is a valid completed scientific evaluation.
    """
    root = Path(pipeline_root).resolve()
    expected = expectations or AuditExpectations()
    requirements, cache = [], {}

    def check(identifier, operation):
        try:
            evidence = operation()
            requirements.append(dict(id=identifier, status="proved", evidence=evidence))
        except (FileNotFoundError, Incomplete) as error:
            requirements.append(dict(id=identifier, status="incomplete", reason=str(error)))
        except Exception as error:
            requirements.append(dict(id=identifier, status="contradicted", reason=str(error),
                                     error_type=type(error).__name__))

    def state():
        if "state" not in cache:
            cache["state"] = read_json(root / "status.json")
        return cache["state"]

    def training_dir():
        path = state().get("training_run")
        if not path:
            raise Incomplete("No training run has been recorded")
        return Path(path)

    def training_manifest():
        return read_json(training_dir() / "manifest.json")

    def policy_path():
        return training_dir() / "final_ckpt.pt"

    def estimator_path():
        return Path(state().get("estimator_checkpoint") or root / "estimator.pt")

    def stage_dir(name, default):
        return Path(state().get(name) or root / default)

    def pipeline_status():
        value = state()
        if value.get("status") != "complete":
            raise Incomplete(f"Pipeline stage is {value.get('status')!r}")
        return dict(path=str(root / "status.json"), status="complete")

    def budget():
        manifest = training_manifest()
        settings = manifest["settings"]
        planned = int(manifest["planned_iterations"])
        require(planned >= expected.minimum_iterations, "Planned learning cycles are below the requested budget")
        require(settings["num_envs"] == expected.num_envs and settings["num_steps"] == expected.rollout_steps,
                "Training parallelism/rollout size differs from the audited contract")
        transitions = planned * settings["num_envs"] * settings["num_steps"]
        require(settings["total_timesteps"] == transitions, "Declared transition budget is inconsistent")
        require(state()["planned_iterations"] == planned, "Supervisor and training iteration budgets disagree")
        require(manifest.get("observations") == 46 and settings.get("condition_on_friction") is True,
                "Training is not the explicit 46-input friction policy")
        require(settings.get("friction_profile") == "independent_grid", "Mass/friction training is not independently crossed")
        return dict(iterations=planned, transitions=transitions, num_envs=settings["num_envs"],
                    rollout_steps=settings["num_steps"])

    def execution():
        manifest = training_manifest()
        planned = int(manifest["planned_iterations"])
        last_epoch, last_step = 0, None
        saved = False
        pattern = re.compile(r"\bEpoch:\s*(\d+),\s*global_step=(\d+)")
        with (training_dir() / "console.log").open(errors="replace") as stream:
            for line in stream:
                match = pattern.search(line)
                if match and int(match[1]) >= last_epoch:
                    last_epoch, last_step = int(match[1]), int(match[2])
                saved |= "model saved to" in line and "final_ckpt.pt" in line
        complete = manifest.get("status") == "complete"
        if last_epoch < planned:
            message = f"Console reached Epoch {last_epoch}; {planned} cycles are required"
            if complete:
                raise ValueError("Completed manifest contradicts its log: " + message)
            raise Incomplete(message)
        batch = manifest["settings"]["num_envs"] * manifest["settings"]["num_steps"]
        require(last_step == (last_epoch - 1) * batch,
                "Logged pre-rollout transition counter is inconsistent with the epoch")
        if not complete:
            raise Incomplete("Final epoch was entered but successful process completion is unproven")
        require(manifest.get("exit_code") == 0, "Training did not record exit code zero")
        require(saved, "Training log does not record writing its final checkpoint")
        return dict(log=str(training_dir() / "console.log"), final_epoch=last_epoch,
                    pre_rollout_global_step=last_step, exit_code=0)

    def weights():
        import torch
        saved = torch.load(policy_path(), map_location="cpu", weights_only=True)
        require(isinstance(saved, dict) and bool(saved), "Final checkpoint is not a model state dict")
        require(all(isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all().item()
                    for tensor in saved.values()), "Final policy contains non-tensor or nonfinite parameters")
        for name in ("actor_mean.0.weight", "strength_mean.0.weight", "critic.0.weight"):
            require(name in saved and saved[name].ndim == 2 and saved[name].shape[1] == 46,
                    f"Final checkpoint does not have 46 inputs in {name}")
        return dict(path=str(policy_path()), sha256=digest(policy_path()), tensors=len(saved), observations=46)

    def guarded_updates():
        manifest=training_manifest()
        enabled=manifest['settings'].get('guard_policy_updates',False)
        if not enabled:
            require(not expected.require_guarded_updates,'A guarded corrective run was required')
            return dict(enabled=False,scope='Original unguarded optimization; no correction is claimed.')
        return dict(enabled=True,**validate_guarded_execution(training_dir(),manifest))

    def frozen_sources():
        hashes = read_json(root / "source_sha256.json")
        require(bool(hashes), "Source snapshot hash inventory is empty")
        for name, fingerprint in hashes.items():
            require(digest(root / "source" / name) == fingerprint, f"Frozen source changed: {name}")
        return dict(files=len(hashes), inventory=str(root / "source_sha256.json"))

    def historical_estimator():
        import torch
        report = read_json(root / "estimator_evaluation.json")
        require(report.get("checkpoint_sha256") == digest(estimator_path()),
                "Historical estimator report does not match the pinned weights")
        require(report.get("split") == "test" and report.get("held_conditions") == expected.retry_conditions,
                "Historical estimator report is not the declared held-condition test")
        require(len(report["conditions"]) == expected.retry_conditions, "Historical condition coverage is incomplete")
        require(all(str(count) in report["attempts"] for count in (1, 2, 3, 5)),
                "Historical estimator evidence-prefix metrics are incomplete")
        for metrics in report["attempts"].values():
            require(np.asarray(metrics["mae"]).shape == (2,) and np.isfinite(metrics["mae"]).all(),
                    "Historical parameter-error metrics are invalid")
        saved = torch.load(estimator_path(), map_location="cpu", weights_only=True)
        feature_names = saved.get("feature_names", [])
        require(bool(feature_names) and all(name.startswith("input__") for name in feature_names),
                "Estimator lacks an explicit observable feature contract")
        require(not any("reported_mass" in name or "friction" in name or "object_weight" in name
                        for name in feature_names), "Estimator features contain supplied physical parameters")
        require(all(torch.isfinite(tensor).all().item() for tensor in saved["state_dict"].values()),
                "Estimator has nonfinite weights")
        return dict(path=str(root / "estimator_evaluation.json"), checkpoint_sha256=digest(estimator_path()),
                    held_conditions=report["held_conditions"], scientific_result_not_required_positive=True)

    def final_policy():
        directory = stage_dir("final_evaluation", "final_evaluation")
        report = read_json(directory / "summary.json")
        require(report.get("checkpoint_sha256") == digest(policy_path()), "Policy evaluation checkpoint hash mismatch")
        seeds = list(range(expected.policy_seed_start, expected.policy_seed_start + expected.policy_seed_count))
        require(report.get("seeds") == seeds, "Reserved final evaluation seeds are missing or different")
        fingerprints = report.get("reset_fingerprints", {})
        require(set(fingerprints) == set(map(str, seeds)) and all(fingerprints.values()),
                "Fresh-state fingerprints do not cover every reserved seed")
        settings = training_manifest()["settings"]
        grid = {physics_key(mass, mu) for mass, mu in
                itertools.product(settings["masses_kg"], settings["cube_friction_coefficients"])}
        require({physics_key(mass, mu) for mass, mu in
                 itertools.product(report["masses_kg"], report["frictions"])} == grid,
                "Final policy evaluation grid differs from training's declared physics grid")
        rows = read_json(directory / "episodes.json")
        keys = [(row["mode"], row["seed"], physics_key(row["mass_kg"], row["friction"])) for row in rows]
        expected_keys = set(itertools.product(POLICY_MODES, seeds, grid))
        require(len(keys) == len(set(keys)) and set(keys) == expected_keys,
                "Six-mode per-grid paired episode coverage is incomplete or duplicated")
        require(set(POLICY_MODES) <= set(report["metrics"]), "Policy summary omits an input intervention")
        for mode in POLICY_MODES:
            success = np.mean([row["final_success"] for row in rows if row["mode"] == mode])
            require(np.isclose(success, report["metrics"][mode]["final_success"]),
                    "Policy summary success differs from its episode records")
        for mode in POLICY_MODES[1:]:
            comparison = report["paired_comparisons"][mode]
            require(comparison.get("metric") == "final_success" and
                    comparison["n_pairs"] == len(seeds) * len(grid) and
                    comparison["n_seed_blocks"] == len(seeds), "Policy paired comparison coverage/metric is inconsistent")
        traces = read_jsonl(directory / "traces.jsonl")
        horizon = int(report["horizon"])
        require(horizon == expected.rollout_steps, "Policy evaluation does not cover the complete trained attempt horizon")
        trace_keys = [(row["mode"], row["step"]) for row in traces if row["seed"] == seeds[0]]
        require(len(trace_keys) == len(set(trace_keys)) and set(trace_keys) ==
                set(itertools.product(("true", "wrong_friction"), range(horizon))),
                "Counterfactual trace coverage is incomplete")
        for row in traces:
            require(np.asarray(row["observation"]).shape == (len(grid), 46) and
                    np.asarray(row["supplied"]).shape == (len(grid), 46) and
                    np.asarray(row["action"]).shape == (len(grid), 5), "Counterfactual trace shape is invalid")
        return dict(directory=str(directory), episodes=len(rows), seed_blocks=len(seeds), physics_settings=len(grid),
                    input_modes=len(POLICY_MODES), trace_rows=len(traces), checkpoint_sha256=digest(policy_path()))

    def final_retries():
        directory = stage_dir("retry_evaluation", "retry_evaluation")
        manifest = read_json(directory / "manifest.json")
        report = read_json(directory / "summary.json")
        require(manifest.get("status") == "complete", "Retry evaluation is not complete")
        require(manifest["checkpoint_sha256"] == dict(policy=digest(policy_path()), estimator=digest(estimator_path())),
                "Retry policy/estimator checkpoint hashes do not match pinned models")
        check_observable_contract(manifest)
        require(report["attempt_budget"] == expected.retry_attempts, "Retry attempt budget differs from the contract")
        settings = manifest["settings"]
        require(settings["seed"] == expected.retry_seed_start and settings["sequences"] == expected.retry_seed_blocks,
                "Retry starting seeds or sequence count differs from the contract")
        conditions = settings["conditions"]
        require(len(conditions) == expected.retry_conditions, "Retry physical-condition coverage is incomplete")
        held_targets=read_json(root / "estimator_evaluation.json")["conditions"]
        require({physics_key(c["mass_kg"],c["friction"]) for c in conditions} ==
                {physics_key(*c["target"]) for c in held_targets},
                "Retry physical conditions do not match the estimator-held test conditions")
        rows = read_jsonl(directory / "attempts.jsonl")
        groups = {}
        for row in rows:
            group = (row["method"], row["condition_index"], row["sequence_index"])
            groups.setdefault(group, []).append(row)
            require(row["seed"] == expected.retry_seed_start +
                    row["sequence_index"] * expected.retry_attempts + row["attempt"] - 1,
                    "Recorded retry starting seeds do not follow matched sequence ordering")
            condition = conditions[row["condition_index"]]
            require(physics_key(row["actual_mass_kg"], row["actual_friction"]) ==
                    physics_key(condition["mass_kg"], condition["friction"]), "Retry physics changed within a condition")
        desired = set(itertools.product(RETRY_MODES, range(expected.retry_conditions), range(expected.retry_seed_blocks)))
        require(set(groups) == desired, "Fixed/adaptive/true methods do not cover identical retry sequences")
        for group, attempts in groups.items():
            attempts.sort(key=lambda row: row["attempt"])
            require([row["attempt"] for row in attempts] == list(range(1, len(attempts) + 1)) and
                    len(attempts) <= expected.retry_attempts, "Retry attempt coverage is discontinuous or exceeds its budget")
            require(not any(row["success"] for row in attempts[:-1]), "Retries continue after success")
            require(attempts[-1]["success"] or len(attempts) == expected.retry_attempts,
                    "A failed sequence stopped before exhausting its budget")
        count = expected.retry_conditions * expected.retry_seed_blocks
        for method in RETRY_MODES:
            summary = report["methods"][method]
            succeeded = sum(any(row["success"] for row in attempts)
                            for group, attempts in groups.items() if group[0] == method)
            require(summary["sequences"] == count and summary["successes_within_budget"] == succeeded,
                    "Retry summary disagrees with raw sequence outcomes")
        require(manifest["completed_attempts"] == len(rows), "Retry completed-attempt count is inconsistent")
        raw = validate_raw_episodes(directory, rows, retries=True)
        return dict(directory=str(directory), sequences_per_method=count,
                    seed_blocks=expected.retry_seed_blocks, attempts=len(rows), **raw)

    def assessment(directory, *, with_transfer=False):
        report = read_json(directory / "assessment.json")
        required_sections = ("policy", "estimator", "closed_loop") + (("estimator_transfer",) if with_transfer else ())
        for name in required_sections:
            require(report[name].get("status") in CAPABILITY_STATUSES,
                    f"Assessment section {name} is missing a resolved scientific conclusion")
        require(report["training"].get("requested_50000_iterations_completed") is True,
                "Assessment does not acknowledge completion of the requested training budget")
        require(report["training"].get("status") == "complete" and
                report["training"].get("planned_iterations", 0) >= expected.minimum_iterations,
                "Assessment training fields do not describe the completed requested run")
        sources = report.get("sources", {})
        required_sources = {
            "policy_evaluation": stage_dir("final_evaluation", "final_evaluation") / "summary.json",
            "estimator_evaluation": root / "estimator_evaluation.json",
            "retry_evaluation": stage_dir("retry_evaluation", "retry_evaluation") / "summary.json",
            "training_manifest": training_dir() / "manifest.json",
        }
        if with_transfer:
            required_sources["transfer_evaluation"] = Path(transfer).resolve() / "summary.json"
        for name, target in required_sources.items():
            require(name in sources and digest(sources[name]) == digest(target),
                    f"Assessment source {name} does not match the audited final artifacts")
        require((directory / "assessment.md").is_file(), "Human-readable assessment is missing")
        return dict(directory=str(directory), conclusions={key: report[key]["status"] for key in required_sections},
                    positive_findings_required=False)

    def final_transfer():
        directory = Path(transfer).resolve()
        manifest, report = read_json(directory / "manifest.json"), read_json(directory / "summary.json")
        require(manifest.get("status") == "complete", "Controlled transfer collection is not complete")
        hashes = dict(policy=digest(policy_path()), estimator=digest(estimator_path()))
        require(manifest["checkpoint_sha256"] == hashes and
                report["policy_checkpoint_sha256"] == hashes["policy"] and
                report["checkpoint_sha256"] == hashes["estimator"], "Transfer model hashes do not match pinned final models")
        check_observable_contract(manifest)
        settings = manifest["settings"]
        require(len(settings["masses"]) == expected.transfer_masses and
                len(settings["frictions"]) == expected.transfer_frictions,
                "Controlled transfer does not cover the declared full crossed grid")
        require(np.allclose(sorted(settings["masses"]),sorted(expected.transfer_mass_values)) and
                np.allclose(sorted(settings["frictions"]),sorted(expected.transfer_friction_values)),
                "Controlled transfer uses different physical values from the predeclared grid")
        require(settings["supplied_mass"] == expected.transfer_supplied_mass and
                settings["supplied_friction"] == expected.transfer_supplied_friction,
                "Controlled transfer uses different fixed guesses from the predeclared contract")
        require(settings["attempts"] == expected.transfer_attempts and
                settings["seed"] == expected.transfer_seed_start, "Transfer attempt budget/starting seeds differ from the contract")
        grid = {physics_key(mass, mu) for mass, mu in itertools.product(settings["masses"], settings["frictions"])}
        require(len(grid) == expected.transfer_masses * expected.transfer_frictions, "Transfer grid contains duplicate physics")
        rows = read_jsonl(directory / "attempts.jsonl")
        keys = [(physics_key(row["actual_mass_kg"], row["actual_friction"]), row["attempt"]) for row in rows]
        require(len(keys) == len(set(keys)) and set(keys) ==
                set(itertools.product(grid, range(1, expected.transfer_attempts + 1))),
                "Controlled transfer grid/evidence-prefix coverage is incomplete")
        for row in rows:
            require(row["seed"] == expected.transfer_seed_start + row["attempt"] - 1,
                    "Transfer starting seeds differ across matched physical conditions")
            require(row["supplied_mass_kg"] == settings["supplied_mass"] and
                    row["supplied_friction"] == settings["supplied_friction"],
                    "Collecting policy estimates changed during fixed-policy transfer")
        require(report["episodes"] == len(rows) == manifest["completed_episodes"], "Transfer episode counts disagree")
        require(report["physical_conditions"] == len(grid) and str(expected.transfer_attempts) in report["attempts"],
                "Transfer summary omits physics conditions or final evidence count")
        raw = validate_raw_episodes(directory, rows)
        return dict(directory=str(directory), physics_settings=len(grid), **raw)

    check("pipeline_finished", pipeline_status)
    check("training_budget", budget)
    check("actual_training_execution", execution)
    check("finite_46_input_final_policy", weights)
    check("guarded_optimizer_execution", guarded_updates)
    check("frozen_source_integrity", frozen_sources)
    check("held_condition_estimator_evaluation", historical_estimator)
    check("matched_final_policy_evaluation", final_policy)
    check("actual_retry_evaluation", final_retries)
    check("final_objective_assessment", lambda: assessment(stage_dir("assessment", "assessment")))
    if transfer is not None:
        check("controlled_final_policy_estimator_transfer", final_transfer)
    if combined_assessment is not None:
        check("combined_objective_assessment", lambda: assessment(Path(combined_assessment).resolve(), with_transfer=transfer is not None))
    ready = all(item["status"] == "proved" for item in requirements)
    return dict(overall_ready=ready,
                status="proved" if ready else "contradicted" if any(item["status"] == "contradicted" for item in requirements) else "incomplete",
                pipeline_root=str(root), expected_contract=asdict(expected), requirements=requirements,
                interpretation="Completion means the requested training and objective evaluations are evidenced; negative scientific findings are valid outcomes.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline_root", type=Path)
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--combined-assessment", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy-seed-count", type=int, default=10)
    parser.add_argument("--retry-sequences", type=int, default=10)
    parser.add_argument("--retry-conditions", type=int, default=9)
    parser.add_argument("--transfer-seed-start", type=int, default=64000)
    parser.add_argument("--require-guarded-updates",action="store_true")
    args = parser.parse_args(argv)
    for name in ("policy_seed_count", "retry_sequences", "retry_conditions"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    result = audit(args.pipeline_root, transfer=args.transfer,
                   combined_assessment=args.combined_assessment,
                   expectations=AuditExpectations(policy_seed_count=args.policy_seed_count,
                       retry_seed_blocks=args.retry_sequences, retry_conditions=args.retry_conditions,
                       transfer_seed_start=args.transfer_seed_start,
                       require_guarded_updates=args.require_guarded_updates))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return result


if __name__ == "__main__":
    main()
