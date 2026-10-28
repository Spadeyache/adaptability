"""Audit the explicitly revised scope after a user stops training.

This is a read-only evidence audit, never a launcher or a substitute for the
strict 50,000-cycle completion audit. Negative research findings are valid.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

from .completion import (CAPABILITY_STATUSES, Incomplete, POLICY_MODES, RETRY_MODES,
                         check_observable_contract, digest, physics_key, read_json,
                         read_jsonl, require, validate_raw_episodes)


TASKS = ("vertical", "native_carry")


def validate_inventory(source, inventory):
    source, inventory = Path(source).resolve(), Path(inventory).resolve()
    hashes = read_json(inventory)
    require(bool(hashes), "Source inventory is empty")
    for name, expected in hashes.items():
        path = (source / name).resolve()
        require(path.is_relative_to(source), "Source inventory escapes its snapshot")
        require(digest(path) == expected, f"Frozen source changed: {name}")
    return dict(source=str(source), inventory=str(inventory),
                inventory_sha256=digest(inventory), files=len(hashes))


def validate_stopped_processes(handles, *, proc_root="/proc"):
    """Check original boot/start-tick identities; never signal a process.

    A different boot ID could be a sandbox namespace, so it is inconclusive
    rather than evidence that a host process exited.
    """
    require({h.get("role") for h in handles} == {"supervisor", "worker"} and len(handles) == 2,
            "Both original supervisor and worker identities are required")
    proc = Path(proc_root)
    boot = (proc / "sys/kernel/random/boot_id").read_text().strip()
    namespace = os.readlink(proc / "self/ns/pid")
    evidence = []
    for handle in handles:
        require(handle.get("boot_id") and int(handle["start_ticks"]) > 0 and int(handle["pid"]) > 0,
                "Incomplete original process identity")
        if boot != handle["boot_id"]:
            raise Incomplete("Current /proc boot identity differs from the recorded host; run audit in the host namespace")
        if not handle.get("pid_namespace") or namespace != handle["pid_namespace"]:
            raise Incomplete("Current /proc PID namespace differs from the recorded host; run audit in the host namespace")
        path = proc / str(handle["pid"]) / "stat"
        try:
            # comm may contain spaces/parentheses. Fields after its closing ')'
            # start with state (3); starttime is field22, offset19 here.
            fields = path.read_text().rsplit(")", 1)[1].split()
            current_ticks, state = int(fields[19]), fields[0]
            original_alive = current_ticks == int(handle["start_ticks"]) and state not in ("Z", "X")
            require(not original_alive, f"Original {handle['role']} is still alive")
            result = "exited_zombie" if current_ticks == int(handle["start_ticks"]) else "pid_reused"
        except FileNotFoundError:
            result, current_ticks, state = "absent", None, None
        evidence.append(dict(**handle, observation=result, observed_start_ticks=current_ticks, observed_state=state))
    return dict(observed_at=datetime.now(timezone.utc).isoformat(), boot_id=boot, pid_namespace=namespace, handles=evidence)


def validate_stopped_execution(directory, manifest, stop, *, require_guarded=False):
    """Verify every completed rollout, without promoting the planned budget."""
    from robot_adaptation.policy.checkpoint_metadata import stopped_training_progress
    directory = Path(directory)
    progress = stopped_training_progress(stop, manifest)
    settings = manifest["settings"]
    batch = settings["num_envs"] * settings["num_steps"]
    require(settings["num_envs"] == 60 and settings["num_steps"] == 50,
            "Stopped training differs from the declared 60-by-50 rollout contract")
    require(manifest["planned_iterations"] == stop["original_planned_iterations"] == 50000 and
            settings["total_timesteps"] == 50000 * batch,
            "Original 50,000-cycle budget was not preserved")
    require(manifest.get("status") == "stopped_by_user" and
            manifest.get("completed_rollouts") == stop["completed_rollouts"],
            "Training manifest does not acknowledge the same explicit user stop")
    enabled = settings.get("guard_policy_updates") is True and manifest.get("trust_region", {}).get("enabled") is True
    require(enabled or not require_guarded, "Guarded updates are required")
    require(enabled, "This stopped-run audit requires sequential guard evidence")
    require(digest(directory / "trust_region.py") == manifest["trust_region_sha256"],
            "Executed optimizer guard source changed")
    count, accepted, maximum = 0, 0, 0.
    limit = settings["max_exact_kl"]
    require(math.isfinite(limit) and math.isclose(limit, .05, abs_tol=1e-12) and
            settings["learning_rate"] == 1e-5,
            "Guarded correction changed its predeclared 0.05 KL bound or 1e-5 learning rate")
    with (directory / "trust_region.jsonl").open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            require(row["iteration"] == count and row["global_step"] == count * batch,
                    "Guard rows are missing, duplicated, out of order, or have incorrect global steps")
            require(row["rollout_observations"] == batch and row["effective_actions"] == 5 and
                    row["optimization_semantics"] == manifest["optimization_semantics"],
                    "Guard evidence does not cover the original full five-action rollout")
            for name in ("max_accepted_exact_kl", "rollout_exact_kl"):
                require(math.isfinite(row[name]) and 0 <= row[name] <= limit + 1e-8,
                        "An accepted rollout update exceeds the exact mean KL bound")
            require(isinstance(row["accepted_updates"], int) and row["accepted_updates"] >= 0,
                    "Invalid accepted update count")
            accepted += row["accepted_updates"]
            maximum = max(maximum, row["max_accepted_exact_kl"])
    require(count == stop["completed_rollouts"] == stop["last_guard_iteration"] and accepted > 0,
            "Executed rollout count does not match the explicit stop record")
    require(stop["collected_transitions"] == count * batch and
            stop["saved_checkpoint_transitions"] == stop["saved_checkpoint_completed_cycles"] * batch,
            "Executed or saved transition accounting is inconsistent")
    return dict(**progress, guarded_rollouts=count, accepted_updates=accepted,
                max_accepted_exact_kl=maximum, mean_kl_limit=limit,
                guard_log_sha256=digest(directory / "trust_region.jsonl"))


def validate_plan(plan, stop):
    require(plan["user_stop_sha256"] == digest(plan["user_stop_record"]), "Plan stop-record hash differs")
    require(plan["executed_rollouts"] == stop["completed_rollouts"] and
            plan["saved_checkpoint_cycles"] == stop["saved_checkpoint_completed_cycles"] and
            plan["saved_checkpoint_transitions"] == stop["saved_checkpoint_transitions"] and
            plan["original_budget_completed"] is False, "Plan changes stopped-training accounting")
    require(Path(plan["policy_checkpoint"]).resolve() == Path(stop["latest_saved_checkpoint"]).resolve() and
            plan["policy_sha256"] == stop["latest_saved_checkpoint_sha256"],
            "Plan selects a different policy from the final saved stopped checkpoint")
    policy, retries, transfer = (plan[k] for k in ("policy", "retries", "transfer"))
    require(policy["masses_kg"] == [.04, .064, .1, .25, .5] and policy["frictions"] == [.2, .5, 1.] and
            policy["modes"] == list(POLICY_MODES) and policy["seeds"] == list(range(91000, 91010)) and
            policy["horizon"] == 50, "Final six-mode policy protocol differs from the declared reserved test")
    require(retries["methods"] == list(RETRY_MODES) and retries["sequences_per_condition"] == 10 and
            retries["attempt_budget"] == 3 and retries["seed_start"] == 54000 and retries["horizon"] == 50,
            "Final retry protocol differs from the original contract")
    require(transfer["masses_kg"] == [.08, .16, .32, .48] and transfer["frictions"] == [.25, .45, .65, .85] and
            transfer["attempts"] == 5 and transfer["seeds"] == list(range(64000, 64005)) and
            transfer["supplied_mass_kg"] == .1 and transfer["supplied_friction"] == .4 and transfer["horizon"] == 50,
            "Final controlled transfer differs from the original contract")
    require(retries["tasks"] == list(TASKS) and transfer["tasks"] == list(TASKS) and plan.get("task_amendment"),
            "Both legacy vertical and native carrying protocols must remain declared")
    require(digest(retries["conditions_file"]) == retries["conditions_sha256"], "Declared retry conditions changed")
    return dict(plan_scope=plan["scope"], executed_rollouts=stop["completed_rollouts"],
                evaluated_saved_cycles=stop["saved_checkpoint_completed_cycles"], original_budget_completed=False)


def validate_policy(directory, plan, policy_sha, training):
    """Validate the unchanged GPU evaluator's full paired outcome/trace files."""
    from experiments.physics_conditioning.evaluate import paired_summary
    directory = Path(directory)
    report, rows = read_json(directory / "summary.json"), read_json(directory / "episodes.json")
    require(report["checkpoint_sha256"] == policy_sha, "Policy evaluation uses another checkpoint")
    for key in ("seeds", "masses_kg", "frictions", "horizon"):
        require(report[key] == plan[key], f"Policy evaluation differs from plan: {key}")
    grid = {physics_key(m, mu) for m, mu in itertools.product(plan["masses_kg"], plan["frictions"])}
    require(grid == {physics_key(m, mu) for m, mu in itertools.product(
        training["settings"]["masses_kg"], training["settings"]["cube_friction_coefficients"])},
        "Final policy grid differs from training")
    keys = [(r["mode"], r["seed"], physics_key(r["mass_kg"], r["friction"])) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == set(itertools.product(POLICY_MODES, plan["seeds"], grid)),
            "Final policy six-mode paired coverage is incomplete or duplicated")
    fingerprints = report["reset_fingerprints"]
    require(set(fingerprints) == set(map(str, plan["seeds"])) and all(fingerprints.values()),
            "Policy reset fingerprints do not cover all reserved seeds")
    for mode in POLICY_MODES:
        selected = [r for r in rows if r["mode"] == mode]
        for metric in ("final_success", "sustained_success", "lifted_ever", "grasp_lost_after_lift", "reward_sum"):
            require(np.isclose(np.mean([r[metric] for r in selected]), report["metrics"][mode][metric]),
                    f"Policy summary disagrees with raw outcomes: {mode}/{metric}")
    for metric, field in (("final_success", "paired_comparisons"), ("sustained_success", "paired_sustained_comparisons")):
        calculated = paired_summary(rows, metric=metric)
        require(calculated == report[field], "Paired seed-block policy summary differs from its raw outcomes")
    actual = {(physics_key(r["mass_kg"], r["static_friction"]), round(r["dynamic_friction"], 6))
              for r in report["physical_readbacks"]}
    require(actual == {(pair, pair[1]) for pair in grid}, "Policy actual mass/material readbacks differ from its labels")
    fixed = report["fixed_material_readbacks"]
    require(all(np.isclose(fixed[f"{finger}_finger_{kind}_friction"], .2)
                for finger in ("left", "right") for kind in ("static", "dynamic")) and
            all(np.isclose(fixed[f"table_{kind}_friction"], .3) for kind in ("static", "dynamic")),
            "Policy fixed finger/table material readbacks differ from the trained task")
    traces = read_jsonl(directory / "traces.jsonl")
    expected_traces = set(itertools.product(("true", "wrong_friction"), range(plan["horizon"])))
    require(len(traces) == len(expected_traces) and {(r["mode"], r["step"]) for r in traces} == expected_traces,
            "Policy diagnostic trace coverage is incomplete or duplicated")
    for row in traces:
        require(row["seed"] == plan["seeds"][0], "Policy trace uses a different seed")
        observation, supplied, action = map(np.asarray, (row["observation"], row["supplied"], row["action"]))
        require(observation.shape == supplied.shape == (len(grid), 46) and action.shape == (len(grid), 5) and
                all(np.isfinite(x).all() for x in (observation, supplied, action)), "Policy trace is malformed or nonfinite")
        columns = list(range(45))
        require(np.array_equal(observation[:, columns], supplied[:, columns]), "Friction intervention changed other inputs")
        target = observation[:, 45] if row["mode"] == "true" else 1.2 - observation[:, 45]
        require(np.allclose(supplied[:, 45], target), "Recorded friction intervention is incorrect")
    return dict(directory=str(directory), episodes=len(rows), seed_blocks=len(plan["seeds"]),
                checkpoint_sha256=policy_sha, raw_artifacts={name: digest(directory / name)
                for name in ("summary.json", "episodes.json", "traces.jsonl")},
                reset_evidence_limit="Per-seed batch fingerprints are recorded; all-mode equality was checked by the frozen evaluator during execution.")


def validate_collection_identity(directory, task, hashes, stop, expected_source=None):
    directory = Path(directory)
    manifest, report = read_json(directory / "manifest.json"), read_json(directory / "summary.json")
    if manifest.get("status") != "complete":
        raise Incomplete(f"Collection has not completed: {manifest.get('status')!r}")
    require(manifest["checkpoint_sha256"] == hashes, "Collection policy/estimator hashes differ from pinned models")
    require(manifest.get("task") == report.get("task") == task, "Collection task differs from its declared protocol")
    check_observable_contract(manifest)
    for document in (manifest, report):
        progress = document["policy_checkpoint_progress"]
        require(progress["completed_learning_cycles"] == stop["saved_checkpoint_completed_cycles"] and
                progress["checkpoint_transitions"] == stop["saved_checkpoint_transitions"] and
                progress["full_planned_run_completed"] is False and
                document["policy_training_iterations"] == stop["saved_checkpoint_completed_cycles"],
                "Collection confuses the saved checkpoint with executed or planned training")
        require(document["policy_training_steps"] == stop["saved_checkpoint_transitions"] and
                document["original_planned_policy_training_iterations"] == 50000 and
                document["original_planned_policy_training_steps"] == 150000000 and
                progress["training_manifest_status"] == "stopped_by_user",
                "Collection status or planned/saved transition metadata is inconsistent")
    required_sources = {"experiments/interaction_adaptation/evaluate.py",
                        "experiments/interaction_adaptation/transfer.py",
                        "robot_adaptation/policy/checkpoint_metadata.py",
                        "robot_adaptation/simulation/collection_lift.py",
                        "robot_adaptation/collection/collect.py",
                        "robot_adaptation/estimators/interaction/predict.py"}
    if task == "native_carry":
        required_sources.add("experiments/physics_conditioning/checkpoint_probe.py")
    require(required_sources <= set(manifest["source_sha256"]), "Collection source inventory omits an executed task/estimator dependency")
    if expected_source is not None:
        require(all(expected_source.get(name) == value for name, value in manifest["source_sha256"].items()),
                "Collection source differs from the post-stop frozen source inventory")
    source = validate_inventory_map(directory / "source", manifest["source_sha256"])
    return manifest, report, source


def validate_inventory_map(source, hashes):
    require(bool(hashes), "Collection source inventory is empty")
    source = Path(source).resolve()
    for name, expected in hashes.items():
        path = (source / name).resolve()
        require(path.is_relative_to(source) and digest(path) == expected, f"Collection source changed: {name}")
    return dict(files=len(hashes), source=str(source), sha256=hashes)


def collection_raw(directory, rows, *, retries=False, task=None):
    from robot_adaptation.collection.schema import validate_episode
    directory = Path(directory)
    evidence = validate_raw_episodes(directory, rows, retries=retries)
    evidence["artifact_sha256"] = {name: digest(directory / name)
                                  for name in ("manifest.json", "summary.json", "attempts.jsonl")}
    evidence["raw_sha256"] = {r["episode_id"]: digest(directory / "episodes" / (r["episode_id"] + ".npz")) for r in rows}
    for row in rows:
        with np.load(directory / "episodes" / (row["episode_id"] + ".npz"), allow_pickle=False) as data:
            validate_episode(data)
            require(len(data["input__action_applied"]) == 50, "Raw attempt does not cover the declared 50-step horizon")
            reset = json.loads(str(data["label__reset_state_json"]))
            reset_hash = hashlib.sha256(json.dumps(reset, sort_keys=True, allow_nan=False).encode()).hexdigest()
            require(str(data["label__reset_state_hash"]) == reset_hash, "Raw reset fingerprint does not hash its saved physical state")
            require(np.isclose(float(data["input__reported_mass_kg"]), row["supplied_mass_kg"]),
                    "Raw supplied mass differs from its recorded intervention")
            require(np.allclose(data["label__finger_static_friction"], .2) and
                    np.allclose(data["label__finger_dynamic_friction"], .2) and
                    np.isclose(float(data["label__table_static_friction"]), .3) and
                    np.isclose(float(data["label__table_dynamic_friction"]), .3) and
                    np.allclose(data["label__size_xyz_m"], [.04] * 3), "Raw nuisance physics differ from the declared nominal task")
            goal = np.asarray(data["input__goal_position_world_m"])
            initial = data["label__initial_object_pose_world_m_wxyz"]
            if task == "vertical":
                require(np.allclose(goal[:2], initial[:2]) and np.isclose(goal[2], .22),
                        "Legacy vertical task geometry differs from its declaration")
            elif task == "native_carry":
                require(.12 - 1e-6 <= goal[2] <= .32 + 1e-6,
                        "Native carry goal height falls outside the saved floating range")
                require(not np.allclose(goal[:2], initial[:2], atol=1e-6),
                        "Native carrying data retain a vertical goal instead of randomized goal XY")
    return evidence


def validate_retries(directory, task, protocol, hashes, stop, held_targets, expected_source=None):
    from .evaluate import summarize
    directory = Path(directory)
    manifest, report, source = validate_collection_identity(directory, task, hashes, stop, expected_source)
    require(report["checkpoint_sha256"] == hashes, "Retry summary model hashes differ")
    settings = manifest["settings"]
    conditions = settings["conditions"]
    require(len(conditions) == 9 and {physics_key(c["mass_kg"], c["friction"]) for c in conditions} ==
            {physics_key(*c["target"]) for c in held_targets}, "Retries do not cover all nine estimator-held physical conditions")
    require(settings["seed"] == protocol["seed_start"] and settings["sequences"] == protocol["sequences_per_condition"] and
            settings["attempts"] == protocol["attempt_budget"] and settings["steps"] == protocol["horizon"],
            "Retry seeds, horizon, or budget differ from plan")
    rows, groups = read_jsonl(directory / "attempts.jsonl"), {}
    for row in rows:
        group = (row["method"], row["condition_index"], row["sequence_index"])
        groups.setdefault(group, []).append(row)
        require(row["seed"] == protocol["seed_start"] + row["sequence_index"] * protocol["attempt_budget"] + row["attempt"] - 1,
                "Retry seeds are not matched across methods and conditions")
        condition = conditions[row["condition_index"]]
        require(physics_key(row["actual_mass_kg"], row["actual_friction"]) == physics_key(condition["mass_kg"], condition["friction"]),
                "Retry physics changed within its declared condition")
    require(set(groups) == set(itertools.product(RETRY_MODES, range(9), range(protocol["sequences_per_condition"]))),
            "Retry methods do not cover the identical full condition/sequence grid")
    for attempts in groups.values():
        attempts.sort(key=lambda r: r["attempt"])
        require([r["attempt"] for r in attempts] == list(range(1, len(attempts) + 1)) and len(attempts) <= protocol["attempt_budget"] and
                not any(r["success"] for r in attempts[:-1]) and (attempts[-1]["success"] or len(attempts) == protocol["attempt_budget"]),
                "Retry sequence stopped early, exceeded budget, duplicated attempts, or continued after success")
    require(manifest["completed_attempts"] == len(rows), "Retry manifest attempt count differs from records")
    calculated = summarize(rows, protocol["attempt_budget"])
    require(calculated["methods"] == report["methods"] and calculated["attempt_budget"] == report["attempt_budget"],
            "Retry summary differs from actual attempts or seed-block comparison")
    return dict(task=task, directory=str(directory), sequences_per_method=90, source=source,
                **collection_raw(directory, rows, retries=True, task=task))


def validate_transfer(directory, task, protocol, hashes, stop, expected_source=None):
    from .transfer import summarize_transfer
    directory = Path(directory)
    manifest, report, source = validate_collection_identity(directory, task, hashes, stop, expected_source)
    require(report["policy_checkpoint_sha256"] == hashes["policy"] and report["checkpoint_sha256"] == hashes["estimator"],
            "Transfer summary model hashes differ")
    settings = manifest["settings"]
    require(settings["masses"] == protocol["masses_kg"] and settings["frictions"] == protocol["frictions"] and
            settings["attempts"] == protocol["attempts"] and settings["seed"] == protocol["seeds"][0] and
            settings["steps"] == protocol["horizon"] and settings["supplied_mass"] == protocol["supplied_mass_kg"] and
            settings["supplied_friction"] == protocol["supplied_friction"], "Controlled transfer collection differs from plan")
    rows = read_jsonl(directory / "attempts.jsonl")
    grid = {physics_key(m, mu) for m, mu in itertools.product(protocol["masses_kg"], protocol["frictions"])}
    keys = [(physics_key(r["actual_mass_kg"], r["actual_friction"]), r["attempt"]) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == set(itertools.product(grid, range(1, protocol["attempts"] + 1))),
            "Controlled transfer lacks the complete crossed physics/evidence-prefix grid")
    for row in rows:
        require(row["seed"] == protocol["seeds"][row["attempt"] - 1] and row["supplied_mass_kg"] == protocol["supplied_mass_kg"] and
                row["supplied_friction"] == protocol["supplied_friction"], "Controlled transfer seeds or fixed guesses changed")
    require(len(rows) == manifest["completed_episodes"] == report["episodes"] == 80,
            "Controlled transfer must contain all 80 independent episodes")
    # JSON converts numeric evidence-prefix keys to strings, including nested
    # per-condition prefixes; compare both summaries in their saved form.
    calculated = json.loads(json.dumps(summarize_transfer(rows, report["prior_median"]), allow_nan=False))
    require(calculated["attempts"] == report["attempts"] and calculated["conditions"] == report["conditions"],
            "Transfer error/uncertainty summary differs from recorded predictions and targets")
    return dict(task=task, directory=str(directory), source=source, **collection_raw(directory, rows, task=task))


def audit(pipeline_root, *, evaluation_root, require_guarded_updates=False, proc_root="/proc"):
    root, post = Path(pipeline_root).resolve(), Path(evaluation_root).resolve()
    requirements, cache = [], {}

    def check(identifier, operation):
        try:
            requirements.append(dict(id=identifier, status="proved", evidence=operation()))
        except (FileNotFoundError, Incomplete) as error:
            requirements.append(dict(id=identifier, status="incomplete", reason=str(error)))
        except Exception as error:
            requirements.append(dict(id=identifier, status="contradicted", reason=str(error), error_type=type(error).__name__))

    def context():
        if not cache:
            stop, plan = read_json(root / "early_stop.json"), read_json(post / "plan.json")
            state = read_json(root / "status.json")
            train = Path(state["training_run"])
            cache.update(stop=stop, plan=plan, train=train, manifest=read_json(train / "manifest.json"),
                         hashes=dict(policy=plan["policy_sha256"], estimator=plan["estimator_sha256"]))
        return cache

    def stop_scope():
        c = context()
        result = validate_plan(c["plan"], c["stop"])
        require(digest(root / "early_stop.json") == digest(post / "early_stop.json"), "Archived user-stop record differs")
        require(Path(c["plan"]["user_stop_record"]).resolve() == root / "early_stop.json", "Plan references another run's stop record")
        return dict(**result, early_stop_sha256=digest(root / "early_stop.json"), plan_sha256=digest(post / "plan.json"))

    def stopped():
        c = context()
        handles = c["plan"].get("stopped_processes")
        if not handles:
            raise Incomplete("Plan does not yet pin the stopped host process identities")
        return validate_stopped_processes(handles, proc_root=proc_root)

    def archive():
        c = context()
        entries = c["plan"].get("archived_stop_metadata")
        if not entries:
            raise Incomplete("Plan does not yet pin archived original termination metadata")
        for row in entries:
            require(digest(row["path"]) == row["sha256"], "Archived original stop metadata changed")
        require(any(Path(row["path"]).resolve() == Path(c["stop"]["original_supervisor_result"]).resolve() for row in entries),
                "Original supervisor termination result is not archived and pinned")
        return dict(artifacts=entries)

    def weights():
        import torch
        c = context()
        saved = torch.load(c["plan"]["policy_checkpoint"], map_location="cpu", weights_only=True)
        require(isinstance(saved, dict) and bool(saved) and all(isinstance(t, torch.Tensor) and torch.isfinite(t).all().item()
                for t in saved.values()), "Saved policy is malformed or nonfinite")
        for name in ("actor_mean.0.weight", "strength_mean.0.weight", "critic.0.weight"):
            require(saved[name].ndim == 2 and saved[name].shape[1] == 46, f"Policy lacks 46 inputs: {name}")
        require(digest(c["plan"]["policy_checkpoint"]) == c["hashes"]["policy"] and
                digest(c["plan"]["estimator_checkpoint"]) == c["hashes"]["estimator"], "Pinned model hash changed")
        return dict(path=c["plan"]["policy_checkpoint"], sha256=c["hashes"]["policy"], observations=46,
                    evaluated_completed_cycles=c["stop"]["saved_checkpoint_completed_cycles"])

    def sources(original):
        c = context()
        if original:
            return validate_inventory(c["plan"]["original_frozen_source"], c["plan"]["original_source_inventory"])
        return validate_inventory(c["plan"].get("evaluation_source", post / "source"),
                                  c["plan"].get("evaluation_source_inventory", post / "source_sha256.json"))

    def historical():
        import torch
        c = context()
        report = read_json(post / "estimator_evaluation.json")
        require(report["checkpoint_sha256"] == c["hashes"]["estimator"] and report["split"] == "test" and
                report["held_conditions"] == len(report["conditions"]) == 9,
                "Historical estimator report identity or held-condition coverage differs")
        require(digest(post / "estimator_evaluation.json") == digest(root / "estimator_evaluation.json"),
                "Historical estimator evidence differs from its pinned original")
        saved = torch.load(c["plan"]["estimator_checkpoint"], map_location="cpu", weights_only=True)
        features = saved["feature_names"]
        require(features and all(name.startswith("input__") and not any(x in name for x in ("reported_mass", "friction", "object_weight"))
                for name in features), "Estimator features leak physical target/supplied parameters")
        require(all(torch.isfinite(t).all().item() for t in saved["state_dict"].values()), "Estimator weights are nonfinite")
        for k in (1, 2, 3, 5):
            value = np.asarray(report["attempts"][str(k)]["mae"])
            require(value.shape == (2,) and np.isfinite(value).all(), "Historical estimator error metric is invalid")
        return dict(report_sha256=digest(post / "estimator_evaluation.json"), checkpoint_sha256=c["hashes"]["estimator"], held_conditions=9)

    def assessment(task):
        c = context()
        directory = post / f"assessment_{task}"
        report = read_json(directory / "assessment.json")
        sections = ("policy", "estimator", "closed_loop", "estimator_transfer")
        for name in sections:
            require(report[name]["status"] in CAPABILITY_STATUSES, f"Scientific assessment is unresolved: {name}")
        training = report["training"]
        require(training["status"] == "stopped_by_user" and training["requested_50000_iterations_completed"] is False and
                training["executed_rollout_cycles"] == c["stop"]["completed_rollouts"] and
                training["saved_checkpoint_completed_cycles"] == c["stop"]["saved_checkpoint_completed_cycles"] and
                training["latest_saved_checkpoint_sha256"] == c["hashes"]["policy"],
                "Assessment misrepresents early-stop status or the evaluated saved policy")
        required = dict(policy_evaluation=post / "policy/summary.json", estimator_evaluation=post / "estimator_evaluation.json",
                        retry_evaluation=post / f"retries_{task}/summary.json", transfer_evaluation=post / f"transfer_{task}/summary.json",
                        training_manifest=c["train"] / "manifest.json", early_stop_record=root / "early_stop.json")
        for name, target in required.items():
            require(digest(report["sources"][name]) == digest(target) == report["source_sha256"][name],
                    f"Assessment source {name} is stale or different")
        frozen = read_json(c["plan"].get("evaluation_source_inventory", post / "source_sha256.json"))
        require(report["assessment_source_sha256"] == frozen["experiments/interaction_adaptation/assess.py"],
                "Assessment implementation differs from its frozen amended source")
        require(report["closed_loop"]["task"] == report["estimator_transfer"]["task"] == task,
                "Assessment confuses native carry and legacy vertical evidence")
        from .assess import assess
        calculated = assess(read_json(required["policy_evaluation"]), read_json(required["estimator_evaluation"]),
                            read_json(required["retry_evaluation"]), c["manifest"],
                            read_json(required["transfer_evaluation"]), c["stop"])
        for name in sections:
            require(all(report[name].get(key) == value for key, value in calculated[name].items()),
                    f"Assessment conclusion/evidence differs from its audited sources: {name}")
        require((directory / "assessment.md").is_file(), "Human-readable objective assessment is missing")
        return dict(directory=str(directory), conclusions={name: report[name]["status"] for name in sections},
                    artifact_sha256={name: digest(directory / name) for name in ("assessment.json", "assessment.md")},
                    positive_findings_required=False)

    check("explicit_user_stop_and_revised_scope", stop_scope)
    check("original_training_processes_stopped", stopped)
    check("archived_original_stop_metadata", archive)
    check("executed_guarded_rollouts_and_saved_accounting", lambda: validate_stopped_execution(
        context()["train"], context()["manifest"], context()["stop"], require_guarded=require_guarded_updates))
    check("finite_46_input_saved_policy", weights)
    check("original_frozen_source_integrity", lambda: sources(True))
    check("post_stop_source_integrity", lambda: sources(False))
    check("held_condition_estimator_evaluation", historical)
    check("reserved_final_policy_evaluation", lambda: validate_policy(post / "policy", context()["plan"]["policy"],
        context()["hashes"]["policy"], context()["manifest"]))
    for task in TASKS:
        check(f"actual_retries_{task}", lambda task=task: validate_retries(post / f"retries_{task}", task,
            context()["plan"]["retries"], context()["hashes"], context()["stop"], read_json(post / "estimator_evaluation.json")["conditions"],
            read_json(context()["plan"].get("evaluation_source_inventory", post / "source_sha256.json"))))
        check(f"controlled_transfer_{task}", lambda task=task: validate_transfer(post / f"transfer_{task}", task,
            context()["plan"]["transfer"], context()["hashes"], context()["stop"],
            read_json(context()["plan"].get("evaluation_source_inventory", post / "source_sha256.json"))))
        check(f"objective_assessment_{task}", lambda task=task: assessment(task))
    ready = all(r["status"] == "proved" for r in requirements)
    return dict(overall_ready=ready, revised_scope_completed=ready, requested_50000_iterations_completed=False,
                status="proved" if ready else "contradicted" if any(r["status"] == "contradicted" for r in requirements) else "incomplete",
                pipeline_root=str(root), evaluation_root=str(post), requirements=requirements,
                interpretation="Completion covers the explicitly revised post-stop evaluation scope only. Training remains stopped; 50,000 cycles were not completed. Negative capability findings are valid.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline_root", type=Path)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-guarded-updates", action="store_true")
    args = parser.parse_args(argv)
    result = audit(args.pipeline_root, evaluation_root=args.evaluation_root, require_guarded_updates=args.require_guarded_updates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "overall_ready", "revised_scope_completed", "requested_50000_iterations_completed")}, indent=2))
    return result


if __name__ == "__main__":
    main()
