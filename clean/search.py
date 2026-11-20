"""Bounded, validation-only, one-change continuation of the locked search.

The module has pure proposal/decision helpers for synthetic tests. Actual
execution reads saved validation results and delegates training and final
evaluation to clean.run. It never loads an episode itself. No search is allowed
once the final-evaluation ledger exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ITERATION_BUDGET = 20
METRICS = ("one_step_nmse", "r2", "rollout_error")
REQUIRED_CHECKS = ("F1", "F2", "F3", "F5", "F6", "F7", "F8", "F9")
HARD_CHECKS = tuple(check for check in REQUIRED_CHECKS if check != "F5")
REQUIRED_MODELS = ("plain", "history", "oracle", "fitted_z", "mean_z", "shuffled_z", "another_cube_z", "learned_z")

# The order and every fallback are explicit. Only one assignment is applied to
# the retained configuration; there is no hidden grid or random search.
OPTIONS = (
    ("z_dim", 4, "A larger latent may represent distinct object effects; F5 will test whether learned-z gains transfer to freshly fitted z."),
    ("z_dim", 8, "Additional latent capacity may recover object effects missed by the current embedding; F5 guards against identity memorization."),
    ("z_dim", 16, "The largest allowed latent may expose a capacity limitation; F5 must still pass on fresh-z validation."),
    ("z_l2", 0.0, "Removing latent shrinkage may let prediction gradients distinguish cubes more strongly."),
    ("fit_steps", 1000, "More frozen-weight latent updates may resolve incomplete adaptation convergence."),
    ("fit_lr", 0.005, "A smaller fitting learning rate may reduce overshoot and noisy object-latent estimates."),
    ("width", 64, "A smaller shared network may reduce memorization of shared trajectories and increase the useful contribution of z."),
    ("width", 32, "Further reducing shared-network width may test whether shared trajectory fitting suppresses object-specific latent use."),
    ("width", 256, "More shared-network capacity may reduce underfitting of contact dynamics and strengthen the Oracle reference."),
    ("depth", 3, "An additional hidden layer may capture nonlinear contact responses that a shallow shared model misses."),
    ("lr", 0.0003, "A smaller training learning rate may reduce optimization noise in the shared laws and object embeddings."),
    ("lr", 0.003, "A larger training learning rate may improve convergence within the unchanged update budget."),
    ("moving_weight", 10.0, "Giving moving transitions more loss weight may focus fitting on the transitions used by the locked metric."),
    ("prediction_horizon", 3, "A short recursive training loss may favor shared dynamics that remain useful beyond one step."),
    ("prediction_horizon", 5, "The longest allowed recursive training loss may reduce unstable shared dynamics while preserving the locked evaluation metric."),
    ("z_l2", 0.01, "Stronger latent regularization may limit identity-specific encoding and improve fresh-z fitting."),
    ("fit_lr", 0.1, "A larger fitting learning rate may overcome slow adaptation within the fixed fitting-step count."),
)
FALLBACK_OPTIONS = (
    ("fit_steps", 2000, "Additional frozen-weight fitting steps may distinguish incomplete convergence from absent object information."),
    ("depth", 1, "A shallower shared model may reduce trajectory-specific overfitting and make the latent contribution identifiable."),
    ("width", 16, "A compact shared model may test whether excess shared capacity obscures cube-specific constants."),
    ("lr", 0.0001, "A lower training learning rate may stabilize shared-law and embedding optimization."),
)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _signature(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _atomic_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if _read(path) != value:
            raise RuntimeError(f"Existing immutable search record differs: {path}")
        return
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists() and not replace:
        temporary.unlink()
        raise FileExistsError(path)
    temporary.replace(path)


def all_success(result):
    return all(result.get("success", {}).get(f"S{i}") is True for i in range(1, 5))


def finite_metrics(result):
    summaries = result.get("summary", {})
    if any(name not in summaries for name in REQUIRED_MODELS):
        return False
    for model in summaries.values():
        for metric in METRICS:
            value = model.get(metric, {})
            if not _finite(value.get("mean")) or not _finite(value.get("std")):
                return False
            values = value.get("values", [])
            if len(values) != 5 or not all(_finite(item) for item in values):
                return False
    return True


def eligibility(result, *, allow_failed_f5=False):
    checks = HARD_CHECKS if allow_failed_f5 else REQUIRED_CHECKS
    failed = [name for name in checks if result.get("failure_checks", {}).get(name) is not True]
    if not finite_metrics(result):
        failed.append("nonfinite_or_missing_five_seed_metrics")
    return not failed, failed


def nmse(result):
    return result["summary"]["fitted_z"]["one_step_nmse"]["mean"]


def select_reference(results, *, allow_failed_f5=False):
    """Honor required F4 feasibility, then minimize the locked fitted-z NMSE."""
    eligible = [result for result in results if eligibility(result)[0]]
    if eligible:
        feasible = [result for result in eligible if result["failure_checks"].get("F4") is True]
        selected = min(feasible or eligible, key=lambda result: (nmse(result), result["iteration"]))
        if feasible:
            return selected, "F4 is a fixed required constraint, not an alternate metric. A reference passing F4 and the other required checks exists, so select the lowest fitted-z validation NMSE within that feasible set; F4-failing candidates cannot displace it."
        return selected, "No current-data candidate passes F4 together with the other required checks. Select the lowest fitted-z validation NMSE among finite candidates passing F1, F2, F3, F5, F6, F7, F8, and F9 only as a failed reference for further corrections. F4 remains a fixed required constraint; the locked metric is unchanged."
    if allow_failed_f5:
        fallback = [result for result in results if eligibility(result, allow_failed_f5=True)[0]]
        if fallback:
            selected = min(fallback, key=lambda result: (nmse(result), result["iteration"]))
            return selected, "No F5-passing reference exists; this is the lowest-NMSE finite candidate passing the other hard checks, retained only as an explicitly failed F5 reference."
    raise RuntimeError("No finite candidate passes the required leakage, contamination, reporting, and memorization checks")


def choose_proposal(base, tested_configs, used_options=()):
    """Apply the next untried explicit option to exactly one configuration key."""
    tested = {_signature(config) for config in tested_configs}
    used = set(used_options)
    skipped = []
    for index, (key, value, mechanism) in enumerate((*OPTIONS, *FALLBACK_OPTIONS)):
        option_id = f"{key}={json.dumps(value)}"
        if option_id in used:
            continue
        config = dict(base["config"])
        if key not in config:
            raise ValueError(f"Retained configuration lacks required key {key}")
        if config[key] == value:
            skipped.append({"option": option_id, "reason": "already_equals_retained_value"})
            continue
        previous = config[key]
        config[key] = value
        if _signature(config) in tested:
            skipped.append({"option": option_id, "reason": "identical_configuration_already_tested"})
            continue
        differences = [name for name in config if config[name] != base["config"][name]]
        if differences != [key]:
            raise RuntimeError(f"A proposal changed more than one setting: {differences}")
        observations = []
        if base["failure_checks"].get("F4") is not True:
            observations.append("The retained reference fails validation F4.")
        gap = base.get("success", {}).get("plain_oracle_gap")
        if not _finite(gap) or gap <= 0:
            observations.append("Oracle has not established a positive Plain-to-Oracle error gap.")
        if base["failure_checks"].get("F5") is not True:
            observations.append("The fallback reference also fails F5.")
        if not observations:
            observations.append("The retained reference has not met all locked validation success criteria.")
        return {"option_id": option_id, "option_index": index,
            "config": config, "base_iteration": base["iteration"],
            "hypothesis": " ".join([*observations, mechanism]),
            "change": f"Change only {key} from {previous} to {value}; every other configuration setting is retained.",
            "changed_key": key, "previous_value": previous, "new_value": value,
            "skipped_options": skipped, "fallback_option": index >= len(OPTIONS)}
    raise RuntimeError("All preregistered single-change options and fallbacks are exhausted")


def decide_candidate(base, result):
    """Return the selected record and an honest kept/reverted explanation."""
    valid, failures = eligibility(result)
    if all_success(result):
        if not valid:
            raise RuntimeError("Saved S1-S4 success conflicts with invalid failure checks or metrics")
        return result, "success", "All locked validation success criteria passed; stop immediately and select this candidate regardless of another candidate's NMSE."
    base_valid = eligibility(base)[0]
    if not valid:
        return base, "reverted", "Rejected this candidate because " + ", ".join(failures) + "; the checks were not loosened."
    base_f4 = base["failure_checks"].get("F4") is True
    candidate_f4 = result["failure_checks"].get("F4") is True
    if base_valid and candidate_f4 and not base_f4:
        return result, "kept", "This candidate satisfies required F4 while the reference fails it. Restore the fixed feasibility constraint before minimizing the unchanged fitted-z NMSE; this is not a metric change. Other success criteria still require correction."
    if base_valid and base_f4 and not candidate_f4:
        return base, "reverted", "This candidate fails required F4 and cannot displace a reference that passes F4 and the other required checks, even with lower NMSE. Preserve the fixed feasibility constraint; the locked metric is unchanged."
    if not base_valid or nmse(result) < nmse(base):
        reason = "Fitted-z validation NMSE decreased relative to the retained eligible reference." if base_valid else "This is the first eligible reference after an explicitly failed F5 fallback."
        if result["failure_checks"].get("F4") is not True:
            reason += " F4 still fails; kept only as a failed reference, not a successful model."
        else:
            reason += " At least one locked success criterion still fails."
        return result, "kept", reason
    return base, "reverted", "Fitted-z validation NMSE did not decrease; retained the lower-error eligible reference."


def _log_once(root, marker, text):
    path = root / "LOOP_LOG.md"
    prior = path.read_text() if path.exists() else ""
    if marker not in prior:
        with path.open("a") as handle:
            handle.write(f"\n{marker}\n\n{text}\n")


def _plan_marker(stage, number, plan):
    """Different retained bases/configurations must produce separate log entries."""
    identity = {"audit": plan["audit_sha256"], "base": plan["base_iteration"],
        "option": plan["option_id"], "config": plan["config"]}
    fingerprint = hashlib.sha256(_signature(identity).encode()).hexdigest()[:12]
    return f"<!-- search-{stage}-{number:02d}-base-{plan['base_iteration']:02d}-{fingerprint} -->"


def _command(number, config_path, plan, device):
    arguments = [".venv/bin/python", "-m", "clean.run", "candidate", "--iteration", str(number),
        "--config", str(config_path), "--base-iteration", str(plan["base_iteration"]),
        "--hypothesis", plan["hypothesis"], "--change", plan["change"], "--device", device]
    return "PYTHONPATH=clean/vendor:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 " + shlex.join(arguments)


def _record_failure(root, error, *, iteration, start, end, audit_sha):
    failure = {"status": "error", "iteration": iteration, "start": start, "end": end,
        "audit_sha256": audit_sha, "error": repr(error), "traceback": traceback.format_exc(),
        "timestamp": time.time(), "final_ledger_exists": (root / "results/final_evaluations.jsonl").exists()}
    _atomic_json(root / "results/search/failures" / f"failure_{time.time_ns()}.json", failure)
    _atomic_json(root / "results/search_failure.json", failure, replace=True)


def run_search(*, device="cuda", start=4, end=20, run_final=False):
    if not 4 <= start <= end <= ITERATION_BUDGET:
        raise ValueError("Search must continue within iterations 4 through 20, inclusive")
    # Lazy imports keep proposal/decision tests independent of data or devices.
    from clean.protocol import PROTOCOL
    from clean.run import DEFAULT, candidate, final, require_audit
    if PROTOCOL["iteration_budget"] != ITERATION_BUDGET:
        raise RuntimeError("Search budget differs from the locked protocol")
    root = ROOT
    ledger = root / "results/final_evaluations.jsonl"
    audit_sha = None
    try:
        if ledger.exists():
            raise RuntimeError("Final evaluation already began; no further search is permitted")
        require_audit()
        audit_sha = _sha(root / "results/full_audit.json")
    except BaseException as error:
        _record_failure(root, error, iteration=None, start=start, end=end, audit_sha=audit_sha)
        raise
    search_root = root / "results/search"
    search_root.mkdir(parents=True, exist_ok=True)
    current_iteration = None
    try:
        paths = {int(path.parent.name.split("_")[-1]): path for path in (root / "runs").glob("iteration_*/result.json")}
        if any(number not in paths for number in range(1, start)):
            raise RuntimeError("Complete every earlier iteration before this search continuation")
        saved = {number: _read(path) for number, path in paths.items()}
        if any(number > ITERATION_BUDGET for number in saved):
            raise RuntimeError("Saved candidate index exceeds the locked global iteration budget")
        for result in saved.values():
            result["config"] = {**DEFAULT, **result["config"]}
        current = {number: result for number, result in saved.items() if result.get("audit_sha256") == audit_sha}
        successes = [result for result in current.values() if all_success(result)]
        if successes:
            selected = min(successes, key=lambda result: result["iteration"])
            if not eligibility(selected)[0]:
                raise RuntimeError("Saved validation success conflicts with its checks")
            outcome = "validation_success"
        else:
            prior = [result for number, result in current.items() if number < start]
            base, reference_reason = select_reference(prior, allow_failed_f5=True)
            reference = {"start": start, "audit_sha256": audit_sha, "iteration": base["iteration"],
                "result_sha256": _sha(paths[base["iteration"]]), "reason": reference_reason,
                "F4": base["failure_checks"].get("F4"), "F5": base["failure_checks"].get("F5"),
                "one_step_nmse": base["summary"]["fitted_z"]["one_step_nmse"]}
            _atomic_json(search_root / f"reference_start_{start:02d}.json", reference)
            _log_once(root, f"<!-- search-reference-start-{start:02d} -->", f"Search starts at iteration {start} from iteration {base['iteration']}. {reference_reason} F4={reference['F4']}; F5={reference['F5']}. This does not assert success.")
            used_options = set()
            for plan_path in (root / "runs").glob("iteration_*/search_plan.json"):
                plan = _read(plan_path)
                if plan.get("audit_sha256") == audit_sha and plan["iteration"] < start:
                    used_options.add(plan["option_id"])
            selected = base
            outcome = "requested_range_complete"
            for number in range(start, end + 1):
                current_iteration = number
                if ledger.exists():
                    raise RuntimeError("Final ledger appeared during search; stop tuning")
                if _sha(root / "results/full_audit.json") != audit_sha:
                    raise RuntimeError("Active dataset audit changed during a search")
                out = root / "runs" / f"iteration_{number:02d}"
                plan_path = out / "search_plan.json"
                result_path = out / "result.json"
                if plan_path.exists():
                    plan = _read(plan_path)
                    if plan.get("audit_sha256") != audit_sha or plan["base_iteration"] != base["iteration"]:
                        raise RuntimeError("Saved search plan has a different data version or retained reference")
                    if len([key for key in plan["config"] if plan["config"][key] != base["config"][key]]) != 1:
                        raise RuntimeError("Saved search plan violates the one-change rule")
                elif result_path.exists():
                    # A completed manually launched continuation is never run
                    # again; adopt its actual recorded one-change plan.
                    completed = _read(result_path)
                    completed["config"] = {**DEFAULT, **completed["config"]}
                    changed = [key for key in completed["config"] if completed["config"][key] != base["config"][key]]
                    if completed.get("audit_sha256") != audit_sha or completed.get("base_iteration") != base["iteration"] or len(changed) != 1:
                        raise RuntimeError("Completed continuation does not match the reconstructed search reference")
                    key = changed[0]
                    plan = {"iteration": number, "audit_sha256": audit_sha, "base_iteration": base["iteration"],
                        "config": completed["config"], "hypothesis": completed["hypothesis"], "change": completed["change"],
                        "option_id": f"{key}={json.dumps(completed['config'][key])}", "changed_key": key,
                        "previous_value": base["config"][key], "new_value": completed["config"][key], "adopted_completed_run": True}
                else:
                    prior_configs = [result["config"] for n, result in current.items() if n < number]
                    plan = choose_proposal(base, prior_configs, used_options)
                    plan.update(iteration=number, audit_sha256=audit_sha)
                config_path = root / "configs" / f"search_iteration_{number:02d}.json"
                _atomic_json(config_path, plan["config"])
                command = _command(number, config_path, plan, device)
                plan["command"] = command
                _atomic_json(plan_path, plan)
                used_options.add(plan["option_id"])
                _log_once(root, _plan_marker("command", number, plan), "Exact candidate command (from repository root; any earlier command for the same iteration with a different retained base is superseded):\n\n```bash\n" + command + "\n```")
                if result_path.exists():
                    result = _read(result_path)
                else:
                    candidate(number, plan["config"], plan["hypothesis"], plan["change"], device, base_iteration=plan["base_iteration"])
                    # The persisted artifact preserves undefined statistics as
                    # JSON null; never let in-memory NaNs break decision logs.
                    result = _read(result_path)
                result["config"] = {**DEFAULT, **result["config"]}
                if result.get("audit_sha256") != audit_sha or result["config"] != plan["config"]:
                    raise RuntimeError("Completed candidate differs from its planned data/configuration")
                retained, action, reason = decide_candidate(base, result)
                decision = {"iteration": number, "audit_sha256": audit_sha, "result_sha256": _sha(result_path),
                    "base_iteration": base["iteration"], "retained_iteration": retained["iteration"],
                    "decision": action, "reason": reason, "option_id": plan["option_id"],
                    "one_step_nmse": result["summary"]["fitted_z"]["one_step_nmse"],
                    "failure_checks": result["failure_checks"], "success": result["success"], "command": command}
                _atomic_json(out / "search_decision.json", decision)
                _log_once(root, _plan_marker("decision", number, plan), f"Iteration {number} decision: **{action}**. {reason} Retained reference: iteration {retained['iteration']}.")
                print(json.dumps({"search_iteration": number, "decision": action, "reason": reason,
                    "retained_iteration": retained["iteration"], "success": result["success"]}), flush=True)
                current[number] = result
                saved[number] = result
                base = selected = retained
                if action == "success":
                    outcome = "validation_success"
                    break
            if outcome != "validation_success" and end == ITERATION_BUDGET:
                missing = [number for number in range(1, ITERATION_BUDGET + 1) if not (root / "runs" / f"iteration_{number:02d}" / "result.json").exists()]
                if missing:
                    raise RuntimeError(f"Global iteration budget is not complete; missing iterations {missing}")
                selected, reason = select_reference(list(current.values()), allow_failed_f5=True)
                outcome = "budget_exhausted_without_validation_success"
                _log_once(root, "<!-- search-budget-exhausted -->", f"All {ITERATION_BUDGET} scientific candidate iterations are complete without validation success. Selected iteration {selected['iteration']} for the single final evaluation. {reason} No threshold was changed.")
        summary = {"status": outcome, "selected_iteration": selected["iteration"], "audit_sha256": audit_sha,
            "validation_success": all_success(selected), "final_requested": bool(run_final),
            "selection_metric": "fitted_z one_step_nmse mean",
            "reference_constraint": "F4 feasibility is required once any current-data candidate passes F4 and all other required checks; failed references are provisional only when none do.",
            "failure_checks": selected["failure_checks"]}
        _atomic_json(search_root / "outcome.json", summary, replace=True)
        can_finalize = outcome in ("validation_success", "budget_exhausted_without_validation_success")
        if run_final and can_finalize:
            if ledger.exists():
                raise RuntimeError("Final invocation is single-use; ledger already exists")
            final(selected["iteration"], device)
            from clean.report import generate_report
            report_path = generate_report()
            summary["report"] = str(report_path)
            summary["final_completed"] = True
            _atomic_json(search_root / "outcome.json", summary, replace=True)
        elif run_final:
            summary["final_not_run_reason"] = "Requested search range ended before success or the global 20-iteration budget."
            _atomic_json(search_root / "outcome.json", summary, replace=True)
        print(json.dumps(summary), flush=True)
        return summary
    except BaseException as error:
        _record_failure(root, error, iteration=current_iteration, start=start, end=end, audit_sha=audit_sha)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start", type=int, default=4)
    parser.add_argument("--end", type=int, default=20)
    parser.add_argument("--final", action="store_true", dest="run_final")
    arguments = parser.parse_args()
    run_search(device=arguments.device, start=arguments.start, end=arguments.end, run_final=arguments.run_final)


if __name__ == "__main__":
    main()
