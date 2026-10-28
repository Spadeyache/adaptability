"""Falsify estimator transfer on a fresh crossed physics grid and frozen policy.

Independent attempts use fixed supplied policy estimates. The estimator pools
their completed sensor traces, but never changes the collecting policy. This
separates parameter inference transfer from adaptive retry success.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from robot_adaptation.collection.collect import rollout, runtime_metadata, write_json
from robot_adaptation.collection.provenance import source_hashes, snapshot_sources
from robot_adaptation.estimators.interaction.data import sha256
from robot_adaptation.estimators.interaction.evaluate import bootstrap_mean_ci
from robot_adaptation.estimators.interaction.predict import InteractionPredictor
from robot_adaptation.policy.train import load_checkpoint
from robot_adaptation.policy.checkpoint_metadata import checkpoint_progress

from .evaluate import (TASK_DESCRIPTIONS, estimator_inputs, evaluation_source_paths,
                       make_task, supplied_observation)


def summarize_transfer(records, prior):
    """Resample full physical conditions, retaining their evidence prefixes."""
    counts = sorted({row["attempt"] for row in records})
    conditions = sorted({row["condition"] for row in records})
    prior = np.asarray(prior)
    errors, baseline, no_contact, out_of_range = {}, [], {}, {}
    for condition in conditions:
        row = next(row for row in records if row["condition"] == condition)
        baseline.append(np.abs(prior - [row["actual_mass_kg"], row["actual_friction"]]))
    baseline = np.asarray(baseline)
    for count in counts:
        rows = [next(r for r in records if r["condition"] == c and r["attempt"] == count) for c in conditions]
        errors[count] = np.asarray([np.abs(np.asarray([r["estimate"]["mass_kg"], r["estimate"]["friction"]]) -
                                        [r["actual_mass_kg"], r["actual_friction"]]) for r in rows])
        no_contact[count] = sum(r["estimate"]["informative_attempts"] == 0 for r in rows)
        out_of_range[count] = {name: sum(r["estimate"]["out_of_training_range"][name] for r in rows)
                               for name in ("mass_kg", "friction")}
    metrics = {}
    for count in counts:
        values = errors[count]
        versus_prior = baseline - values
        versus_one = errors[1] - values
        prior_ci, more_ci = bootstrap_mean_ci(versus_prior), bootstrap_mean_ci(versus_one)
        metrics[count] = dict(mae=values.mean(axis=0).tolist(), mae_condition_bootstrap_95ci=bootstrap_mean_ci(values),
                              improvement_over_median_prior=versus_prior.mean(axis=0).tolist(),
                              improvement_condition_bootstrap_95ci=prior_ci,
                              supported_over_prior=[low > 0 for low, high in prior_ci],
                              improvement_over_one_attempt=versus_one.mean(axis=0).tolist(),
                              improvement_over_one_attempt_condition_bootstrap_95ci=more_ci,
                              supported_over_one_attempt=[low > 0 for low, high in more_ci],
                              conditions_without_contact=no_contact[count],
                              out_of_training_range_conditions=out_of_range[count])
    condition_rows = []
    for index, condition in enumerate(conditions):
        row = next(row for row in records if row["condition"] == condition)
        condition_rows.append(dict(condition_id=condition,
                                   target=[row["actual_mass_kg"], row["actual_friction"]],
                                   attempts={count: errors[count][index].tolist() for count in counts}))
    return dict(physical_conditions=len(conditions), held_conditions=len(conditions),
                conditions=condition_rows, units=["mass_kg", "object_friction_coefficient"],
                prior_median=prior.tolist(), prior_median_mae=baseline.mean(axis=0).tolist(), attempts=metrics,
                bootstrap=dict(draws=10000, seed=73518, unit="physical condition"),
                scope="Frozen collecting policy with fixed supplied estimates; new crossed physics grid and fresh starting seeds; independent attempts pooled, not adaptive retries",
                limitation="All physical conditions share the same five starting seeds; condition bootstrap quantifies physics-grid variation conditional on that fixed seed set.")


def evaluate_transfer(args):
    torch.set_num_threads(args.threads)
    task = getattr(args, "task", "vertical")
    policy_path, estimator_path = Path(args.policy).resolve(), Path(args.estimator).resolve()
    policy, trained = load_checkpoint(policy_path)
    if trained.get("observations") != 46:
        raise ValueError("Transfer collection requires the explicit46-input policy")
    policy.cpu().eval().requires_grad_(False)
    progress = checkpoint_progress(policy_path, trained)
    predictor = InteractionPredictor(estimator_path)
    prior = predictor.normalization["prior_median"]
    output = Path(args.output or ("runs/interaction_adaptation/transfer_" +
                  datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    source_files = evaluation_source_paths(task)
    source_identity = source_hashes(source_files)
    snapshot_sources(output / "source", source_files)
    hashes = dict(policy=sha256(policy_path), estimator=sha256(estimator_path))
    manifest = dict(status="running", settings=vars(args), checkpoint_sha256=hashes,
                    policy_training_status=trained.get("status"),
                    policy_training_iterations=progress["completed_learning_cycles"],
                    policy_training_steps=progress["checkpoint_transitions"],
                    original_planned_policy_training_iterations=progress["original_planned_iterations"],
                    original_planned_policy_training_steps=progress["original_planned_transitions"],
                    policy_checkpoint_progress=progress, task=task, task_description=TASK_DESCRIPTIONS[task],
                    policy=str(policy_path), estimator=str(estimator_path),
                    runtime=runtime_metadata(), source_sha256=source_identity,
                    estimator_input_boundary="input__ plus timestamp/frequency metadata only",
                    policy_input="45 collection observations including fixed supplied mass plus fixed supplied friction",
                    purpose="Controlled transfer falsification; no retraining or model selection on these data")
    write_json(output / "manifest.json", manifest)
    records, started = [], time.monotonic()
    try:
        for mass_index, mass in enumerate(args.masses):
            for friction_index, friction in enumerate(args.frictions):
                condition = f"mass{mass_index}_friction{friction_index}"
                predictor.reset()
                properties = dict(mass_kg=mass, static_friction=friction, dynamic_friction=friction,
                                  finger_friction=.2, table_friction=.3, size_xyz_m=[.04] * 3)
                env = make_task(task, properties, reported_mass_kg=args.supplied_mass, settings=trained["settings"])
                try:
                    for attempt in range(1, args.attempts + 1):
                        seed = args.seed + attempt - 1
                        with torch.inference_mode():
                            data, _ = rollout(env, lambda obs: policy.get_action(
                                supplied_observation(obs, args.supplied_friction), deterministic=True), seed,
                                dict(episode_steps=args.steps, contact_threshold_n=.05))
                        estimate = predictor.update_episode(estimator_inputs(data))
                        episode_id = f"{condition}_seed{seed}"
                        np.savez_compressed(output / "episodes" / (episode_id + ".npz"), **data)
                        row = dict(condition=condition, episode_id=episode_id, attempt=attempt, seed=seed,
                                   actual_mass_kg=mass, actual_friction=friction,
                                   supplied_mass_kg=args.supplied_mass, supplied_friction=args.supplied_friction,
                                   success=bool(data["label__held_goal_success_final"]), estimate=estimate)
                        records.append(row)
                        with (output / "attempts.jsonl").open("a") as stream:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                        print(json.dumps(row, allow_nan=False), flush=True)
                finally:
                    env.close()
        if hashes != dict(policy=sha256(policy_path), estimator=sha256(estimator_path)):
            raise RuntimeError("A pinned checkpoint changed during transfer evaluation")
        if source_hashes(source_files) != source_identity:
            raise RuntimeError("Evaluation source changed during transfer collection")
        result = summarize_transfer(records, prior)
        result.update(checkpoint=str(estimator_path), checkpoint_sha256=hashes["estimator"],
                      policy_checkpoint=str(policy_path), policy_checkpoint_sha256=hashes["policy"],
                      collecting_policy_iterations=progress["completed_learning_cycles"],
                      policy_training_iterations=progress["completed_learning_cycles"],
                      policy_training_steps=progress["checkpoint_transitions"],
                      original_planned_policy_training_iterations=progress["original_planned_iterations"],
                      original_planned_policy_training_steps=progress["original_planned_transitions"],
                      policy_checkpoint_progress=progress, task=task, task_description=TASK_DESCRIPTIONS[task])
        result["elapsed_seconds"] = time.monotonic() - started
        result["final_held_goal_successes"] = sum(row["success"] for row in records)
        result["episodes"] = len(records)
        write_json(output / "summary.json", result)
        manifest.update(status="complete", completed_episodes=len(records), elapsed_seconds=result["elapsed_seconds"])
        write_json(output / "manifest.json", manifest)
        return result
    except BaseException:
        manifest.update(status="interrupted", completed_episodes=len(records))
        write_json(output / "manifest.json", manifest)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True)
    p.add_argument("--estimator", required=True)
    p.add_argument("--output")
    p.add_argument("--masses", type=float, nargs="+", default=[.08, .16, .32, .48])
    p.add_argument("--frictions", type=float, nargs="+", default=[.25, .45, .65, .85])
    p.add_argument("--seed", type=int, default=62000)
    p.add_argument("--attempts", type=int, default=5)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--supplied-mass", type=float, default=.1)
    p.add_argument("--supplied-friction", type=float, default=.4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--task", choices=tuple(TASK_DESCRIPTIONS), default="vertical")
    args = p.parse_args(argv)
    if min(args.attempts, args.steps, args.threads) < 1 or args.seed < 0:
        p.error("Attempts, steps and threads must be positive; seed must be nonnegative")
    if any(not np.isfinite(v) or v <= 0 for v in [*args.masses, *args.frictions,
                                                                args.supplied_mass, args.supplied_friction]):
        p.error("Physical/supplied masses and friction coefficients must be finite and positive")
    print(json.dumps(evaluate_transfer(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
