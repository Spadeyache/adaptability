"""Evaluate actual retries, keeping simulator targets outside estimator inputs.

The old collection task supplies its established 45 observations; the explicit
friction estimate is appended for the new policy. Every matched attempt rebuilds
its physics scene. This is a CPU diagnostic, independent of PPO training seeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from robot_adaptation.collection.collect import rollout, write_json
from robot_adaptation.collection.provenance import source_hashes, snapshot_sources
from robot_adaptation.estimators.interaction.predict import InteractionPredictor
from robot_adaptation.paths import ROOT
from robot_adaptation.policy.checkpoint_metadata import checkpoint_progress
from robot_adaptation.policy.train import load_checkpoint
from robot_adaptation.simulation.collection_lift import WeightedCollectionEnv

TASK_DESCRIPTIONS = {
    "vertical": "Legacy vertical pick-and-hold: goal directly above the spawn at fixed 0.2 m clearance",
    "native_carry": "Native trained carry goal geometry: randomized PickCube XY and saved floating-Z clearance range; CPU adapter preserves trained reward",
}


def make_task(task, properties, *, reported_mass_kg, settings):
    """Keep legacy protocol the default; explicitly select the trained carry geometry."""
    if task == "vertical":
        return WeightedCollectionEnv(properties, reported_mass_kg=reported_mass_kg,
            goal_clearance_m=.2, grip_force_limits_n=settings["grip_force_limits_n"])
    if task == "native_carry":
        # Lazy import: the adapter also imports supplied_observation from here.
        from experiments.physics_conditioning.checkpoint_probe import ProbeCarryTask
        return ProbeCarryTask(properties, reported_mass_kg=reported_mass_kg,
            goal_clearance_m=settings["goal_clearance_m"],
            grip_force_limits_n=settings["grip_force_limits_n"],
            grip_force_cost=settings["grip_force_cost"])
    raise ValueError("Choose vertical or native_carry task geometry")


def evaluation_source_paths(task):
    if task not in TASK_DESCRIPTIONS:
        raise ValueError("Unknown evaluation task")
    paths = [Path(__file__), Path(__file__).with_name("transfer.py"),
             ROOT / "robot_adaptation/policy/checkpoint_metadata.py",
             ROOT / "robot_adaptation/policy/model.py", ROOT / "robot_adaptation/policy/train.py",
             ROOT / "robot_adaptation/policy/trust_region.py",
             ROOT / "robot_adaptation/policy/examples/maniskill_pick_cube/ppo_upstream.py"]
    paths += list((ROOT / "robot_adaptation/estimators/interaction").glob("*.py"))
    if task == "native_carry":
        paths += [ROOT / "experiments/physics_conditioning/checkpoint_probe.py",
                  ROOT / "experiments/physics_conditioning/evaluate.py"]
    return paths


def estimator_inputs(data):
    """The prediction boundary excludes simulator masks and supervised targets."""
    return {key: value for key, value in data.items()
            if key.startswith("input__") or key in {
                "meta__state_time_s", "meta__control_frequency_hz",
                "meta__simulation_frequency_hz"}}


def supplied_observation(observation, friction):
    """Append a supplied coefficient without changing the legacy observations."""
    if tuple(observation.shape) != (1, 45):
        raise ValueError("Expected the collection task's 45-input observation")
    if not np.isfinite(friction) or friction <= 0:
        raise ValueError("Supplied friction must be finite and positive")
    coefficient = torch.full((1, 1), float(friction), dtype=observation.dtype,
                             device=observation.device)
    return torch.cat((observation, coefficient), dim=-1)


def summarize(records, budget):
    """Compare equal-budget sequence outcomes and paired changes in success."""
    methods = sorted({row["method"] for row in records})
    summary = {}
    for method in methods:
        rows = [row for row in records if row["method"] == method]
        groups = {}
        for row in rows:
            groups.setdefault(row["sequence_key"], []).append(row)
        success = {key: any(row["success"] for row in values)
                   for key, values in groups.items()}
        summary[method] = {"sequences": len(groups), "attempts": len(rows),
                           "successes_within_budget": sum(success.values()),
                           "success_rate_within_budget": float(np.mean(list(success.values()))),
                           "by_sequence": success}
    if "adaptive" in summary and "fixed" in summary:
        keys = sorted(set(summary["adaptive"]["by_sequence"]) &
                      set(summary["fixed"]["by_sequence"]))
        difference = np.asarray([int(summary["adaptive"]["by_sequence"][key]) -
                                 int(summary["fixed"]["by_sequence"][key]) for key in keys])
        if len(difference):
            # Conditions share the same seed sequence. Keep that dependence
            # when resampling, rather than counting conditions as new seeds.
            blocks = {}
            for key, value in zip(keys, difference):
                block = next((str(row["sequence_index"]) for row in records
                              if row["sequence_key"] == key and "sequence_index" in row),
                             key.rsplit("_sequence", 1)[-1])
                blocks.setdefault(block, []).append(value)
            block_differences = np.asarray([np.mean(values) for values in blocks.values()])
            rng = np.random.default_rng(73519)
            means = rng.choice(block_differences, size=(10000, len(block_differences)), replace=True).mean(axis=1)
            summary["adaptive_vs_fixed"] = {
                "paired_sequences": len(keys), "success_rate_difference": float(difference.mean()),
                "independent_seed_blocks": len(blocks),
                "bootstrap_95_interval": np.quantile(means, [.025, .975]).tolist(),
                "supported_on_this_sample": bool(np.quantile(means, .025) > 0),
                "limitation": "Seed-block bootstrap retains every physical condition; small numbers of seed blocks limit inference."}
    return {"attempt_budget": budget, "methods": summary,
            "scope": "Actual retries; a bounded diagnostic does not establish broad reliability."}


def evaluate(args):
    torch.set_num_threads(args.threads)
    task = getattr(args, "task", "vertical")
    policy_path = Path(args.policy).resolve()
    estimator_path = Path(args.estimator).resolve()
    policy, manifest = load_checkpoint(policy_path)
    if manifest.get("observations") != 46:
        raise ValueError("Retry evaluation requires the explicit 46-input friction policy")
    policy.cpu().eval().requires_grad_(False)
    progress = checkpoint_progress(policy_path, manifest)
    predictor = InteractionPredictor(estimator_path)
    conditions = json.loads(Path(args.conditions).read_text()) if args.conditions else [
        {"mass_kg": .04, "friction": .2},
        {"mass_kg": .1, "friction": .4},
        {"mass_kg": .5, "friction": .8}]
    output = Path(args.output or ("runs/interaction_adaptation/" +
                  datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    checkpoint_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in (("policy", policy_path), ("estimator", estimator_path))}
    settings = dict(vars(args), conditions=conditions, policy=str(policy_path),
                    estimator=str(estimator_path), output=str(output))
    source_paths = evaluation_source_paths(task)
    source_identity = source_hashes(source_paths)
    snapshot_sources(output / "source", source_paths)
    write_json(output / "manifest.json", {
        "status": "running", "settings": settings, "checkpoint_sha256": checkpoint_hashes,
        "policy_training_status": manifest.get("status"),
        "policy_training_steps": progress["checkpoint_transitions"],
        "policy_training_iterations": progress["completed_learning_cycles"],
        "original_planned_policy_training_steps": progress["original_planned_transitions"],
        "original_planned_policy_training_iterations": progress["original_planned_iterations"],
        "policy_checkpoint_progress": progress,
        "task": task, "task_description": TASK_DESCRIPTIONS[task], "source_sha256": source_identity,
        "estimator_input_boundary": "input__ fields and timestamp/frequency metadata only",
        "method": "fixed policy, true/fixed/adaptive estimates, fresh matched scene resets",
        "assessment_spec": str(Path("configs/adaptation_assessment.json").resolve())})
    records = []
    started = time.monotonic()
    for index, condition in enumerate(conditions):
        mass, friction = float(condition["mass_kg"]), float(condition["friction"])
        properties = dict(mass_kg=mass, static_friction=friction, dynamic_friction=friction,
                          finger_friction=.2, table_friction=.3, size_xyz_m=[.04] * 3)
        for sequence in range(args.sequences):
            for method in ("fixed", "adaptive", "true"):
                predictor.reset()
                estimated_mass, estimated_friction = args.initial_mass, args.initial_friction
                env = make_task(task, properties, reported_mass_kg=estimated_mass, settings=manifest["settings"])
                try:
                    for attempt in range(args.attempts):
                        if method == "true":
                            estimated_mass, estimated_friction = mass, friction
                        env.reported_mass_kg = estimated_mass
                        seed = args.seed + sequence * args.attempts + attempt
                        with torch.inference_mode():
                            data, _ = rollout(env,
                                lambda observation: policy.get_action(
                                    supplied_observation(observation, estimated_friction),
                                    deterministic=True), seed,
                                dict(episode_steps=args.steps, contact_threshold_n=.05))
                        success = bool(data["label__held_goal_success_final"])
                        episode_id = f"condition{index}_sequence{sequence}_{method}_attempt{attempt}"
                        np.savez_compressed(output / "episodes" / (episode_id + ".npz"), **data)
                        updated = None
                        if method == "adaptive":
                            updated = predictor.update_episode(estimator_inputs(data))
                            if updated["informative_attempts"] == 0:
                                # A no-contact miss supplies no physical evidence.
                                # Keep the same initial guess as the fixed arm.
                                updated = dict(updated, mass_kg=estimated_mass,
                                               friction=estimated_friction)
                        row = dict(episode_id=episode_id, condition_index=index,
                            sequence_key=f"condition{index}_sequence{sequence}",
                            sequence_index=sequence,
                            method=method, attempt=attempt + 1, seed=seed,
                            actual_mass_kg=mass, actual_friction=friction,
                            supplied_mass_kg=estimated_mass, supplied_friction=estimated_friction,
                            success=success, post_attempt_estimate=updated)
                        records.append(row)
                        with (output / "attempts.jsonl").open("a") as stream:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                        print(json.dumps(row, allow_nan=False), flush=True)
                        if success:
                            break
                        if updated is not None:
                            estimated_mass, estimated_friction = updated["mass_kg"], updated["friction"]
                finally:
                    env.close()
    result = summarize(records, args.attempts)
    result.update(task=task, task_description=TASK_DESCRIPTIONS[task],
                  policy_checkpoint=str(policy_path), estimator_checkpoint=str(estimator_path),
                  checkpoint_sha256=checkpoint_hashes, policy_checkpoint_progress=progress,
                  policy_training_iterations=progress["completed_learning_cycles"],
                  policy_training_steps=progress["checkpoint_transitions"],
                  original_planned_policy_training_iterations=progress["original_planned_iterations"],
                  original_planned_policy_training_steps=progress["original_planned_transitions"])
    if source_hashes(source_paths) != source_identity:
        raise RuntimeError("Evaluation source changed during retry collection")
    if checkpoint_hashes != {name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for name, path in (("policy", policy_path), ("estimator", estimator_path))}:
        raise RuntimeError("A pinned checkpoint changed during retry collection")
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "summary.json", result)
    saved = json.loads((output / "manifest.json").read_text())
    saved["status"] = "complete"
    saved["completed_attempts"] = len(records)
    write_json(output / "manifest.json", saved)
    print(json.dumps({"output": str(output), "summary": result}, indent=2), flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--estimator", required=True)
    parser.add_argument("--conditions", help="JSON list of mass_kg/friction settings")
    parser.add_argument("--output")
    parser.add_argument("--sequences", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=51000)
    parser.add_argument("--initial-mass", type=float, default=.1)
    parser.add_argument("--initial-friction", type=float, default=.4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--task", choices=tuple(TASK_DESCRIPTIONS), default="vertical")
    args = parser.parse_args(argv)
    for name in ("sequences", "attempts", "steps", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    for name in ("initial_mass", "initial_friction"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    return evaluate(args)


if __name__ == "__main__":
    main()
