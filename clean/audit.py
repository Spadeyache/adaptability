"""Training-cube-only collection audits and supervised diagnostic probes.

Physical labels are read solely for correlations and the two explicitly labelled
diagnostic probes. They never become dynamics inputs here. No held-out test or
out-of-range episode path is opened by this module.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from clean.protocol import PROTOCOL, SEEDS, verify
from clean.schema import ACTION_COLUMNS, COLUMNS


ROOT = Path(__file__).resolve().parent
PROBE_UPDATES = PROTOCOL["probe_updates"]
PROBE_HIDDEN = 32
PROBE_BATCH = 32
PROBE_LR = 0.003
PROPERTIES = ("mass_kg", "friction")
SUMMARY_NAMES = ("mean", "std", "min", "max")
ALLOWED_KEYS = {"state", "action", "next_state", "cube_id", "episode_id", "step"}
OLD_MODULES = {
    "experiments", "robot_adaptation", "crest", "data_check", "failure_collection",
    "failure_discovery", "grip_support", "visual_mass_estimator", "toy",
}
PHYSICAL_PREFIXES = (
    "joint_position_", "joint_velocity_", "joint_effort_", "gripper_opening_",
    "finger_position_error_", "cube_position_", "cube_rotation_",
    "cube_linear_velocity_", "cube_angular_velocity_", "cube_relative_position_",
    "cube_relative_rotation_",
)
CRITICAL_COLLECTION_SOURCES = ("schema.py", "script.py", "native.py")


def summarize(values):
    """Five-seed sample standard deviation; undefined statistics remain null."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or len(values) < 2:
        return {"mean": None, "std": None, "n": len(values)}
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "n": len(values)}


def r2_per_property(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = np.square(target - target.mean(axis=0)).sum(axis=0)
    numerator = np.square(target - prediction).sum(axis=0)
    return np.divide(
        denominator - numerator, denominator,
        out=np.full(denominator.shape, np.nan), where=denominator > 1e-20,
    )


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator <= 1e-20:
        return None
    return float(np.clip(np.dot(x, y) / denominator, -1, 1))


def moving_mask(state, next_state):
    """True cube translation >1e-4 m OR true rotation >1e-3 radians."""
    names = [column["name"] for column in COLUMNS]
    position = [names.index(f"cube_position_{i}") for i in range(3)]
    rotation = [names.index(f"cube_rotation_{i}") for i in range(6)]
    a = np.asarray(state[:, rotation], dtype=np.float64)
    b = np.asarray(next_state[:, rotation], dtype=np.float64)
    # The schema stores the first two rotation-matrix columns consecutively.
    ra = np.stack((a[:, :3], a[:, 3:], np.cross(a[:, :3], a[:, 3:])), axis=-1)
    rb = np.stack((b[:, :3], b[:, 3:], np.cross(b[:, :3], b[:, 3:])), axis=-1)
    cosine = np.clip((np.sum(ra * rb, axis=(1, 2)) - 1) / 2, -1, 1)
    angle = np.arccos(cosine)
    displacement = np.linalg.norm(next_state[:, position] - state[:, position], axis=1)
    return (displacement > 1e-4) | (angle > 1e-3)


def static_provenance(root=ROOT):
    """Record source hashes/imports and check the measurement/planning boundary."""
    root = Path(root)
    imports = {}
    source_hashes = {}
    violations = []
    trees = {}
    for path in sorted(root.glob("*.py")):
        raw = path.read_bytes()
        source_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw, filename=str(path))
        trees[path.name] = tree
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        imports[path.name] = modules
        for module in modules:
            if module.split(".")[0] in OLD_MODULES:
                violations.append(f"{path.name}: forbidden old-project import {module}")
    # The source schema remains the authoritative explicit whitelist.
    for column in COLUMNS:
        if not column["name"].startswith(PHYSICAL_PREFIXES):
            violations.append(f"Unrecognized physical state column: {column['name']}")
        if any(word in column["name"].lower() for word in ("mass", "friction", "inertia")):
            violations.append(f"Physical label named as state: {column['name']}")
    for name in ACTION_COLUMNS:
        if any(word in name.lower() for word in ("mass", "friction", "inertia")):
            violations.append(f"Physical label named as action: {name}")
    for filename, function_names in (("schema.py", {"observe"}), ("script.py", {"plan_episode", "sample_settings"})):
        tree = trees.get(filename)
        if tree is None:
            violations.append(f"Missing provenance source: {filename}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in function_names:
                continue
            if node.name == "plan_episode":
                arguments = [argument.arg for argument in node.args.args]
                if arguments != ["cube_pose", "settings"]:
                    violations.append(f"Unexpected planner input signature: {arguments}")
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and any(
                    word in child.attr.lower() for word in ("mass", "friction", "inertia", "physical_material")
                ):
                    violations.append(f"{filename}:{child.lineno}: property read {child.attr}")
                if isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value in {
                    "mass", "mass_kg", "friction", "inertia", "physical_material", "cubes.json"
                }:
                    violations.append(f"{filename}:{child.lineno}: physical property key {child.value}")
    return {
        "pass": not violations, "violations": violations, "imports": imports,
        "source_sha256": source_hashes,
        "limitation": "AST import/property checks support, but do not replace, the explicit source-path audit.",
    }


def verify_collection_receipts(data_root, cube_ids):
    """Cheap independent receipt check; never runs or repeats diagnostic probes."""
    data_root = Path(data_root).resolve()
    expected_sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in CRITICAL_COLLECTION_SOURCES
    }
    verified = 0
    for cube_id in cube_ids:
        if not 0 <= cube_id < 200:
            raise ValueError(f"Receipt audit forbidden from loading nontraining cube {cube_id}")
        for episode in range(10):
            path = data_root / "train" / f"cube_{cube_id:04d}_episode_{episode:02d}.npz"
            receipt = json.loads(path.with_suffix(".json").read_text())
            if receipt["cube_id"] != cube_id or receipt["episode_id"] != episode:
                raise ValueError(f"Collection receipt identity mismatch: {path.name}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
                raise ValueError(f"Collection receipt data digest mismatch: {path.name}")
            for name, expected in expected_sources.items():
                if receipt["source_sha256"].get(name) != expected:
                    raise ValueError(f"Collection provenance differs from audited {name}: {path.name}")
            verified += 1
    return {"pass": True, "verified_episode_receipts": verified,
            "critical_source_sha256": expected_sources,
            "scope": "Training cube episode receipts and data only; current critical source must match collection source."}


def load_training_episodes(data_root, cube_ids):
    """Load named train paths only; reject unexpected fields and inconsistent IDs."""
    data_root = Path(data_root).resolve()
    train_root = data_root / "train"
    summaries = []
    probe_states = []
    probe_actions = []
    episode_reports = []
    hashes = {}
    reference_actions = {}
    all_actions_match = True
    for cube_id in cube_ids:
        if not 0 <= cube_id < 200:
            raise ValueError(f"Audit forbidden from loading nontraining cube {cube_id}")
        cube_states, cube_actions = [], []
        for episode in range(10):
            path = train_root / f"cube_{cube_id:04d}_episode_{episode:02d}.npz"
            if path.resolve().parent != train_root.resolve():
                raise ValueError(f"Unexpected episode path: {path}")
            hashes[str(path.relative_to(data_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != ALLOWED_KEYS:
                    raise ValueError(f"Unexpected fields in {path.name}: {archive.files}")
                state = np.asarray(archive["state"], dtype=np.float32)
                next_state = np.asarray(archive["next_state"], dtype=np.float32)
                action = np.asarray(archive["action"], dtype=np.float32)
                n = len(state)
                if n == 0 or state.shape != (n, len(COLUMNS)) or next_state.shape != state.shape:
                    raise ValueError(f"Invalid state shape in {path.name}")
                if action.shape != (n, len(ACTION_COLUMNS)):
                    raise ValueError(f"Invalid action shape in {path.name}")
                if not all(np.isfinite(array).all() for array in (state, next_state, action)):
                    raise ValueError(f"Nonfinite physical measurements in {path.name}")
                if not np.array_equal(archive["step"], np.arange(n)):
                    raise ValueError(f"Invalid step index in {path.name}")
                if not np.array_equal(archive["cube_id"], np.full(n, cube_id)):
                    raise ValueError(f"Invalid cube identity in {path.name}")
                if not np.array_equal(archive["episode_id"], np.full(n, episode)):
                    raise ValueError(f"Invalid episode identity in {path.name}")
                if n > 1 and not np.array_equal(state[1:], next_state[:-1]):
                    raise ValueError(f"Transition discontinuity in {path.name}")
            if episode not in reference_actions:
                reference_actions[episode] = action.copy()
            elif not np.array_equal(reference_actions[episode], action):
                all_actions_match = False
            moving = moving_mask(state, next_state)
            episode_reports.append({
                "cube_id": int(cube_id), "episode_id": episode, "steps": n,
                "moving_steps": int(moving.sum()), "finite": True,
            })
            if episode < 8:
                cube_states.append(state)
                cube_actions.append(action)
            if episode == 0:
                probe_states.append(state)
                probe_actions.append(action)
        matrix = np.concatenate((np.concatenate(cube_states), np.concatenate(cube_actions)), axis=1)
        summaries.append(np.stack((matrix.mean(axis=0), matrix.std(axis=0), matrix.min(axis=0), matrix.max(axis=0))))
    return {
        "summaries": np.asarray(summaries, dtype=np.float64),
        "probe_states": probe_states, "probe_actions": probe_actions,
        "episodes": episode_reports, "episode_sha256": hashes,
        "actions_bitwise_matched_between_cubes": all_actions_match,
    }


def correlation_report(cube_summaries, labels):
    names = [column["name"] for column in COLUMNS] + list(ACTION_COLUMNS)
    sources = [column["source"] for column in COLUMNS] + [
        "clean/script.py:plan_episode(cube_pose, settings), preplanned command; no dynamics feedback"
        for _ in ACTION_COLUMNS
    ]
    records = []
    flags = []
    for index, (name, source) in enumerate(zip(names, sources)):
        correlations = {}
        kind = "physical_sensor" if index < len(COLUMNS) else "independent_command"
        record = {
            "column": name, "kind": kind, "code_path": "clean/schema.py:observe" if kind == "physical_sensor" else "clean/script.py:plan_episode",
            "provenance": source, "correlations": correlations,
        }
        for summary_index, summary_name in enumerate(SUMMARY_NAMES):
            correlations[summary_name] = {}
            for property_index, property_name in enumerate(PROPERTIES):
                coefficient = pearson(cube_summaries[:, summary_index, index], labels[:, property_index])
                correlations[summary_name][property_name] = coefficient
                if coefficient is not None and abs(coefficient) > 0.9:
                    classification = "physical_sensor" if kind == "physical_sensor" else "leak_requires_investigation"
                    flags.append({
                        "column": name, "summary": summary_name, "property": property_name,
                        "r": coefficient, "classification": classification, "provenance": source,
                        "reason": "Measured physical response; no property lookup in observe()." if kind == "physical_sensor" else "Command-label association must be resolved before collection/training continues.",
                    })
        records.append(record)
    return {
        "pass": not any(flag["classification"] == "leak_requires_investigation" for flag in flags),
        "aggregation": "Per-cube mean/std/min/max across episodes 0–7; Pearson across training cubes.",
        "constant_column_correlation": None,
        "columns": records, "flags_abs_r_gt_0_9": flags,
    }


class EpisodeProbe(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.gru = nn.GRU(features, PROBE_HIDDEN, batch_first=True)
        self.head = nn.Linear(PROBE_HIDDEN, 2)

    def forward(self, sequence, lengths):
        packed = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(hidden[-1])


def probe_fold(sequences, labels, train_indices, heldout_indices, seed, updates=PROBE_UPDATES):
    """Normalization and all gradient updates use the training cube fold only."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_values = np.concatenate([sequences[index] for index in train_indices]).astype(np.float64)
    x_mean = train_values.mean(axis=0)
    x_std = np.maximum(train_values.std(axis=0), 1e-6)
    y_mean = labels[train_indices].mean(axis=0)
    y_std = np.maximum(labels[train_indices].std(axis=0), 1e-6)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.int64)
    padded = np.zeros((len(sequences), int(lengths.max()), len(x_mean)), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        padded[index, :len(sequence)] = (sequence - x_mean) / x_std
    x = torch.from_numpy(padded)
    y = torch.tensor((labels - y_mean) / y_std, dtype=torch.float32)
    model = EpisodeProbe(len(x_mean))
    optimizer = torch.optim.Adam(model.parameters(), lr=PROBE_LR)
    model.train()
    for _ in range(updates):
        selected = rng.choice(train_indices, size=min(PROBE_BATCH, len(train_indices)), replace=False)
        prediction = model(x[selected], lengths[selected])
        loss = torch.square(prediction - y[selected]).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite diagnostic probe loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(x[heldout_indices], lengths[heldout_indices]).numpy() * y_std + y_mean
    return r2_per_property(labels[heldout_indices], prediction)


def run_probes(states, actions, labels, cube_ids, updates=PROBE_UPDATES):
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    state_action = [np.concatenate((state, action), axis=1) for state, action in zip(states, actions)]
    results = {"actions_only": [], "state_action": [], "actions_constant_reference": []}
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(cube_ids))
        n_heldout = max(2, int(np.ceil(len(cube_ids) * 0.2)))
        heldout_indices, train_indices = order[:n_heldout], order[n_heldout:]
        if len(train_indices) < 2:
            raise ValueError("Insufficient training cubes for a held-out-cube probe")
        fold = {"seed": seed, "train_cube_ids": [int(cube_ids[index]) for index in train_indices],
                "heldout_cube_ids": [int(cube_ids[index]) for index in heldout_indices]}
        for name, sequences in (("actions_only", actions), ("state_action", state_action)):
            scores = probe_fold(sequences, labels, train_indices, heldout_indices, seed, updates=updates)
            record = dict(fold, r2={prop: float(score) if np.isfinite(score) else None for prop, score in zip(PROPERTIES, scores)})
            results[name].append(record)
            print(json.dumps({"audit_probe": name, "seed": seed, "r2": record["r2"]}), flush=True)
        constant = np.broadcast_to(labels[train_indices].mean(axis=0), (n_heldout, 2))
        scores = r2_per_property(labels[heldout_indices], constant)
        results["actions_constant_reference"].append(dict(fold, r2={prop: float(score) if np.isfinite(score) else None for prop, score in zip(PROPERTIES, scores)}))
    output = {
        "configuration": {"episode_id": 0, "seeds": list(SEEDS), "heldout_fraction": 0.2,
            "hidden_size": PROBE_HIDDEN, "gru_layers": 1, "optimizer": "Adam", "learning_rate": PROBE_LR,
            "optimizer_updates": updates, "maximum_batch_cubes": PROBE_BATCH,
            "normalization": "Training-cube fold only; whole episode sequence, no downsampling.",
            "target_labels": "Read from cubes.json for supervised diagnostic probes only."},
        "f8_interpretation": "Empirical one-episode recoverability benchmark, not a mathematical signal ceiling or upper bound; five-lift z uses more evidence.",
    }
    for name, records in results.items():
        output[name] = {"seeds": records, "r2": {prop: summarize([record["r2"][prop] if record["r2"][prop] is not None else np.nan for record in records]) for prop in PROPERTIES}}
    output["f2_pass"] = all(
        record["r2"][prop] is not None and record["r2"][prop] < 0.1
        for record in results["actions_only"] for prop in PROPERTIES
    )
    output["f2_rule"] = "Both properties must have held-out-cube R² < 0.1 in each of five seeds; threshold not relaxed for pilot."
    return output


def save_report(report, stage, results_root):
    results_root = Path(results_root)
    archive = results_root / "audits"
    archive.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    history = archive / f"{stage}_{stamp}.json"
    report["saved_run"] = str(history.relative_to(results_root.parent))
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with history.open("x") as handle:
        handle.write(serialized)
    temporary = results_root / f".{stage}_audit.{stamp}.tmp"
    temporary.write_text(serialized)
    temporary.replace(results_root / f"{stage}_audit.json")


def run_audit(stage, data_root=None, results_root=None):
    if stage not in {"pilot", "full"}:
        raise ValueError(stage)
    data_root = Path(data_root) if data_root else ROOT / "data"
    results_root = Path(results_root) if results_root else ROOT / "results"
    cube_ids = list(range(10 if stage == "pilot" else 200))
    started = time.monotonic()
    report = {
        "stage": stage, "status": "running", "cube_ids": cube_ids,
        "heldout_test_or_oor_episodes_read": False,
        "label_storage": "True per-cube properties are read only from data/cubes.json and never copied to this report.",
        "warning": "Pilot probe folds have only two held-out training cubes; noisy R² is not strong evidence of recoverability." if stage == "pilot" else None,
    }
    try:
        if data_root.resolve() == (ROOT / "data").resolve():
            verify()
            report["immutable_protocol_verified"] = True
        property_path = data_root / "cubes.json"
        property_bytes = property_path.read_bytes()
        properties = {int(cube["cube_id"]): cube for cube in json.loads(property_bytes)["cubes"]}
        for cube_id in cube_ids:
            if properties[cube_id]["split"] != "train":
                raise ValueError(f"Cube {cube_id} is not designated train")
        labels = np.array([[properties[cube_id][prop] for prop in PROPERTIES] for cube_id in cube_ids], dtype=np.float64)
        if not np.isfinite(labels).all():
            raise ValueError("Nonfinite probe labels")
        report["properties_sha256"] = hashlib.sha256(property_bytes).hexdigest()
        split_path = data_root / "split.json"
        if not split_path.exists():
            raise FileNotFoundError("Immutable split.json is required before collection audit")
        report["split_sha256"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
        report["static_provenance"] = static_provenance()
        if not report["static_provenance"]["pass"]:
            raise ValueError("Static provenance audit failed")
        report["collection_receipts"] = verify_collection_receipts(data_root, cube_ids)
        loaded = load_training_episodes(data_root, cube_ids)
        report["episodes"] = loaded["episodes"]
        report["episode_sha256"] = loaded["episode_sha256"]
        report["actions_bitwise_matched_between_cubes"] = loaded["actions_bitwise_matched_between_cubes"]
        report["finite_data_pass"] = True
        report["episode_count"] = len(loaded["episodes"])
        report["total_steps"] = sum(episode["steps"] for episode in loaded["episodes"])
        report["total_moving_steps"] = sum(episode["moving_steps"] for episode in loaded["episodes"])
        report["all_cubes_have_moving_steps"] = all(any(episode["cube_id"] == cube_id and episode["moving_steps"] > 0 for episode in loaded["episodes"]) for cube_id in cube_ids)
        report["f1"] = correlation_report(loaded["summaries"], labels)
        report["probes"] = run_probes(loaded["probe_states"], loaded["probe_actions"], labels, cube_ids)
        report["f2_pass"] = report["probes"]["f2_pass"]
        report["f3_pass"] = True
        report["f7_pass"] = all(len(report["probes"][name]["seeds"]) == 5 for name in ("actions_only", "state_action"))
        report["f8_pass"] = all(report["probes"]["state_action"]["r2"][prop]["mean"] is not None for prop in PROPERTIES)
        passed = all((report["f1"]["pass"], report["f2_pass"], report["f3_pass"], report["f7_pass"], report["f8_pass"], report["all_cubes_have_moving_steps"]))
        report["status"] = "pass" if passed else "fail"
    except Exception as error:
        report["status"] = "fail"
        report["error"] = repr(error)
    report["pass"] = report["status"] == "pass"
    report["elapsed_seconds"] = time.monotonic() - started
    save_report(report, stage, results_root)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    arguments = parser.parse_args()
    report = run_audit(arguments.stage)
    print(json.dumps({"stage": arguments.stage, "status": report["status"], "error": report.get("error"), "saved_run": report["saved_run"]}), flush=True)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
