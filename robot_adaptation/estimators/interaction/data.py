"""Aligned snippets and condition-disjoint data preparation.

Ground-truth contacts may align saved training clips but are never features.
Every feature is a measured input__ signal or a command. Outcomes are computed
from those same observations, after the snippet ends. No reported/true mass,
friction, simulator grasp flag, episode ID or condition ID reaches the model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

FEATURE_SPECS = (
    ("input__joint_position_rad_m", "state", 9),
    ("input__joint_velocity_rad_m_s", "state", 9),
    ("input__finger_width_m", "state", 1),
    ("input__finger_width_velocity_m_s", "state", 1),
    ("input__object_pose_tcp_m_wxyz", "state", 7),
    ("input__action_applied", "transition", 5),
    ("input__grip_motor_limit_n", "transition", 2),
    ("input__finger_compression_mean_n", "transition", 2),
    ("input__finger_force_mean_world_n", "transition", 6),
    ("input__object_translation_response_world_m", "transition", 3),
    ("input__relative_translation_response_tcp_m", "transition", 3),
    ("input__tcp_translation_response_world_m", "transition", 3),
)
FEATURE_NAMES = tuple(f"{name}[{i}]" for name, _, n in FEATURE_SPECS for i in range(n))
OUTCOME_NAMES = ("observed_goal_hold", "weak_terminal_contact", "object_descended", "duration_s")
FIXED_PHYSICS_FIELDS = (
    "label__size_xyz_m", "label__com_offset_m", "label__gravity_world_m_s2",
    "label__linear_damping", "label__angular_damping",
    "label__finger_static_friction", "label__finger_dynamic_friction",
    "label__table_static_friction", "label__table_dynamic_friction",
    "label__object_restitution", "label__finger_restitution", "label__table_restitution",
    "label__goal_clearance_m", "label__spawn_offset_xy_m", "label__spawn_yaw_offset_rad",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def transition_features(data):
    """Row t contains state[t], command[t], and its measured response at t+1."""
    count = len(data["input__action_applied"])
    columns = []
    for name, timeline, size in FEATURE_SPECS:
        value = np.asarray(data[name], dtype=np.float32)
        expected = count + (timeline == "state")
        if value.shape[0] != expected:
            raise ValueError(f"Unaligned feature: {name}")
        value = value[:count].reshape(count, -1)
        if value.shape[1] != size or not np.isfinite(value).all():
            raise ValueError(f"Invalid feature: {name}")
        columns.append(value)
    return np.concatenate(columns, axis=1)


def observable_contacts(data, threshold_n=.05):
    """Causal force proxy, available without simulator object-contact masks."""
    force = np.asarray(data["input__finger_compression_mean_n"])
    return np.r_[False, np.any(force > threshold_n, axis=1)]


def observed_hold(data):
    """End-of-transition success proxy using pose, speed and measured squeeze."""
    count = len(data["input__action_applied"])
    distance = np.asarray(data["input__object_goal_distance_m"])
    speed = np.asarray(data["input__joint_velocity_rad_m_s"])
    force = np.asarray(data["input__finger_compression_mean_n"])
    return np.r_[False, (distance[1:] < .025) & (np.abs(speed[1:, :7]).max(axis=1) <= .2)
                 & (force.min(axis=1) > .5)].reshape(count + 1)


def extract_snippets(data, *, alignment="observable", release_states=3, hold_seconds=.25):
    """Return complete non-overlapping contact bouts, retaining unsuccessful touches.

    start/end are transition indices [start,end); state[end] is the final
    observation. Three absent-contact observations establish release. Sustained
    observed goal holding ends an interaction; the remaining attempt is not
    relabeled as a second interaction until contact is actually lost/reacquired.
    Oracle alignment is an explicit offline diagnostic option only.
    """
    if alignment not in ("observable", "oracle") or release_states < 1 or hold_seconds <= 0:
        raise ValueError("Invalid extraction settings")
    x = transition_features(data)
    n = len(x)
    times = np.asarray(data["meta__state_time_s"], dtype=float)
    if times.shape != (n + 1,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid state timeline")
    contacts = (np.asarray(data["sim__object_contact"], dtype=bool)
                if alignment == "oracle" else observable_contacts(data))
    if contacts.shape != (n + 1,):
        raise ValueError("Invalid contact timeline")
    held = observed_hold(data)
    snippets, i = [], 1
    while i <= n:
        if not contacts[i]:
            i += 1
            continue
        start, absent, hold_start = i - 1, 0, None
        reason, end = "timeout", n
        for j in range(i, n + 1):
            absent = 0 if contacts[j] else absent + 1
            hold_start = (j if hold_start is None else hold_start) if held[j] else None
            if absent >= release_states:
                reason, end = "release", j
                break
            if hold_start is not None and times[j] - times[hold_start] >= hold_seconds:
                reason, end = "stable_hold", j
                break
        compression = np.asarray(data["input__finger_compression_mean_n"])
        world = np.asarray(data["input__object_pose_world_m_wxyz"])
        outcome = np.asarray([held[end], np.all(compression[end - 1] < .05),
                              world[start, 2] - world[end, 2] > .02,
                              times[end] - times[start]], dtype=np.float32)
        snippets.append(dict(start=start, end=end, reason=reason,
                             x=x[start:end].copy(), outcome=outcome,
                             start_time_s=float(times[start]), end_time_s=float(times[end])))
        i = end + 1
        if reason == "stable_hold":
            while i <= n and contacts[i]:
                i += 1
    return snippets


def condition_splits(rows, seed=73):
    """Split physical configurations BEFORE extracting any clips or fitting scales."""
    conditions = sorted({row["condition_id"] for row in rows})
    if len(conditions) < 10:
        raise ValueError("Need at least ten physical conditions for held-condition evaluation")
    shuffled = np.random.default_rng(seed).permutation(conditions).tolist()
    n_test, n_val = max(2, round(.2 * len(shuffled))), max(2, round(.2 * len(shuffled)))
    return {condition: "test" if i < n_test else "validation" if i < n_test + n_val else "train"
            for i, condition in enumerate(shuffled)}


def prepare(source, output, *, seed=73, alignment="observable"):
    source, output = Path(source).resolve(), Path(output).resolve()
    rows = json.loads((source / "episodes.json").read_text())
    selected = [r for r in rows if set(r.get("changed_factors", [])) <= {"mass", "object_friction"}
                and np.isclose(r["factor_values"]["dynamic_friction_ratio"], 1)]
    # A condition ID is never a model feature; it only prevents leakage and
    # identifies episodes with identical physical parameters for offline pooling.
    split = condition_splits(selected, seed)
    output.mkdir(parents=True, exist_ok=False)
    x_rows, outcomes, offsets, metadata, episodes = [], [], [0], [], []
    no_contact = []
    fixed_physics = None
    for row in selected:
        with np.load(source / row["file"], allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        if not np.isclose(data["label__object_static_friction"], data["label__object_dynamic_friction"]):
            raise ValueError("Two-variable baseline requires equal static/dynamic friction")
        physics = {key: np.asarray(data[key]).tolist() for key in FIXED_PHYSICS_FIELDS}
        physics["shape"] = str(data["label__shape"])
        physics["inertia_per_kg"] = (data["label__inertia_principal_kg_m2"] / data["label__mass_kg"]).tolist()
        if fixed_physics is None:
            fixed_physics = physics
        elif any((physics[key] != value if key == "shape" else not np.allclose(physics[key], value))
                 for key, value in fixed_physics.items()):
            raise ValueError("Recorded nuisance physics vary inside the two-variable baseline")
        if (float(data["label__mass_kg"]) <= 0 or float(data["label__object_static_friction"]) <= 0
                or not np.isclose(data["label__mass_kg"], row["factor_values"]["mass"])
                or not np.isclose(data["label__object_static_friction"], row["factor_values"]["object_friction"])):
            raise ValueError("Physical readback disagrees with the selected condition")
        snippets = extract_snippets(data, alignment=alignment)
        ep = dict(file=row["file"], condition_id=row["condition_id"], seed=row["seed"],
                  split=split[row["condition_id"]], target=[float(data["label__mass_kg"]),
                  float(data["label__object_static_friction"])], snippets=[])
        if not snippets:
            no_contact.append(ep.copy())
        for snippet in snippets:
            index = len(metadata)
            ep["snippets"].append(index)
            x_rows.append(snippet["x"])
            outcomes.append(snippet["outcome"])
            offsets.append(offsets[-1] + len(snippet["x"]))
            metadata.append(dict(episode=row["file"], split=ep["split"], condition_id=ep["condition_id"],
                                 target=ep["target"], **{k: snippet[k] for k in
                                 ("start", "end", "reason", "start_time_s", "end_time_s")}))
        episodes.append(ep)
    all_x = np.concatenate(x_rows)
    train_x = np.concatenate([x_rows[i] for i, r in enumerate(metadata) if r["split"] == "train"])
    train_o = np.stack([outcomes[i] for i, r in enumerate(metadata) if r["split"] == "train"])
    train_targets = np.asarray([e["target"] for e in episodes if e["split"] == "train"])
    normalization = dict(mean=train_x.mean(axis=0).tolist(), std=np.maximum(train_x.std(axis=0), 1e-5).tolist(),
                         outcome_mean=train_o.mean(axis=0).tolist(),
                         outcome_std=np.maximum(train_o.std(axis=0), 1e-3).tolist(),
                         target_log_mean=np.log(train_targets).mean(axis=0).tolist(),
                         target_log_std=np.maximum(np.log(train_targets).std(axis=0), .1).tolist(),
                         prior=train_targets.mean(axis=0).tolist(), fitted_on="train only")
    np.savez_compressed(output / "snippets.npz", x=all_x, offsets=np.asarray(offsets), outcomes=np.stack(outcomes))
    write_json(output / "episodes.json", episodes)
    write_json(output / "snippets.json", metadata)
    write_json(output / "normalization.json", normalization)
    write_json(output / "no_contact.json", no_contact)
    manifest = dict(source=str(source), source_manifest_sha256=sha256(source / "manifest.json"),
                    source_index_sha256=sha256(source / "episodes.json"), seed=seed,
                    selected_episodes=len(selected), snippets=len(metadata), no_contact=len(no_contact),
                    splits={s: dict(conditions=sum(v == s for v in split.values()),
                                   episodes=sum(e["split"] == s for e in episodes),
                                   snippets=sum(e["split"] == s for e in metadata))
                            for s in ("train", "validation", "test")},
                    conditions=split, feature_names=FEATURE_NAMES, outcome_names=OUTCOME_NAMES,
                    alignment=alignment, end_reasons=dict(Counter(r["reason"] for r in metadata)),
                    physics_target="object mass kg and object material friction (static=dynamic)",
                    pooling="independently reset fixed-policy attempts at identical physical conditions; not adaptive retries",
                    selection="Only mass/object-friction factors differ; all other factors nominal",
                    fixed_physics_readback=fixed_physics,
                    scope="single-factor sweeps and two pair diagonals; not full crossed coverage")
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--alignment", choices=("observable", "oracle"), default="observable")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, seed=args.seed, alignment=args.alignment), indent=2))


if __name__ == "__main__":
    main()
