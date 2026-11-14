"""Inventory archived fixed-policy lift attempts inside the policy's marginal ranges."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DENSE = ROOT / "runs/failure_collection/dense_single_pair_fixed40g"
HISTORICAL = ROOT / "runs/failure_collection/varied_fixed40g"
SELECTION = ROOT / "runs/interaction_estimator/contact_dataset_v1"
TRAJECTORY_FIELDS = [
    "input__joint_position_rad_m", "input__joint_velocity_rad_m_s", "input__action_applied",
    "input__object_pose_world_m_wxyz", "input__tcp_pose_world_m_wxyz",
    "sim__object_linear_velocity_world_m_s", "sim__object_angular_velocity_world_rad_s",
    "sim__physics_object_pair_force_world_n", "sim__is_grasped", "sim__slip_event_mask",
]


def read(path):
    return json.loads(path.read_text())


def check(value, reason):
    if not value:
        raise ValueError(reason)


def archive_key(mass, friction, seed):
    # Readbacks are float32. The design grid has at most six decimal places.
    return (round(float(mass), 6), round(float(friction), 6), int(seed))


def first_index(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if indices.size else -1


def verify_controls(z, row, fixed, checkpoint_sha):
    readback = json.loads(z["meta__physics_readback_json"].item())
    mass, friction = row["target"]
    np.testing.assert_allclose(readback["mass_kg"], mass)
    np.testing.assert_allclose(z["label__mass_kg"], mass)
    for material in readback["object_materials"]:
        np.testing.assert_allclose([material["static_friction"], material["dynamic_friction"]], friction)
    for name, value in fixed.items():
        if name == "inertia_per_kg":
            # Both the measured inertia and mass were rounded to float32 separately.
            np.testing.assert_allclose(np.array(readback["inertia_principal_kg_m2"])/mass, value,
                                       rtol=3*np.finfo(np.float32).eps)
        elif name == "shape":
            check(readback["shape"] == value, "Shape changed")
        else:
            np.testing.assert_allclose(z[name], value)
    check(z["meta__checkpoint_sha256"].item() == checkpoint_sha, "Checkpoint changed")
    np.testing.assert_allclose(z["input__reported_mass_kg"], .04)
    np.testing.assert_allclose(z["sim__policy_observation"][:,42], .04*np.linalg.norm(readback["gravity_world_m_s2"]))
    return z["meta__controller_config_json"].item()


def events(z):
    pair = z["sim__physics_object_pair_force_world_n"]
    physics_contact = np.linalg.norm(pair, axis=-1).max(axis=1) > z["meta__contact_threshold_n"]
    np.testing.assert_array_equal(physics_contact, z["sim__physics_object_contact"])
    # Map physics ticks to the issued action, including contact before a release.
    touched_actions = np.unique(z["meta__physics_action_index"][physics_contact])
    np.testing.assert_array_equal(touched_actions, np.flatnonzero(z["sim__object_contact"][1:]))
    lift = z["sim__is_grasped"] & (z["sim__object_clearance_m"] >= z["meta__lift_threshold_m"])
    first_lift = first_index(lift)
    check(first_lift == int(z["label__first_lift_state_index"]), "Lift label mismatch")
    slip = bool(np.any(z["sim__slip_event_mask"]))
    check(slip == bool(z["label__slip_observed"]), "Slip label mismatch")
    np.testing.assert_allclose([z["meta__contact_threshold_n"], z["meta__lift_threshold_m"],
                                z["label__slip_translation_threshold_m"], z["label__slip_rotation_threshold_rad"]],
                               [.05, .02, .005, .15])
    success = bool(z["label__success_final"])
    check(success == bool(z["sim__task_success"][-1]), "Final task label mismatch")
    return dict(contact_steps=int(len(touched_actions)), any_finger_contact=bool(len(touched_actions)),
                control_steps=len(z["input__action_applied"]), first_lift_state_index=first_lift,
                lifted=first_lift >= 0, slip=slip, final_task_success=success)


def totals(trials):
    flags = ("any_finger_contact", "lifted", "slip", "final_task_success")
    out = dict(attempts=len(trials))
    out.update({key: sum(row[key] for row in trials) for key in flags})
    out.update({key: sum(row[key] for row in trials) for key in ("contact_steps", "control_steps")})
    return out


def main():
    manifest = read(DENSE / "manifest.json")
    policy = manifest["training_manifest"]
    bounds_m = sorted(policy["settings"]["friction_mass_range_kg"])
    bounds_f = sorted(policy["settings"]["cube_friction_range"])
    fixed = read(SELECTION / "physical_readback_audit.json")["fixed_fields"]
    historical = {}
    for row in read(HISTORICAL / "episodes.json"):
        if set(row["changed_properties"]) <= {"mass_kg", "static_friction", "dynamic_friction"}:
            props = row["properties"]
            historical[archive_key(props["mass_kg"], props["static_friction"], row["seed"])] = row["file"]
    controls = set()
    conditions = {}
    rows = read(SELECTION / "episodes.json")
    for row in rows:
        mass, friction, seed = archive_key(*row["target"], row["seed"])
        if not (bounds_m[0] <= mass <= bounds_m[1] and bounds_f[0] <= friction <= bounds_f[1]):
            continue
        path = DENSE / row["file"]
        with np.load(path, allow_pickle=False) as z:
            check(z["label__condition_id"].item() == row["condition_id"], "Condition ID mismatch")
            check(z["label__seed"].item() == seed, "Starting seed mismatch")
            controls.add(verify_controls(z, row, fixed, manifest["checkpoint_sha256"]))
            result = dict(seed=seed, path=str(path.relative_to(ROOT)), **events(z))
            other = historical.get((mass, friction, seed))
            if other:
                with np.load(HISTORICAL / other, allow_pickle=False) as original:
                    same = all(np.array_equal(z[k], original[k]) for k in TRAJECTORY_FIELDS)
                result["historical_match"] = dict(path=str((HISTORICAL / other).relative_to(ROOT)),
                                                  trajectory_arrays_equal=same)
        condition = conditions.setdefault(row["condition_id"], dict(
            condition_id=row["condition_id"], mass_kg=mass, cube_friction=friction,
            historical_split=row["split"], trials=[]))
        check((condition["mass_kg"], condition["cube_friction"], condition["historical_split"])
              == (mass, friction, row["split"]), "Inconsistent condition")
        condition["trials"].append(result)
    conditions = sorted(conditions.values(), key=lambda c: (c["mass_kg"], c["cube_friction"]))
    for c in conditions:
        check(len(set(r["seed"] for r in c["trials"])) == len(c["trials"]), "Duplicate start")
        c["counts"] = totals(c["trials"])
    check(len(controls) == 1, "Controller settings varied between trials")
    all_trials = [r for c in conditions for r in c["trials"]]
    overlaps = [r["historical_match"] for r in all_trials if "historical_match" in r]
    coverage = {axis: dict(sorted(Counter(c[axis] for c in conditions).items()))
                for axis in ["mass_kg", "cube_friction", "historical_split"]}
    source_files = [DENSE / "manifest.json", SELECTION / "episodes.json", SELECTION / "physical_readback_audit.json",
                    HISTORICAL / "episodes.json"]
    inventory = dict(
        selection="Archived fixed-policy lift attempts with only mass and equal static/dynamic cube friction varying. "
                  "Restrict both properties to the checkpoint's marginal ranges.",
        policy_training_support=policy["friction_experiment"]["coefficients_by_mass"],
        mass_bounds_kg=bounds_m, friction_bounds=bounds_f, source_clean_conditions=len(set(r["condition_id"] for r in rows)),
        in_range_conditions=len(conditions), counts=totals(all_trials),
        conditions_with_any={k: sum(c["counts"][k] > 0 for c in conditions)
                             for k in ("any_finger_contact", "lifted", "slip", "final_task_success")},
        condition_counts_by_value=coverage, clean_historical_matches=len(overlaps),
        clean_historical_equal_trajectories=sum(o["trajectory_arrays_equal"] for o in overlaps),
        historical_arrays_compared=TRAJECTORY_FIELDS,
        definitions=dict(contact_steps="Issued commands with a physics tick during which either finger exerted "
                                       "a pair force above 0.05 N on the cube.",
                         lifted="First archived state with simultaneous grasp and cube bottom clearance of at least "
                                "0.02 m above the support. A first state index of -1 means no lift.",
                         slip="Recorded after or during a lift when relative finger/tool-to-cube motion during a "
                              "continuous grasp meets 5 mm translation or 0.15 rad rotation between state samples.",
                         final_task_success="The task's end-of-trial goal and hold criterion."),
        controls_checked=dict(checkpoint_sha256=manifest["checkpoint_sha256"], policy_guess_kg=.04,
                              controller=json.loads(controls.pop()), fixed_fields=fixed),
        sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        conditions=conditions)
    (OUT/"lift_inventory.json").write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: inventory[k] for k in ("in_range_conditions", "counts", "conditions_with_any",
                      "condition_counts_by_value", "clean_historical_matches", "clean_historical_equal_trajectories")}, indent=2))


if __name__ == "__main__":
    main()
