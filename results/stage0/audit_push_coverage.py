"""Check existing push identities and pre-grasp tabletop motion in saved lift attempts."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


OUTPUT = Path(__file__).resolve().parent
RUNS = OUTPUT.parents[1] / "runs"
INDEX = RUNS / "interaction_estimator/contact_dataset_v1/episodes.json"
DENSE = RUNS / "failure_collection/dense_single_pair_fixed40g"
CONTEXT = RUNS / "context_dynamics"


def settings(pairs):
    # Compare stored parameters at the float32 precision of the simulator readbacks.
    return {tuple(pair) for pair in np.asarray(pairs, dtype=np.float32)}


def check_commanded_pushes(records):
    dense_settings = settings([record["target"] for record in records])
    all_pushes, all_picks = set(), set()
    sources = {}
    for name in ["data", "matched_train", "grip_pulses", "confirmation"]:
        folder = CONTEXT / "nov13" / name
        meta = json.loads((folder / "dataset.json").read_text())
        columns = [meta["constants"].index(field) for field in ("mass_kg", "cube_friction")]
        picks, pushes = set(), set()
        for shard in sorted(folder.glob("shard_*.npz")):
            with np.load(shard) as saved:
                pairs = saved["constants"][:, columns]
                tasks = saved["task_id"]
                picks.update(settings(pairs[tasks == meta["tasks"].index("PickCube-v1")]))
                pushes.update(settings(pairs[tasks == meta["tasks"].index("PushCube-v1")]))
        sources[str(folder.relative_to(RUNS))] = dict(pick_settings=len(picks), push_settings=len(pushes))
        all_picks.update(picks)
        all_pushes.update(pushes)
    index = CONTEXT / "pick_gpu_2m/data/episodes.csv"
    with index.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for source, combined in [("policy_noise", all_picks), ("scripted_push", all_pushes)]:
        pairs = settings([[row["mass_kg"], row["cube_friction"]] for row in rows if row["source"] == source])
        sources.setdefault(str(index.relative_to(RUNS)), {})[source + "_settings"] = len(pairs)
        combined.update(pairs)
    return dict(sources=sources, matching_rule="Exact mass and cube friction at float32 precision.",
                dense_lift_settings_matching_a_scripted_push=len(all_pushes & dense_settings),
                context_pick_settings_matching_a_scripted_push=len(all_pushes & all_picks))


def pre_grasp_bouts(saved):
    grasp = saved["sim__is_grasped"]
    before_first_grasp = ~np.maximum.accumulate(grasp)
    clearance = saved["sim__object_clearance_m"]
    near_table = np.abs(clearance) <= .001
    contact = saved["sim__physics_object_contact"].reshape(len(grasp) - 1, -1)
    if contact.shape[1] != 5:
        raise ValueError("Expected five physics samples per control step")
    active = before_first_grasp[1:] & near_table[:-1] & near_table[1:] & (contact.sum(axis=1) >= 3)
    xy = saved["input__object_pose_world_m_wxyz"][:, :2]
    bouts = []
    for start, end in np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2):
        if end - start < 2:
            continue
        distances = np.linalg.norm(xy[start + 2:end + 1] - xy[start], axis=1)
        peak = int(np.argmax(distances))
        bouts.append(dict(start_state=int(start), end_state=int(end), peak_state=int(start + 2 + peak),
                          horizontal_travel_mm=float(1000 * distances[peak])))
    return bouts


def audit():
    records = json.loads(INDEX.read_text())
    episodes = []
    for row in records:
        mass, friction = row["target"]
        # The tolerance accommodates float32 representation of the declared endpoints.
        if not (.04 - 1e-6 <= mass <= .5 + 1e-6 and .4 - 1e-6 <= friction <= 1 + 1e-6):
            continue
        with np.load(DENSE / row["file"]) as saved:
            bouts = pre_grasp_bouts(saved)
            episodes.append(dict(condition=row["condition_id"], split=row["split"], file=row["file"],
                                 mass_g=mass * 1000, friction=friction, seed=row["seed"], bouts=bouts,
                                 lift=bool(saved["label__first_lift_state_index"].item() >= 0),
                                 slip=bool(saved["label__slip_observed"].item()),
                                 max_travel_mm=max((b["horizontal_travel_mm"] for b in bouts), default=0.0)))
    by_cube = defaultdict(list)
    for episode in episodes:
        by_cube[episode["condition"]].append(episode)
    summaries = []
    for distance in [5, 10, 20]:
        moving = {cube for cube, rows in by_cube.items() if any(r["max_travel_mm"] >= distance for r in rows)}
        both = sorted(cube for cube in moving if any(r["lift"] for r in by_cube[cube]))
        summaries.append(dict(min_distance_mm=distance,
                              attempts_with_motion=sum(r["max_travel_mm"] >= distance for r in episodes),
                              settings_with_motion=len(moving), settings_with_motion_and_lift=both,
                              settings_with_motion_and_lift_and_slip=sum(any(r["slip"] for r in by_cube[c]) for c in both)))
    result = dict(source_index=str(INDEX.relative_to(RUNS.parent)),
                  source_folder=str(DENSE.relative_to(RUNS.parent)), episodes_scanned=len(episodes),
                  settings_scanned=len(by_cube), scripted_push_check=check_commanded_pushes(records),
                  incidental_motion_rule=(
                      "Before the first recorded grasp, cube bottom within 1 mm of table at both ends "
                      "of each control step, finger-to-cube contact above the recorder threshold (0.05 N) "
                      "on at least three of five physics samples, for at least two consecutive control "
                      "steps. Displacement is peak horizontal distance from first state of the bout "
                      "after two or more steps. The 5/10/20 mm cutoffs are exploratory. The same "
                      "setting can supply a recorded lift or slip on a different attempt."),
                  coverage=summaries, episodes=episodes)
    (OUTPUT / "push_coverage.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return summaries


if __name__ == "__main__":
    for row in audit():
        print(row["min_distance_mm"], "mm:", len(row["settings_with_motion_and_lift"]), "settings with both")
