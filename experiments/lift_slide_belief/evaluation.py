"""Fit beliefs from lifts and slow slides, then evaluate held-out shove episodes."""
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


FIELDS = (
    "seed", "cube_id", "condition", "learning_episodes", "lift_episodes", "slide_episodes",
    "shove_episodes", "adaptation_steps", "shove_error", "moving_steps",
    "endpoint_error_m", "endpoint_status", "valid_endpoints", "nonfinite_endpoints",
    "unstopped_endpoints", "predicted_final_speed_m_s", "predicted_moving_endpoints",
    "donor_cube_id", "retrieved_shove_error",
    "retrieved_endpoint_error_m", "true_mass", "true_friction", "predicted_mass",
    "predicted_friction", "target_landing_error_m", "target_success_rate", "target_planner_failures",
)
TRANSITION_FIELDS = ("state", "action", "next_state", "cube_moving")
ENDPOINT_FIELDS = ("step", "force_start_step", "stop_step", "observed_stopped")
LEARNING_COUNTS = {"lifts_only": 5, "slides_only": 5, "both": 10}


def _episode_rows(rows, indices):
    # Observable transitions alone enter the latent fit and one-step predictor.
    return {key: rows[key][indices] for key in TRANSITION_FIELDS}


def _cube_groups(rows):
    ids = rows["cube_id"]
    order = np.argsort(ids, kind="stable")
    cube_ids, starts = np.unique(ids[order], return_index=True)
    return dict(zip(cube_ids.tolist(), np.split(order, starts[1:])))


def _episodes(rows, cube_indices, task):
    indices = cube_indices[rows["task"][cube_indices] == task]
    ids = rows["episode_id"][indices]
    if not np.array_equal(np.unique(ids), np.arange(10)):
        raise ValueError(f"Expected episodes 0 through 9 for task {task} on each test cube")
    return [indices[ids == episode] for episode in range(10)]


def _physics(rows, indices):
    labels = []
    for key in ("true_mass", "true_friction"):
        values = rows[key][indices]
        if not np.isfinite(values[0]) or not (values == values[0]).all():
            raise ValueError(f"{key} must be finite and constant for each cube")
        labels.append(values[0])
    return labels


def _linear_decoder(model, rows):
    groups = _cube_groups(rows)
    if set(groups) != set(model.cube_ids):
        raise ValueError("Training labels and latent table have different cube IDs")
    targets = [_physics(rows, groups[cube_id]) for cube_id in model.cube_ids]
    design = np.column_stack((model.train_z, np.ones(len(targets))))
    return np.linalg.lstsq(design, targets, rcond=None)[0]


def _prove_episodes(rows, cube_indices):
    selected = cube_indices[(rows["task"][cube_indices] == 2)
                            & np.isin(rows["episode_id"][cube_indices], np.arange(5))]
    if len(np.unique(rows["episode_id"][selected])) != 5:
        raise ValueError("Prove shoves must contain episodes 0 through 4 for each cube")
    return [selected[rows["episode_id"][selected] == episode] for episode in range(5)]


def _score(model, rows, episodes, z):
    selected = np.concatenate(episodes)
    in_shove = ((rows["step"][selected] >= rows["force_start_step"][selected])
                & (rows["step"][selected] <= rows["stop_step"][selected]))
    prove = _episode_rows(rows, selected[in_shove])
    ordered_episodes = []
    for indices in episodes:
        indices = indices[np.argsort(rows["step"][indices], kind="stable")]
        episode = _episode_rows(rows, indices)
        episode.update({key: rows[key][indices] for key in ENDPOINT_FIELDS})
        ordered_episodes.append(episode)
    endpoints = model.stopping_errors(ordered_episodes, z)
    valid = sum(result["endpoint_error_m"] is not None for result in endpoints)
    errors = [result["endpoint_error_m"] for result in endpoints]
    speeds = [result["predicted_final_speed_m_s"] for result in endpoints]
    statuses = {result["endpoint_status"] for result in endpoints}
    return dict(
        shove_error=model.moving_error(prove, z) if prove["cube_moving"].any() else None,
        moving_steps=int(np.asarray(prove["cube_moving"], dtype=bool).sum()),
        endpoint_error_m=float(np.mean(errors)) if valid == 5 else None,
        endpoint_status="ok" if valid == 5 else "|".join(sorted(statuses - {"ok", "finite"})),
        valid_endpoints=valid,
        nonfinite_endpoints=sum(r["endpoint_status"] == "nonfinite_prediction" for r in endpoints),
        unstopped_endpoints=sum(r["endpoint_status"] == "observed_not_stopped" for r in endpoints),
        predicted_final_speed_m_s=float(np.mean(speeds)) if valid == 5 else None,
        predicted_moving_endpoints=sum(speed is not None and speed > .01 for speed in speeds),
    )


def retrieve(model, test_rows, beliefs_path, cube_ids=None, condition="both"):
    """Replay recorded shoves for selected cube IDs using a saved final belief."""
    if condition not in LEARNING_COUNTS:
        raise ValueError(f"Choose a stored learning condition from {tuple(LEARNING_COUNTS)}")
    groups = _cube_groups(test_rows)
    selected = list(groups) if cube_ids is None else np.atleast_1d(cube_ids).tolist()
    with np.load(beliefs_path, allow_pickle=False) as saved:
        ids = saved["cube_id"]
        snapshots = saved["z_" + condition]
        expected = (len(ids), LEARNING_COUNTS[condition] + 1, 4)
        if len(np.unique(ids)) != len(ids) or snapshots.shape != expected:
            raise ValueError("Invalid cube IDs or belief dimensions in the saved bank")
        final_z = dict(zip(ids, snapshots[:, -1]))
        for cube_id in selected:
            if cube_id not in final_z:
                raise KeyError(f"No stored belief for cube {cube_id}")
            if cube_id not in groups:
                raise ValueError(f"No prove shoves supplied for cube {cube_id}")
        return {int(i): _score(model, test_rows, _prove_episodes(test_rows, groups[i]), final_z[i])
                for i in selected}


def _save_beliefs(path, ids, beliefs, model, ceilings, donors):
    arrays = {"cube_id": ids, "prior_mean": model.prior_mean, "prior_sd": model.prior_sd,
              "z_shove_ceiling": np.stack([ceilings[i] for i in ids]),
              "donor_cube_id": np.array([donors[i] for i in ids])}
    for condition in LEARNING_COUNTS:
        arrays["z_" + condition] = np.stack([beliefs[i][condition] for i in ids])
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        pending = Path(tmp.name)
    try:
        np.savez(pending, **arrays)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def evaluate_seed(model, train_rows, test_rows, *, seed, output_dir, fit_steps=100):
    """Evaluate one seed. Every prefix and ceiling fit receives fit_steps updates."""
    if not model.validation["passed"]:
        raise ValueError("The moving-row validation gate must pass before test evaluation")
    if fit_steps < 1:
        raise ValueError("fit_steps must be positive")
    groups = _cube_groups(test_rows)
    ids = np.array(sorted(groups))
    if len(ids) < 2 or np.intersect1d(ids, model.cube_ids).size:
        raise ValueError("Evaluation needs at least two cubes held out from training")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decoder = _linear_decoder(model, train_rows)
    supports = {}
    for cube_id in ids:
        lifts = _episodes(test_rows, groups[cube_id], task=0)[:5]
        slides = _episodes(test_rows, groups[cube_id], task=1)[:5]
        both = [episode for pair in zip(lifts, slides) for episode in pair]
        supports[cube_id] = dict(lifts_only=lifts, slides_only=slides, both=both)
    beliefs = {cube_id: {} for cube_id in ids}
    for condition, count in LEARNING_COUNTS.items():
        # Each prefix starts from the same mean and receives the same update budget.
        # Batched cubes retain independent objectives, latent parameters and Adam state.
        snapshots = [np.tile(model.prior_mean, (len(ids), 1))]
        for n in range(1, count + 1):
            observations = [_episode_rows(test_rows, np.concatenate(supports[i][condition][:n])) for i in ids]
            snapshots.append(model.fit_many_z(observations, steps=fit_steps))
        snapshots = np.stack(snapshots, axis=1)
        for i, cube_id in enumerate(ids):
            beliefs[cube_id][condition] = snapshots[i]
    # Every recipient gets one different donor, and every donor is used exactly once.
    order = np.random.default_rng(seed).permutation(ids)
    donors = dict(zip(order, np.roll(order, 1)))
    support = [_episode_rows(test_rows, np.concatenate(_episodes(test_rows, groups[i], task=2)[5:])) for i in ids]
    ceilings = dict(zip(ids, model.fit_many_z(support, steps=fit_steps)))
    beliefs_path = output_dir / "beliefs.npz"
    _save_beliefs(beliefs_path, ids, beliefs, model, ceilings, donors)
    result = []
    final_scores = {}
    for cube_id in ids:
        cube_indices = groups[cube_id]
        prove = _prove_episodes(test_rows, cube_indices)
        donor = donors[cube_id]
        truth = _physics(test_rows, cube_indices)
        conditions = [("train_mean", 0, model.prior_mean)]
        for condition in LEARNING_COUNTS:
            conditions.extend((condition, n, z) for n, z in enumerate(beliefs[cube_id][condition][1:], 1))
        conditions.extend([("other_both", 10, beliefs[donor]["both"][-1]),
                           ("shove_ceiling", 5, ceilings[cube_id])])
        for condition, n, z in conditions:
            combined = condition in ("both", "other_both")
            lifts = (n + 1) // 2 if combined else n * (condition == "lifts_only")
            slides = n // 2 if combined else n * (condition == "slides_only")
            row = dict(seed=int(seed), cube_id=int(cube_id), condition=condition,
                       learning_episodes=n, lift_episodes=lifts, slide_episodes=slides,
                       shove_episodes=n * (condition == "shove_ceiling"),
                       adaptation_steps=0 if condition == "train_mean" else fit_steps,
                       donor_cube_id=int(donor) if condition == "other_both" else "",
                       retrieved_shove_error="", retrieved_endpoint_error_m="",
                       true_mass=truth[0], true_friction=truth[1],
                       predicted_mass="", predicted_friction="", **_score(model, test_rows, prove, z))
            if condition in LEARNING_COUNTS and n == LEARNING_COUNTS[condition]:
                row["predicted_mass"], row["predicted_friction"] = np.append(z, 1.) @ decoder
                final_scores[condition, cube_id] = row
            result.append(row)

    for condition in LEARNING_COUNTS:
        for cube_id, score in retrieve(model, test_rows, beliefs_path, condition=condition).items():
            row = final_scores[condition, cube_id]
            for metric in ("shove_error", "endpoint_error_m"):
                if score[metric] is not None and row[metric] is not None:
                    np.testing.assert_allclose(score[metric], row[metric], rtol=1e-6, atol=1e-10)
                elif score[metric] != row[metric]:
                    raise AssertionError(f"Retrieved {metric} validity changed")
            if score["endpoint_status"] != row["endpoint_status"]:
                raise AssertionError("Retrieved endpoint status changed")
            row["retrieved_shove_error"] = score["shove_error"]
            row["retrieved_endpoint_error_m"] = score["endpoint_error_m"]
    return result


def write_results(rows, path):
    """Atomically replace the aggregate CSV so interrupted runs cannot truncate it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [(int(r["seed"]), int(r["cube_id"]), r["condition"], int(r["learning_episodes"])) for r in rows]
    if len(set(keys)) != len(keys) or not keys:
        raise ValueError("Result rows must contain unique (seed, cube, condition, episode count) keys")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
