"""Fit test-cube beliefs on five lifts and score identical held-out slide transitions."""
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


FIELDS = (
    "seed", "cube_id", "condition", "number_of_lifts", "slide_error", "moving_steps",
    "donor_cube_id", "retrieved_slide_error", "true_mass", "true_friction",
    "predicted_mass", "predicted_friction",
)
TRANSITION_FIELDS = ("state", "action", "next_state", "cube_moving")


def _episode_rows(rows, indices):
    # Only observable inputs and the motion mask reach the model.
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


def _five_lift_snapshots(model, rows, episodes, fit_steps):
    z = None
    snapshots = []
    for count in range(1, 6):
        support = _episode_rows(rows, np.concatenate(episodes[:count]))
        z = model.fit_z(support, initial=z, steps=fit_steps)
        snapshots.append(z.copy())
    return np.stack(snapshots)


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


def _prove_rows(rows, cube_indices):
    selected = cube_indices[(rows["task"][cube_indices] == 1)
                            & np.isin(rows["episode_id"][cube_indices], np.arange(5))]
    if len(np.unique(rows["episode_id"][selected])) != 5:
        raise ValueError("Prove slides must contain episodes 0 through 4 for each cube")
    prove = _episode_rows(rows, selected)
    if not prove["cube_moving"].any():
        raise ValueError(f"No moving steps in prove slides for cube {rows['cube_id'][cube_indices[0]]}")
    return prove


def retrieve(model, test_rows, beliefs_path, cube_ids=None):
    """Score prove slides for the requested cube ID(s), defaulting to all IDs in test_rows."""
    groups = _cube_groups(test_rows)
    selected = list(groups) if cube_ids is None else np.atleast_1d(cube_ids).tolist()
    with np.load(beliefs_path, allow_pickle=False) as saved:
        ids = saved["cube_id"]
        snapshots = saved["z_after_lift"]
        if len(np.unique(ids)) != len(ids) or snapshots.shape != (len(ids), 5, 4):
            raise ValueError("Invalid cube IDs or belief dimensions in the saved bank")
        final_z = dict(zip(ids, snapshots[:, -1]))
        for cube_id in selected:
            if cube_id not in final_z:
                raise KeyError(f"No stored belief for cube {cube_id}")
            if cube_id not in groups:
                raise ValueError(f"No prove slides supplied for cube {cube_id}")
        return {
            int(i): model.moving_error(_prove_rows(test_rows, groups[i]), final_z[i]) for i in selected
        }


def evaluate_seed(model, train_rows, test_rows, *, seed, output_dir, fit_steps=100):
    """Evaluate one trained seed. Pass its own directory for the persisted beliefs."""
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
    beliefs = {}
    for cube_id in ids:
        lifts = _episodes(test_rows, groups[cube_id], task=0)
        beliefs[cube_id] = _five_lift_snapshots(model, test_rows, lifts[:5], fit_steps)
    beliefs_path = output_dir / "beliefs.npz"
    with NamedTemporaryFile(dir=output_dir, suffix=".npz", delete=False) as tmp:
        pending = Path(tmp.name)
    try:
        np.savez(pending, cube_id=ids, z_after_lift=np.stack([beliefs[i] for i in ids]))
        pending.replace(beliefs_path)
    finally:
        pending.unlink(missing_ok=True)

    # A random cycle pairs every cube with one different donor, using each donor once.
    order = np.random.default_rng(seed).permutation(ids)
    donors = dict(zip(order, np.roll(order, 1)))
    result = []
    final_scores = {}
    for cube_id in ids:
        cube_indices = groups[cube_id]
        prove = _prove_rows(test_rows, cube_indices)
        support = _episodes(test_rows, cube_indices, task=1)[5:]
        ceiling = model.fit_z(_episode_rows(test_rows, np.concatenate(support)), steps=5 * fit_steps)
        donor = donors[cube_id]
        truth = _physics(test_rows, cube_indices)
        conditions = [("zero", 0, np.zeros(4, dtype=np.float32))]
        conditions.extend(("own_lifts", n, z) for n, z in enumerate(beliefs[cube_id], 1))
        conditions.extend([("other_lifts", 5, beliefs[donor][-1]), ("slide_ceiling", 0, ceiling)])
        for condition, n, z in conditions:
            score = model.moving_error(prove, z)
            row = dict(seed=int(seed), cube_id=int(cube_id), condition=condition,
                       number_of_lifts=n, slide_error=score, moving_steps=int(prove["cube_moving"].sum()),
                       donor_cube_id=int(donor) if condition == "other_lifts" else "",
                       retrieved_slide_error="", true_mass=truth[0], true_friction=truth[1],
                       predicted_mass="", predicted_friction="")
            if condition == "own_lifts" and n == 5:
                row["predicted_mass"], row["predicted_friction"] = np.append(z, 1.) @ decoder
                final_scores[cube_id] = row
            result.append(row)

    for cube_id, score in retrieve(model, test_rows, beliefs_path).items():
        row = final_scores[cube_id]
        np.testing.assert_allclose(score, row["slide_error"], rtol=1e-6, atol=1e-10)
        row["retrieved_slide_error"] = score
    return result


def write_results(rows, path):
    """Atomically replace the aggregate CSV so interrupted runs cannot truncate it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [(int(r["seed"]), int(r["cube_id"]), r["condition"], int(r["number_of_lifts"])) for r in rows]
    if len(set(keys)) != len(keys) or not keys:
        raise ValueError("Result rows must contain unique (seed, cube, condition, lift count) keys")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
