"""Choose capped shove forces with frozen beliefs and measure actual target landing."""
import csv
from functools import partial
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .collect import physics, start_worker
from .evaluation import write_results
from .model import load_world_model
from .planner import choose_force
from .scripted import shove_actions, simulate_shove


TARGET_DISTANCES_M = (.005, .010, .015, .010, .015)
TARGET_TOLERANCE_M = .0025
CANDIDATE_FORCES_N = np.linspace(1., 4., 13)
FINAL_COUNTS = dict(train_mean=0, lifts_only=5, slides_only=5, both=10, other_both=10, shove_ceiling=5)
TARGET_FIELDS = ("target_landing_error_m", "target_success_rate", "target_planner_failures")


def _save_npz(path, arrays):
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        pending = Path(stream.name)
    try:
        np.savez_compressed(pending, **arrays)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _episodes(data_dir, cfg):
    """Recover exact simulator parameters; expose measured ready states to planning."""
    episodes = {}
    start = cfg["train_cubes"]
    for cube in range(start, start + cfg["test_cubes"]):
        mass, friction = physics(cfg["seed"], cube, cfg)
        with np.load(data_dir / "test" / f"cube_{cube:04}.npz", allow_pickle=False) as saved:
            for episode_id in range(5):
                selected = np.flatnonzero((saved["task"] == 2) & (saved["episode_id"] == episode_id))
                selected = selected[np.argsort(saved["step"][selected])]
                if not np.array_equal(saved["step"][selected], np.arange(cfg["task_steps"][2])):
                    raise ValueError(f"Missing complete prove shove for cube {cube}, episode {episode_id}")
                force_start = int(saved["force_start_step"][selected[0]])
                stop = int(saved["stop_step"][selected[0]])
                if not 0 <= force_start <= stop < len(selected):
                    raise ValueError("Invalid archived shove interval")
                rng = np.random.default_rng([cfg["seed"], cube, 1, 2, episode_id])
                reset_seed = int(rng.integers(2**31))
                angle = rng.uniform(-np.pi, np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
                if not np.all(saved["reset_seed"][selected] == reset_seed):
                    raise ValueError("Archived reset seed differs from the declared protocol")
                np.testing.assert_allclose(saved["motion_direction"][selected],
                                           np.tile(direction, (len(selected), 1)), atol=1e-7, rtol=0)
                episodes[cube, episode_id] = dict(
                    cube_id=cube, episode_id=episode_id, mass=mass, friction=friction,
                    reset_seed=reset_seed, direction=direction, force_start=force_start, stop=stop,
                    initial_state=saved["state"][selected[force_start]].copy(),
                )
    return episodes


def _final_beliefs(path):
    with np.load(path, allow_pickle=False) as bank:
        ids = bank["cube_id"]
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate cube IDs in the stored belief bank")
        both = dict(zip(ids, bank["z_both"][:, -1]))
        result = {}
        for i, cube in enumerate(ids):
            donor = bank["donor_cube_id"][i]
            if donor == cube or donor not in both:
                raise ValueError("Stored donor must be another cube in the bank")
            result[int(cube)] = dict(
                train_mean=bank["prior_mean"].copy(),
                lifts_only=bank["z_lifts_only"][i, -1].copy(),
                slides_only=bank["z_slides_only"][i, -1].copy(),
                both=bank["z_both"][i, -1].copy(),
                other_both=both[donor].copy(),
                shove_ceiling=bank["z_shove_ceiling"][i].copy(),
            )
    return result


def _plan_seed(seed, model, beliefs, episodes, cfg):
    records = []
    for (cube, episode_id), episode in episodes.items():
        horizon = episode["stop"] - episode["force_start"] + 1
        actions = np.stack([shove_actions(episode["direction"], force, horizon, cfg)
                            for force in CANDIDATE_FORCES_N])
        for condition in FINAL_COUNTS:
            # Simulator parameters and recorded outcomes never enter choose_force.
            choice = choose_force(model, episode["initial_state"], episode["direction"],
                                  TARGET_DISTANCES_M[episode_id], CANDIDATE_FORCES_N, actions,
                                  z=beliefs[cube][condition], force_cap_n=cfg["shove_force_cap_n"],
                                  fallback_force_n=2.5)
            records.append(dict(
                seed=seed, cube_id=cube, episode_id=episode_id, condition=condition,
                candidate_index=choice["selected_candidate_index"], force_n=choice["selected_force_n"],
                planner_status=choice["planner_status"], target_xy=choice["target_xy"],
                target_distance_m=TARGET_DISTANCES_M[episode_id], tolerance_m=TARGET_TOLERANCE_M,
                predicted_endpoint_xy=choice["predicted_endpoint_xy"] or [np.nan, np.nan],
                predicted_target_error_m=(np.nan if choice["predicted_target_error_m"] is None
                                          else choice["predicted_target_error_m"]),
                predicted_final_speed_m_s=(np.nan if choice["predicted_final_speed_m_s"] is None
                                           else choice["predicted_final_speed_m_s"]),
                valid_candidates=choice["valid_candidates"],
                candidate_target_errors_m=[np.nan if x is None else x for x in choice["candidate_target_errors_m"]],
                candidate_divergence_steps=[-1 if x is None else x for x in choice["candidate_divergence_steps"]],
            ))
    return {key: np.asarray([record[key] for record in records]) for key in records[0]}


def _simulate_variant(job, *, cfg, cache_dir):
    cube, episode_id, candidate, episode = job
    path = cache_dir / f"cube_{cube:04}_episode_{episode_id}_candidate_{candidate:02}.npz"
    if path.exists():
        return path
    force = float(CANDIDATE_FORCES_N[candidate])
    rows = simulate_shove(episode["mass"], episode["friction"], episode["reset_seed"],
                          episode["direction"], force, cfg)
    start = int(rows["force_start_step"][0])
    stop = int(rows["stop_step"][0])
    if (start, stop) != (episode["force_start"], episode["stop"]):
        raise ValueError(f"Replay shove interval changed for cube {cube}, episode {episode_id}")
    np.testing.assert_allclose(rows["state"][start], episode["initial_state"], rtol=0, atol=1e-5,
                               err_msg=f"Replay ready state changed for cube {cube}, episode {episode_id}")
    expected_actions = shove_actions(episode["direction"], force, stop-start+1, cfg)
    np.testing.assert_allclose(rows["action"][start:stop+1], expected_actions, rtol=0, atol=1e-7)
    if not np.isfinite(rows["next_state"]).all():
        raise ValueError(f"Nonfinite simulator state for cube {cube}, episode {episode_id}")
    rows.update(cube_id=np.full(len(rows["action"]), cube, np.int32),
                episode_id=np.full(len(rows["action"]), episode_id, np.int16),
                candidate_index=np.asarray(candidate), selected_force_n=np.asarray(force))
    _save_npz(path, rows)
    return path


def _execute(choices, episodes, cfg, cache_dir, workers):
    selected = set(zip(choices["cube_id"].tolist(), choices["episode_id"].tolist(),
                       choices["candidate_index"].tolist()))
    jobs = [(cube, ep, candidate, episodes[cube, ep]) for cube, ep, candidate in sorted(selected)]
    worker = partial(_simulate_variant, cfg=cfg, cache_dir=cache_dir)
    if workers == 1:
        return [worker(job) for job in jobs]
    # New processes safely construct CPU simulators after planning has used CUDA.
    with mp.get_context("spawn").Pool(min(workers, len(jobs)), initializer=start_worker) as pool:
        paths = []
        for count, path in enumerate(pool.imap_unordered(worker, jobs), 1):
            paths.append(path)
            if count % 100 == 0 or count == len(jobs):
                print(f"target replay {count}/{len(jobs)} distinct simulator trajectories", flush=True)
        return paths


def _measure(choices, paths):
    endpoints = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            key = (int(saved["cube_id"][0]), int(saved["episode_id"][0]), int(saved["candidate_index"]))
            stop = int(saved["stop_step"][0])
            endpoints[key] = (saved["next_state"][stop, 19:21].astype(np.float64),
                              bool(saved["observed_stopped"][0]))
    actual, stopped = zip(*(endpoints[int(cube), int(ep), int(candidate)]
                           for cube, ep, candidate in zip(choices["cube_id"], choices["episode_id"],
                                                           choices["candidate_index"])))
    choices["actual_endpoint_xy"] = np.stack(actual)
    choices["actual_stopped"] = np.array(stopped)
    delta = choices["actual_endpoint_xy"] - choices["target_xy"]
    choices["actual_target_error_m"] = np.hypot(delta[:, 0], delta[:, 1])
    choices["target_success"] = choices["actual_stopped"] & (choices["actual_target_error_m"] <= TARGET_TOLERANCE_M)
    return choices


def _update_results(rows, trials):
    for row in rows:
        row.update({key: "" for key in TARGET_FIELDS})
        condition = row["condition"]
        if int(row["learning_episodes"]) != FINAL_COUNTS[condition]:
            continue
        selected = ((trials["seed"] == int(row["seed"])) & (trials["cube_id"] == int(row["cube_id"]))
                    & (trials["condition"] == condition))
        if not np.array_equal(np.sort(trials["episode_id"][selected]), np.arange(5)):
            raise ValueError("Target evaluation must contain exactly five prove shoves per final condition")
        row.update(target_landing_error_m=float(trials["actual_target_error_m"][selected].mean()),
                   target_success_rate=float(trials["target_success"][selected].mean()),
                   target_planner_failures=int(np.sum(trials["planner_status"][selected] != "finite_prediction")))
    return rows


def evaluate_targets(data_dir, study_dir, device="cpu", workers=8):
    """Update the existing aggregate results CSV after planning and actual simulation."""
    if workers < 1:
        raise ValueError("workers must be positive")
    data_dir, study_dir = Path(data_dir), Path(study_dir)
    with (study_dir / "results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    seeds = sorted({int(row["seed"]) for row in rows})
    if not seeds:
        raise ValueError("Run held-out prediction evaluation before target control")
    cfg = json.loads((data_dir / "dataset.json").read_text())
    folder = study_dir / "target_control"
    folder.mkdir(exist_ok=True)
    cache_dir = folder / "trajectories"
    cache_dir.mkdir(exist_ok=True)
    files = [data_dir / "dataset.json", *sorted((data_dir / "test").glob("cube_*.npz"))]
    files += [study_dir / f"seed_{seed}" / name for seed in seeds for name in ("model.pt", "beliefs.npz")]
    files += [Path(__file__).with_name(name) for name in ("model.py", "planner.py", "scripted.py", "collect.py", "target_control.py")]
    protocol = dict(
        source_sha256={str(path.resolve()): _sha256(path) for path in files},
        target_distances_m=TARGET_DISTANCES_M, target_tolerance_m=TARGET_TOLERANCE_M,
        candidate_forces_n=CANDIDATE_FORCES_N.tolist(), fallback_force_n=2.5,
        initial_state="measured force-onset state", rollout="all state dimensions without resets",
        target_error="terminal observed XY distance to fixed target, all attempts included",
        target_success="observed rest and XY target error at most tolerance",
        fallback="2.5 N when every candidate becomes nonfinite; retained in error and success means",
    )
    # Canonical JSON also turns tuples into lists before equality checks on resume.
    text = json.dumps(protocol, sort_keys=True, indent=2) + "\n"
    protocol_path = folder / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(text):
        raise ValueError("Target-control cache has a different protocol or data/model fingerprint")
    temporary = protocol_path.with_suffix(".tmp")
    temporary.write_text(text)
    temporary.replace(protocol_path)
    episodes = _episodes(data_dir, cfg)
    plans = []
    for seed in seeds:
        path = folder / f"choices_seed_{seed}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                choices = {key: saved[key] for key in saved.files}
        else:
            model = load_world_model(study_dir / f"seed_{seed}" / "model.pt", device=device)
            beliefs = _final_beliefs(study_dir / f"seed_{seed}" / "beliefs.npz")
            choices = _plan_seed(seed, model, beliefs, episodes, cfg)
            _save_npz(path, choices)
            del model
        plans.append(choices)
        print(f"target planning seed {seed} complete", flush=True)
    choices = {key: np.concatenate([plan[key] for plan in plans]) for key in plans[0]}
    paths = _execute(choices, episodes, cfg, cache_dir, workers)
    trials = _measure(choices, paths)
    _save_npz(folder / "trials.npz", trials)
    rows = _update_results(rows, trials)
    write_results(rows, study_dir / "results.csv")
    return rows
