"""Locked moving-step metrics and recursively predicted ten-step rollouts.

This module never opens a dataset or chooses a model. The caller supplies an
authorized pack. Only the explicitly named Oracle reads object properties.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from clean.models import append_history


METRICS = ("one_step_nmse", "r2", "rollout_error")


def derangement(seed: int, n: int) -> np.ndarray:
    """A reproducible permutation with no cube receiving its own latent."""
    if n < 2:
        raise ValueError("shuffled-between-cubes z requires at least two cubes")
    rng = np.random.default_rng(seed)
    identity = np.arange(n)
    while True:
        permutation = rng.permutation(n)
        if np.all(permutation != identity):
            return permutation


def _condition(model, pack, indices, normalizer, z_by_cube):
    """Construct model-specific inputs without exposing labels to other models."""
    kind = model.kind
    if kind == "oracle":
        # Lazy import also makes the property read explicit and auditable.
        from clean.dataset import properties_for
        return {"properties": properties_for(pack.cube_ids[indices], normalizer)}
    if kind == "two_part":
        ids = pack.cube_ids[indices]
        if z_by_cube is not None:
            return {"z": z_by_cube[ids]}
        if bool((ids >= 200).any()):
            raise ValueError("held-out cubes require explicit fitted/controlled z")
        return {"cube_ids": ids}
    if kind == "history":
        history, lengths = pack.history(indices)
        return {"history": history, "lengths": lengths}
    if kind == "plain":
        return {}
    raise ValueError(f"unknown model kind: {kind}")


@torch.no_grad()
def evaluate(model, pack, normalizer, z_by_cube=None, batch_size: int = 1024):
    """Evaluate the immutable metrics on the supplied scoring pack.

    One-step NMSE gives every cube equal weight after averaging its moving
    transitions and all state coordinates. R² pools all moving transitions,
    using a separate scoring-set mean for each normalized next-state column.
    Rollouts start at moving transitions with ten remaining actions, predict
    all ten states recursively, and average errors across times/coordinates
    before averaging starts within cube and cubes equally. Future observed
    states are accessed solely to score predictions.

    Nonfinite/undefined values stay nonfinite here and are explicitly listed.
    ``json_safe`` converts them to JSON null while recording their paths.
    A cube without eligible data makes its aggregate metric undefined; it is
    never silently omitted from the equal-weight cube average.
    """
    if batch_size < 1 or pack.x.ndim != 2 or len(pack.x) == 0:
        raise ValueError("evaluation requires a nonempty pack and positive batch size")
    if pack.y.shape != pack.x.shape or len(pack.a) != len(pack.x):
        raise ValueError("state, next state, and action rows must align")
    device = pack.x.device
    ids = pack.cube_ids.to(device=device, dtype=torch.long)
    if bool((ids < 0).any()):
        raise ValueError("cube IDs must be nonnegative")
    cubes = torch.unique(ids, sorted=True)
    size, features = int(cubes[-1]) + 1, pack.x.shape[1]
    counts = torch.zeros(size, device=device, dtype=torch.float64)
    residual = torch.zeros_like(counts)
    target_sum = torch.zeros((size, features), device=device, dtype=torch.float64)
    target_square_sum = torch.zeros_like(target_sum)
    rollout_counts = torch.zeros_like(counts)
    rollout_error_sum = torch.zeros_like(counts)
    moving = torch.nonzero(pack.moving, as_tuple=False).flatten()
    starts = torch.nonzero(pack.moving & (pack.remaining >= 10), as_tuple=False).flatten()
    if len(starts):
        ends = starts + 9
        if bool((ends >= len(pack.x)).any()):
            raise ValueError("remaining-step metadata extends beyond the pack")
        if bool(((ids[starts] != ids[ends]) |
                 (pack.episode_ids[starts] != pack.episode_ids[ends]) |
                 (pack.step[ends] != pack.step[starts] + 9)).any()):
            raise ValueError("rollout would cross an episode boundary")
    was_training = model.training
    model.eval()
    try:
        for indices in moving.split(batch_size):
            conditions = _condition(model, pack, indices, normalizer, z_by_cube)
            prediction = pack.x[indices] + model(pack.x[indices], pack.a[indices], **conditions)
            truth = pack.y[indices].double()
            squared = (prediction.double() - truth).square()
            cube = ids[indices]
            counts.index_add_(0, cube, torch.ones(len(indices), device=device, dtype=torch.float64))
            residual.index_add_(0, cube, squared.sum(dim=1))
            target_sum.index_add_(0, cube, truth)
            target_square_sum.index_add_(0, cube, truth.square())

        for indices in starts.split(batch_size):
            conditions = _condition(model, pack, indices, normalizer, z_by_cube)
            predicted_state = pack.x[indices]
            start_error = torch.zeros(len(indices), device=device, dtype=torch.float64)
            for offset in range(10):
                action = pack.a[indices + offset]
                if offset and model.kind == "history":
                    history, lengths = append_history(
                        conditions["history"], conditions["lengths"], predicted_state, action)
                    conditions = {"history": history, "lengths": lengths}
                predicted_state = predicted_state + model(predicted_state, action, **conditions)
                # Ground truth affects this score only, never the model input.
                difference = predicted_state.double() - pack.y[indices + offset].double()
                start_error += difference.square().mean(dim=1) / 10
            cube = ids[indices]
            rollout_counts.index_add_(0, cube, torch.ones(len(indices), device=device, dtype=torch.float64))
            rollout_error_sum.index_add_(0, cube, start_error)
    finally:
        model.train(was_training)

    per_cube = {}
    for cube in cubes.tolist():
        count, rollout_count = int(counts[cube]), int(rollout_counts[cube])
        sse = float(residual[cube])
        total_variance = float((target_square_sum[cube] - target_sum[cube].square() / counts[cube]).sum()) if count else 0.0
        per_cube[str(cube)] = {
            "one_step_nmse": sse / (count * features) if count else math.nan,
            "r2": 1 - sse / total_variance if total_variance > 0 else math.nan,
            "rollout_error": float(rollout_error_sum[cube]) / rollout_count if rollout_count else math.nan,
            "moving_steps": count, "rollout_starts": rollout_count,
        }
    total_count = float(counts.sum())
    denominator = float((target_square_sum.sum(dim=0) - target_sum.sum(dim=0).square() / total_count).sum()) if total_count else 0.0
    result = {
        "one_step_nmse": float(np.mean([row["one_step_nmse"] for row in per_cube.values()])),
        "r2": 1 - float(residual.sum()) / denominator if denominator > 0 else math.nan,
        "rollout_error": float(np.mean([row["rollout_error"] for row in per_cube.values()])),
        "per_cube": per_cube,
        "moving_steps": int(total_count), "rollout_starts": int(rollout_counts.sum()),
        "missing_moving_cubes": [int(c) for c, row in per_cube.items() if not row["moving_steps"]],
        "missing_rollout_cubes": [int(c) for c, row in per_cube.items() if not row["rollout_starts"]],
    }
    result["nonfinite_metrics"] = [name for name in METRICS if not math.isfinite(result[name])]
    return result


def aggregate_seed_metrics(seed_results: Sequence[Mapping]):
    """Aggregate exactly five independently seeded saved-run metric dicts."""
    if len(seed_results) != 5:
        raise ValueError("the locked protocol requires exactly five seed results")
    result = {}
    for name in METRICS:
        values = [float(run[name]) if run[name] is not None else math.nan for run in seed_results]
        array = np.asarray(values, dtype=np.float64)
        # Undefined seeds remain undefined rather than being dropped.
        mean = float(array.mean()) if np.isfinite(array).all() else math.nan
        std = float(array.std(ddof=1)) if np.isfinite(array).all() else math.nan
        result[name] = {"mean": mean, "std": std, "values": values}
    result["nonfinite_metrics"] = [name for name in METRICS if not math.isfinite(result[name]["mean"])]
    return result


def success_criteria(plain, history, oracle, fitted, shuffled, failure_checks=None):
    """Locked decisions depend exclusively on aggregate one-step NMSE."""
    p, h, o, f, s = [run["one_step_nmse"] for run in (plain, history, oracle, fitted, shuffled)]
    values = [p["mean"], h["mean"], o["mean"], f["mean"], f["std"], s["mean"], s["std"]]
    finite = all(value is not None and math.isfinite(value) for value in values)
    gap = p["mean"] - o["mean"] if finite else math.nan
    recovery = (p["mean"] - f["mean"]) / gap if finite and gap > 0 else math.nan
    if failure_checks is None:
        checks_pass = False
    else:
        checks_pass = all(failure_checks.get(f"F{i}") is True for i in range(1, 10))
    return {
        "S1": bool(finite and gap > 0 and recovery >= .5),
        "S2": bool(finite and f["mean"] + f["std"] < s["mean"] - s["std"]),
        "S3": bool(finite and f["mean"] <= 1.1 * h["mean"]),
        "S4": checks_pass,
        "plain_oracle_gap": gap, "gap_recovery": recovery,
    }


def json_safe(value):
    """Convert nonfinite numbers to JSON null and explicitly record all paths.

    The returned top-level mapping preserves mapping keys and adds
    ``nonfinite_paths``. For a non-mapping input, the converted value is under
    ``value``. This never turns an invalid scientific result into a valid one.
    """
    paths = []
    def convert(item, path):
        if isinstance(item, Mapping):
            return {str(k): convert(v, f"{path}.{k}" if path else str(k)) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v, f"{path}[{i}]") for i, v in enumerate(item)]
        if isinstance(item, np.ndarray):
            return convert(item.tolist(), path)
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            paths.append(path)
            return None
        if isinstance(item, np.generic):
            return item.item()
        return item
    converted = convert(value, "")
    result = converted if isinstance(converted, dict) else {"value": converted}
    # Re-serializing a saved result must preserve the explanation for its nulls.
    previous = result.get("nonfinite_paths", [])
    result["nonfinite_paths"] = list(dict.fromkeys([*previous, *paths]))
    return result
