"""Fixed-budget training and frozen-weight, per-cube latent fitting.

This module imports no old experiment packages. Training never opens episodes;
the dataset loader supplies an explicitly authorized pack. True properties are
requested only for the Oracle. Every optimizer update uses the fixed protocol
budget, with an identical minibatch random stream for every model of a seed.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from clean.models import DynamicsModel, append_history
from clean.protocol import PROTOCOL


ROOT = Path(__file__).resolve().parent


def resolve_config(config: dict, state_dim: int = 48) -> dict:
    """Apply protocol defaults without silently changing the fixed budget."""
    result = {
        "width": PROTOCOL["default_width"],
        "depth": PROTOCOL["default_depth"],
        "z_dim": PROTOCOL["default_z_dim"],
        "lr": PROTOCOL["default_learning_rate"],
        "z_l2": PROTOCOL["default_z_l2"],
        "fit_steps": PROTOCOL["default_fit_steps"],
        "fit_lr": PROTOCOL["default_fit_lr"],
        "moving_weight": PROTOCOL["default_moving_loss_weight"],
        "prediction_horizon": PROTOCOL["default_prediction_horizon"],
        "input_features": list(range(state_dim)),
        "grip_force_max": 30.0,
    }
    unknown = set(config) - set(result)
    if unknown:
        raise ValueError(f"Unknown model configuration keys: {sorted(unknown)}")
    result.update(config)
    if result["input_features"] != list(range(state_dim)):
        raise NotImplementedError("Feature-subset candidates require an explicit full-output model adaptation")
    if result["z_dim"] not in (2, 4, 8, 16):
        raise ValueError("Locked allowed z dimensions are 2, 4, 8, 16")
    if result["prediction_horizon"] not in (1, 2, 3, 4, 5):
        raise ValueError("Prediction-loss horizon must be 1 through 5")
    if min(result["width"], result["depth"], result["fit_steps"]) < 1:
        raise ValueError("Width, depth, and fitting steps must be positive")
    if min(result["lr"], result["fit_lr"], result["moving_weight"]) <= 0 or result["z_l2"] < 0:
        raise ValueError("Learning rates/moving weight must be positive; z penalty nonnegative")
    return result


def parameter_digest(model: torch.nn.Module) -> str:
    """Hash every parameter and buffer, including all training embeddings."""
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(str(value.dtype).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def source_hashes() -> dict[str, str]:
    names = ("train.py", "models.py", "dataset.py", "protocol.py", "schema.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in names if (ROOT / name).exists()}


def _check_pack(pack, *, training: bool):
    if len(pack.x) == 0:
        raise ValueError("Cannot optimize on an empty pack")
    if pack.x.shape != pack.y.shape or len(pack.a) != len(pack.x):
        raise ValueError("Unaligned training state, target, or action arrays")
    if training and bool(((pack.cube_ids < 0) | (pack.cube_ids >= PROTOCOL["train_cubes"])).any()):
        raise ValueError("Weight training is restricted to the 200 training cubes")
    allowed = PROTOCOL["train_episodes"] if training else PROTOCOL["fit_episodes"]
    episode_ok = torch.zeros_like(pack.episode_ids, dtype=torch.bool)
    for episode in allowed:
        episode_ok |= pack.episode_ids == episode
    if not bool(episode_ok.all()):
        raise ValueError("Pack includes episodes outside the authorized training/fitting partition")


def _indices_with_horizon(pack, horizon: int) -> torch.Tensor:
    indices = torch.nonzero(pack.remaining >= horizon, as_tuple=False).flatten()
    if len(indices) == 0:
        raise ValueError("No sequence is long enough for the configured prediction loss")
    return indices


def _prediction_loss(
    model: DynamicsModel,
    pack,
    indices: torch.Tensor,
    config: dict,
    *,
    properties: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recursive 1–5 step loss, with no future measured state as model input."""
    horizon = config["prediction_horizon"]
    state = pack.x[indices]
    cube_ids = pack.cube_ids[indices]
    history = lengths = None
    if model.kind == "history":
        history, lengths = pack.history(indices)
    numerator = state.new_zeros(())
    denominator = state.new_zeros(())
    for ahead in range(horizon):
        rows = indices + ahead
        action = pack.a[rows]
        context = {}
        if model.kind == "two_part":
            context = {"cube_ids": cube_ids} if z is None else {"z": z}
        elif model.kind == "oracle":
            context = {"properties": properties[indices]}
        elif model.kind == "history":
            context = {"history": history, "lengths": lengths}
        state = state + model(state, action, **context)
        error = (state - pack.y[rows]).square().mean(dim=-1)
        weight = 1 + pack.moving[rows].to(error.dtype) * (config["moving_weight"] - 1)
        numerator = numerator + (error * weight).sum()
        denominator = denominator + weight.sum()
        if model.kind == "history" and ahead + 1 < horizon:
            history, lengths = append_history(history, lengths, state, pack.a[rows + 1])
    return numerator / denominator


def train_model(kind: str, pack, normalizer: dict, config: dict, seed: int, output_path):
    """Train exactly 2000 updates of 256 rows, return (model, saved run log).

    The caller must run data checks before invoking this function. Identical
    seeds and candidate horizons yield identical sampled rows across all four
    model kinds. There is no validation evaluation or early stopping here.
    Existing checkpoints are never overwritten.
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    _check_pack(pack, training=True)
    config = resolve_config(config, pack.x.shape[-1])
    torch.set_num_threads(2)
    torch.manual_seed(int(seed))
    if pack.x.device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    rng = torch.Generator(device="cpu").manual_seed(int(seed) + 10000)
    model = DynamicsModel(kind, pack.x.shape[-1], pack.a.shape[-1],
        width=config["width"], depth=config["depth"], z_dim=config["z_dim"],
        num_cubes=PROTOCOL["train_cubes"]).to(pack.x.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    eligible = _indices_with_horizon(pack, config["prediction_horizon"])
    properties = None
    if kind == "oracle":
        from clean.dataset import properties_for
        properties = torch.as_tensor(properties_for(pack.cube_ids, normalizer),
            dtype=pack.x.dtype, device=pack.x.device)
    started = time.monotonic()
    records = []
    updates, batch_size = PROTOCOL["train_updates"], PROTOCOL["batch_size"]
    for update in range(updates):
        choices = torch.randint(len(eligible), (batch_size,), generator=rng).to(pack.x.device)
        indices = eligible[choices]
        optimizer.zero_grad(set_to_none=True)
        prediction_loss = _prediction_loss(model, pack, indices, config, properties=properties)
        z_penalty = prediction_loss.new_zeros(())
        if kind == "two_part":
            z_penalty = config["z_l2"] * model.embedding.weight.square().mean()
        loss = prediction_loss + z_penalty
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite loss for {kind} seed {seed} update {update}")
        loss.backward()
        optimizer.step()
        if update == 0 or (update + 1) % 200 == 0 or update + 1 == updates:
            record = {"update": update + 1, "loss": float(loss.detach()),
                "prediction_loss": float(prediction_loss.detach()),
                "z_penalty": float(z_penalty.detach()),
                "elapsed_seconds": time.monotonic() - started}
            records.append(record)
            print(json.dumps({"stage": "train", "kind": kind, "seed": seed, **record}), flush=True)
    model.eval()
    log = {"kind": kind, "seed": int(seed), "updates": updates, "batch_size": batch_size,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "num_embedding_parameters": model.embedding.weight.numel() if model.embedding is not None else 0,
        "sampled_sequences": updates * batch_size,
        "prediction_horizon": config["prediction_horizon"],
        "eligible_sequences": len(eligible), "training_rows": len(pack.x),
        "training_cube_ids": sorted(set(pack.cube_ids.detach().cpu().tolist())),
        "training_episode_ids": sorted(set(pack.episode_ids.detach().cpu().tolist())),
        "device": str(pack.x.device), "elapsed_seconds": time.monotonic() - started,
        "records": records, "parameter_sha256": parameter_digest(model)}
    checkpoint = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "kind": kind, "state_dim": pack.x.shape[-1], "action_dim": pack.a.shape[-1],
        "config": config, "seed": int(seed), "normalizer": normalizer,
        "budget": {"updates": updates, "batch_size": batch_size},
        "source_sha256": source_hashes(), "log": log}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        torch.save(checkpoint, handle)
    return model, log


def fit_z(model: DynamicsModel, fitpack, config: dict, seed: int) -> dict:
    """Fit one fresh z per cube using only its first five observed episodes.

    Each update draws 64 sequences per cube, so every cube receives the same
    adaptation budget. All weights (including training embeddings) stay frozen.
    Initial z is the mean of the training embeddings. There is no regression to
    training z or property supervision. The return has sorted ``cube_ids`` and
    corresponding ``z`` rows, plus a log proving weights stayed unchanged.
    """
    if model.kind != "two_part":
        raise ValueError("Only two-part models have a fitted object latent")
    _check_pack(fitpack, training=False)
    config = resolve_config(config, fitpack.x.shape[-1])
    torch.set_num_threads(2)
    model.eval().requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    frozen_before = parameter_digest(model)
    cube_ids = torch.unique(fitpack.cube_ids, sorted=True)
    eligible = _indices_with_horizon(fitpack, config["prediction_horizon"])
    per_cube_rows = [eligible[fitpack.cube_ids[eligible] == cube] for cube in cube_ids]
    if any(len(rows) == 0 for rows in per_cube_rows):
        raise ValueError("A cube has no valid fitting sequences")
    counts = torch.tensor([len(rows) for rows in per_cube_rows], device=fitpack.x.device)
    table = torch.zeros((len(cube_ids), int(counts.max())), dtype=torch.long, device=fitpack.x.device)
    for row, indices in enumerate(per_cube_rows):
        table[row, :len(indices)] = indices
    samples_per_cube = 64
    cube_rows = torch.arange(len(cube_ids), device=fitpack.x.device)[:, None]
    latent_rows = cube_rows.expand(-1, samples_per_cube).reshape(-1)
    fitted = torch.nn.Parameter(model.mean_z().expand(len(cube_ids), -1).clone())
    optimizer = torch.optim.Adam([fitted], lr=config["fit_lr"])
    rng = torch.Generator(device="cpu").manual_seed(int(seed) + 20000)
    started = time.monotonic()
    records = []
    for step in range(config["fit_steps"]):
        draws = torch.rand((len(cube_ids), samples_per_cube), generator=rng).to(fitpack.x.device)
        choices = (draws * counts[:, None]).long()
        indices = table[cube_rows, choices].reshape(-1)
        optimizer.zero_grad(set_to_none=True)
        loss = _prediction_loss(model, fitpack, indices, config, z=fitted[latent_rows])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite z-fit loss at step {step}")
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == config["fit_steps"]:
            record = {"step": step + 1, "loss": float(loss.detach()),
                "elapsed_seconds": time.monotonic() - started}
            records.append(record)
            print(json.dumps({"stage": "fit_z", "seed": seed, **record}), flush=True)
    frozen_after = parameter_digest(model)
    if frozen_before != frozen_after:
        raise RuntimeError("Frozen model parameters changed during latent fitting")
    log = {"seed": int(seed), "steps": config["fit_steps"], "samples_per_cube_per_step": samples_per_cube,
        "cube_ids": cube_ids.detach().cpu().tolist(),
        "fit_episode_ids": sorted(set(fitpack.episode_ids.detach().cpu().tolist())),
        "initialization": "mean of training embeddings", "property_supervision": False,
        "parameter_sha256_before": frozen_before, "parameter_sha256_after": frozen_after,
        "all_model_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "records": records, "elapsed_seconds": time.monotonic() - started}
    return {"cube_ids": cube_ids.detach(), "z": fitted.detach(), "log": log}
