"""Presealed supervised diagnostics for the final object-property recovery report.

These predictors are diagnostic only: their labels, gradients, and predictions
never enter a dynamics model or fitted object z. Preparation uses training cubes
only. Final scoring reads no motion files and uses only the supplied fit pack.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from clean.audit import EpisodeProbe, PROBE_BATCH, PROBE_HIDDEN, PROBE_LR, r2_per_property
from clean.dataset import load_pack
from clean.protocol import ROOT, PROTOCOL, SEEDS, digest, verify, write_new


PROPERTIES = ("mass_kg", "friction")
RIDGE_LAMBDA = 0.001
NOTES = (
    "The final GRU is fitted on all 200 training cubes, unlike the training-cube "
    "cross-validation probe reported during tuning. It reads one complete first "
    "episode; z is fitted from five lifts, so the information budgets differ. "
    "The GRU is an empirical recovery benchmark, not a mathematical upper bound. "
    "Both diagnostic property decoders are frozen before final held-out evaluation."
)


def _normalizer_fingerprint(normalizer):
    encoded = json.dumps(normalizer, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _labels(cube_ids, normalizer, device):
    """Property supervision exclusively for the explicitly named diagnostics."""
    rows = json.loads((ROOT / "data/cubes.json").read_text())["cubes"]
    by_id = {int(row["cube_id"]): row for row in rows}
    values = torch.tensor(
        [[by_id[int(cube)][name] for name in PROPERTIES] for cube in cube_ids],
        dtype=torch.float32, device=device,
    )
    mean = torch.tensor(normalizer["oracle_property_mean"], dtype=values.dtype, device=device)
    scale = torch.tensor(normalizer["oracle_property_std"], dtype=values.dtype, device=device)
    if not bool(torch.isfinite(values).all()) or not bool((scale > 0).all()):
        raise ValueError("Invalid diagnostic properties or training-only label normalization")
    return (values - mean) / scale


def episode_zero_sequences(pack):
    """Extract only the first complete lift; later fit observations are ignored."""
    if bool(((pack.episode_ids < 0) | (pack.episode_ids > 4)).any()):
        raise ValueError("Recovery input may contain only first-five-lift fit episodes")
    cube_ids = torch.unique(pack.cube_ids, sorted=True)
    sequences = []
    for cube in cube_ids:
        rows = torch.nonzero((pack.cube_ids == cube) & (pack.episode_ids == 0), as_tuple=False).flatten()
        if len(rows) == 0:
            raise ValueError("Each recovery cube must include its first episode")
        if not torch.equal(pack.step[rows], torch.arange(len(rows), device=pack.step.device)):
            raise ValueError("Diagnostic first episode must be complete and in temporal order")
        sequences.append(torch.cat((pack.x[rows], pack.a[rows]), dim=-1))
    if not sequences:
        raise ValueError("Empty recovery input")
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    return cube_ids, pad_sequence(sequences, batch_first=True), lengths


def align_latents(cube_ids, fitted_z_dict):
    latent_ids = torch.as_tensor(fitted_z_dict["cube_ids"], dtype=torch.long).detach().cpu()
    latents = torch.as_tensor(fitted_z_dict["z"]).detach().cpu()
    requested = torch.as_tensor(cube_ids, dtype=torch.long).detach().cpu()
    if latent_ids.ndim != 1 or latents.ndim != 2 or len(latent_ids) != len(latents):
        raise ValueError("Invalid fitted-z identity table")
    if len(set(latent_ids.tolist())) != len(latent_ids):
        raise ValueError("Duplicate cube IDs in fitted-z table")
    if set(latent_ids.tolist()) != set(requested.tolist()):
        raise ValueError("Diagnostic z rows must match exactly the supplied motion cubes")
    lookup = {int(cube): index for index, cube in enumerate(latent_ids.tolist())}
    result = latents[[lookup[int(cube)] for cube in requested.tolist()]].double().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite fitted object z")
    return result


def fit_ridge_decoder(z, normalized_labels):
    """Fixed ridge fit, with train-z scaling and an unpenalized intercept."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(normalized_labels, dtype=np.float64)
    if z.ndim != 2 or y.shape != (len(z), 2) or len(z) < 2:
        raise ValueError("Ridge diagnostic needs aligned cube z and two-property labels")
    if not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError("Ridge diagnostic inputs must be finite")
    mean = z.mean(axis=0)
    scale = np.maximum(z.std(axis=0), 1e-8)
    design = np.column_stack(((z - mean) / scale, np.ones(len(z))))
    regularizer = np.eye(design.shape[1]) * RIDGE_LAMBDA
    regularizer[-1, -1] = 0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    return {"z_mean": torch.from_numpy(mean), "z_std": torch.from_numpy(scale),
            "weights": torch.from_numpy(weights), "lambda": RIDGE_LAMBDA,
            "intercept_penalized": False}


def ridge_predict(z, decoder):
    mean = torch.as_tensor(decoder["z_mean"]).double().cpu().numpy()
    scale = torch.as_tensor(decoder["z_std"]).double().cpu().numpy()
    weights = torch.as_tensor(decoder["weights"]).double().cpu().numpy()
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != len(mean):
        raise ValueError("Ridge input z dimension differs from the frozen decoder")
    return np.column_stack(((z - mean) / scale, np.ones(len(z)))) @ weights


def _preparation_identity(candidate_dir, normalizer):
    return {
        "diagnostic": "final_object_property_recovery",
        "candidate_directory": candidate_dir.name,
        "seeds": list(SEEDS), "training_cube_ids": list(range(200)),
        "gru_episode_ids": [0], "gru_updates": PROTOCOL["probe_updates"],
        "gru_hidden_size": PROBE_HIDDEN, "gru_layers": 1,
        "gru_optimizer": "Adam", "gru_learning_rate": PROBE_LR,
        "gru_batch_cubes": PROBE_BATCH, "ridge_lambda": RIDGE_LAMBDA,
        "normalizer_sha256": _normalizer_fingerprint(normalizer),
        "split_sha256": digest(ROOT / "data/split.json"),
        "properties_sha256": digest(ROOT / "data/cubes.json"),
        "training_z_sha256": {
            str(seed): digest(candidate_dir / f"z_seed_{seed}_train.pt") for seed in SEEDS
        },
        "source_sha256": {
            name: digest(ROOT / name)
            for name in ("recovery.py", "audit.py", "dataset.py", "protocol.py")
        },
        "notes": NOTES,
    }


def prepare_recovery(candidate_dir, normalizer, device):
    """Fit fixed diagnostics on training cubes before the final ledger opens.

    Existing complete seed checkpoints are reused only when their immutable
    preparation metadata match. Checkpoints are created exclusively, never
    overwritten. This function does not load or evaluate any held-out motion.
    """
    if (ROOT / "results/final_evaluations.jsonl").exists():
        raise PermissionError("Recovery diagnostics must be trained before final evaluation begins")
    verify()
    candidate_dir = Path(candidate_dir)
    directory = candidate_dir / "recovery"
    directory.mkdir(exist_ok=True)
    identity = _preparation_identity(candidate_dir, normalizer)
    preparation_path = directory / "preparation.json"
    if preparation_path.exists():
        if json.loads(preparation_path.read_text()) != identity:
            raise RuntimeError("Existing recovery preparation differs; refusing to mix runs")
    else:
        write_new(preparation_path, identity)
    preparation_sha256 = digest(preparation_path)
    paths = {seed: directory / f"seed_{seed}.pt" for seed in SEEDS}
    pending = []
    for seed, path in paths.items():
        if path.exists():
            old = torch.load(path, map_location="cpu", weights_only=True)
            if old["seed"] != seed or old["preparation_sha256"] != preparation_sha256:
                raise RuntimeError("Existing recovery checkpoint has incompatible preparation metadata")
        else:
            pending.append(seed)
    if not pending:
        return None
    pack = load_pack("train", [0], normalizer, device, purpose="probe_train")
    cube_ids, sequences, lengths = episode_zero_sequences(pack)
    if cube_ids.detach().cpu().tolist() != list(range(200)) or pack.group != "train":
        raise ValueError("Final recovery preparation requires exactly the 200 training cubes")
    labels = _labels(cube_ids.detach().cpu().tolist(), normalizer, sequences.device)
    torch.set_num_threads(2)
    for seed in pending:
        torch.manual_seed(seed)
        if sequences.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        probe = EpisodeProbe(sequences.shape[-1]).to(sequences.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
        probe.train()
        started = time.monotonic()
        records = []
        for update in range(PROTOCOL["probe_updates"]):
            sampled = rng.choice(len(cube_ids), size=PROBE_BATCH, replace=False)
            indices = torch.as_tensor(sampled, device=sequences.device)
            prediction = probe(sequences[indices], lengths[sampled])
            loss = torch.square(prediction - labels[indices]).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite final-recovery GRU diagnostic loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.0)
            optimizer.step()
            if update == 0 or (update + 1) % 100 == 0:
                record = {"update": update + 1, "loss": float(loss.detach()),
                          "elapsed_seconds": time.monotonic() - started}
                records.append(record)
                print(json.dumps({"stage": "prepare_recovery", "seed": seed, **record}), flush=True)
        fitted_z = torch.load(candidate_dir / f"z_seed_{seed}_train.pt", map_location="cpu", weights_only=True)
        z = align_latents(cube_ids, fitted_z)
        decoder = fit_ridge_decoder(z, labels.detach().cpu().double().numpy())
        probe.eval().requires_grad_(False)
        checkpoint = {
            "seed": seed, "diagnostic": identity["diagnostic"],
            "preparation_sha256": preparation_sha256,
            "normalizer_sha256": identity["normalizer_sha256"],
            "input_features": sequences.shape[-1],
            "probe_state_dict": {name: value.detach().cpu() for name, value in probe.state_dict().items()},
            "ridge_decoder": decoder,
            "training_log": {"records": records, "optimizer_updates": PROTOCOL["probe_updates"],
                             "training_cubes": 200, "training_episode_ids": [0],
                             "elapsed_seconds": time.monotonic() - started},
            "raw_per_cube_labels_saved": False, "notes": NOTES,
        }
        with paths[seed].open("xb") as handle:
            torch.save(checkpoint, handle)
    return None


def _check_final_context(candidate_dir, fitpack):
    group = getattr(fitpack, "group", None)
    if group not in ("test", "out_of_range") or getattr(fitpack, "purpose", None) != "final_fit":
        raise PermissionError("Property recovery scores are final-fit-pack diagnostics only")
    expected = list(range(200, 230)) if group == "test" else list(range(230, 260))
    if torch.unique(fitpack.cube_ids, sorted=True).detach().cpu().tolist() != expected:
        raise ValueError("Final recovery pack has unexpected cube membership")
    ledger = ROOT / "results/final_evaluations.jsonl"
    if not ledger.exists():
        raise PermissionError("Final evaluation ledger has not begun")
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    begins = [event for event in events if event["event"] == "begin"]
    if len(begins) != 1 or any(event["event"] == "complete" for event in events):
        raise PermissionError("No active unique final evaluation")
    if not begins[0]["candidate"].startswith(candidate_dir.name + "_"):
        raise PermissionError("Recovery diagnostic does not match the sealed final candidate")


@torch.no_grad()
def score_recovery(candidate_dir, seed, fitpack, fitted_z_dict, norm):
    """Score frozen property probes on supplied first-lift motion and fitted z.

    This function does not open any motion dataset or optimize any parameter.
    Labels are loaded solely to calculate final diagnostic R², after checking
    that a unique final evaluation is active for this candidate.
    """
    candidate_dir = Path(candidate_dir)
    if seed not in SEEDS:
        raise ValueError("Recovery seed must be one of the five locked seeds")
    _check_final_context(candidate_dir, fitpack)
    directory = candidate_dir / "recovery"
    checkpoint = torch.load(directory / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
    if checkpoint["preparation_sha256"] != digest(directory / "preparation.json"):
        raise RuntimeError("Recovery preparation metadata changed")
    if checkpoint["normalizer_sha256"] != _normalizer_fingerprint(norm):
        raise RuntimeError("Recovery normalization differs from preparation")
    cube_ids, sequences, lengths = episode_zero_sequences(fitpack)
    if checkpoint["input_features"] != sequences.shape[-1]:
        raise ValueError("Final probe inputs differ from training features")
    probe = EpisodeProbe(sequences.shape[-1]).to(sequences.device)
    probe.load_state_dict(checkpoint["probe_state_dict"])
    probe.eval().requires_grad_(False)
    gru_prediction = probe(sequences, lengths).detach().cpu().double().numpy()
    z = align_latents(cube_ids, fitted_z_dict)
    z_prediction = ridge_predict(z, checkpoint["ridge_decoder"])
    labels = _labels(cube_ids.detach().cpu().tolist(), norm, sequences.device).cpu().double().numpy()
    output = {"seed": int(seed), "cube_ids": cube_ids.detach().cpu().tolist(),
              "gru_input_episode_ids": [0], "z_fit_episode_ids": list(PROTOCOL["fit_episodes"]),
              "diagnostic_checkpoint_sha256": digest(directory / f"seed_{seed}.pt"),
              "notes": NOTES}
    for name, prediction in (("gru", gru_prediction), ("z", z_prediction)):
        scores = r2_per_property(labels, prediction)
        output[name] = {prop: float(score) if np.isfinite(score) else None for prop, score in zip(PROPERTIES, scores)}
    return output
