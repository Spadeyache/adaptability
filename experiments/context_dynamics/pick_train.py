"""Fit the matched five-step PickCube world models without consulting the holdout."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import torch

from robot_adaptation.estimators.context_dynamics.model import ContextDynamics
from .data import Trajectories


SETTINGS = dict(latent=8, width=512, horizon=5, updates=12000, batch=384, lr=.0005,
                seed=91109, swap_seed=110961, bootstrap_seed=110962, bootstrap_draws=3000,
                examples_per_source=2, interpolation_intervals=10, low_probe_r2=.20)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_torch(path, contents):
    temporary = path.with_suffix(".tmp")
    torch.save(contents, temporary)
    temporary.replace(path)


def train(folder, run, device):
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fit(folder, run, device)


def fit(folder, run, device):
    info = json.loads((folder / "complete.json").read_text())
    if not info["verified"] or (info["train_episodes"], info["validation_episodes"]) != (36000, 4000):
        raise ValueError("Expected the verified 36,000 / 4,000 PickCube split")
    hashes = json.loads((folder / "checksums.json").read_text())
    for name, expected in hashes.items():
        if digest(folder / name) != expected:
            raise ValueError(f"Dataset checksum mismatch in {name}")
    cfg = SETTINGS
    sources = [Path(__file__), Path(__file__).with_name("data.py"),
               Path("robot_adaptation/estimators/context_dynamics/model.py")]
    protocol = dict(**cfg, dataset=str(folder.resolve()), device=device,
                    gpu=torch.cuda.get_device_name() if device.startswith("cuda") else None,
                    pytorch=str(torch.__version__), checksums_sha256=digest(folder / "checksums.json"),
                    source_sha256={p.name: digest(p) for p in sources},
                    loss="Mean of five free-running horizon MSEs in training one-step delta-SD units",
                    checkpoint="Fixed final update, zero held-out episodes for checkpoint selection")
    settings = run / "settings.json"
    if settings.exists() and json.loads(settings.read_text()) != protocol:
        raise ValueError("Existing study has a different data, source or training specification")
    if (run / "training_complete.json").exists():
        print("Both fixed-budget models are already fitted.", flush=True)
        return
    settings.write_text(json.dumps(protocol, indent=2) + "\n")
    source_dir = run / "source"
    source_dir.mkdir(exist_ok=True)
    for path in sources:
        shutil.copyfile(path, source_dir / path.name)

    torch.manual_seed(cfg["seed"])
    torch.set_float32_matmul_precision("high")
    data = Trajectories(folder, device)
    model = ContextDynamics(cfg["latent"], cfg["width"]).to(device)
    data.fit_scaling(model)
    data.states = model.normalize(data.states)
    baseline = copy.deepcopy(model)
    # Preserve all shared initial weights and make the baseline independent of any z,
    # including when loaded with the repository's ordinary checkpoint reader.
    with torch.no_grad():
        for parameter in baseline.encoder.parameters():
            parameter.zero_()
        baseline.dynamics[0].weight[:, -cfg["latent"]:].zero_()
    baseline.encoder.requires_grad_(False)
    models = {"W": model, "W_noz": baseline}
    optimizers = {name: torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                    lr=cfg["lr"], weight_decay=1e-5, fused=device.startswith("cuda"))
                  for name, net in models.items()}
    rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 1)
    checkpoint = run / "training_state.pt"
    history, start = [], 1
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        for name, net in models.items():
            net.load_state_dict(saved["models"][name])
            optimizers[name].load_state_dict(saved["optimizers"][name])
        rng.set_state(saved["rng"].cpu())
        history, start = saved["history"], saved["update"] + 1
        del saved
    clock = time.monotonic()
    previous_time = history[-1]["elapsed_s"] if history else 0.
    zeros = data.states.new_zeros((cfg["batch"], cfg["latent"]))
    for step in range(start, cfg["updates"] + 1):
        ids = data.train_ids[torch.randint(len(data.train_ids), (cfg["batch"],), generator=rng, device=device)]
        starts = torch.randint(data.horizon - cfg["horizon"] + 1, (len(ids),), generator=rng, device=device)
        states, actions = data.windows(ids, starts, cfg["horizon"])
        record = step == 1 or step % 1000 == 0 or step == cfg["updates"]
        row = {"update": step}
        fraction = (step - 1) / max(1, cfg["updates"] - 1)
        lr = cfg["lr"] * (.12 + .88 * (1 + math.cos(math.pi * fraction)) / 2)
        for name, net in models.items():
            optimizer = optimizers[name]
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            z = model.encode(data.constants[ids]) if name == "W" else zeros
            loss = net.error_by_horizon(states, actions, z).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if record:
                row[name] = loss.item()
        if record:
            row["elapsed_s"] = previous_time + time.monotonic() - clock
            history.append(row)
            save_torch(checkpoint, dict(update=step, history=history, rng=rng.get_state(),
                       models={n: net.state_dict() for n, net in models.items()},
                       optimizers={n: opt.state_dict() for n, opt in optimizers.items()}))
            (run / "training.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
            print(json.dumps(row), flush=True)

    for name, net in models.items():
        path = run / name
        path.mkdir(exist_ok=True)
        save_torch(path / "model.pt", dict(model=net.state_dict(), latent=cfg["latent"], width=cfg["width"],
                   horizon=cfg["horizon"], update=cfg["updates"], zero_context=name == "W_noz"))
    completion = {name: digest(run / name / "model.pt") for name in models}
    (run / "training_complete.json").write_text(json.dumps(completion, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(args.data, args.run, args.device)
