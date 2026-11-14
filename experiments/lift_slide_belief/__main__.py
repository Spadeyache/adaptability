"""Train shared dynamics, infer beliefs from lifts and slides, then predict capped shoves."""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import TrainConfig, fit_world_model, load_world_model


def write_json(path, value):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def training_digest(rows):
    digest = hashlib.sha256()
    for key in ("cube_id", "task", "episode_id", "step", "state", "action", "next_state", "cube_moving"):
        value = np.ascontiguousarray(rows[key])
        digest.update(f"{key} {value.shape} {value.dtype}\n".encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def run(args):
    from .collect import load_rows
    from .evaluation import evaluate_seed, write_results
    from .plotting import plot_results

    output = args.output
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.threads)
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    train = load_rows(args.data, "train")
    tasks = np.unique(train["task"])
    if not np.array_equal(tasks, [0, 1, 2]):
        raise ValueError("The current study requires lift, slow-slide and shove data")
    config = TrainConfig(max_epochs=args.max_epochs, patience=args.patience)
    protocol = dict(data=str(args.data.resolve()), training_sha256=training_digest(train),
                    source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ("model.py", "evaluation.py", "plotting.py", "__main__.py")},
                    dataset_manifest_sha256=hashlib.sha256((args.data / "dataset.json").read_bytes()).hexdigest(),
                    optimizer_seeds=args.seeds, training=asdict(config), fit_steps=args.fit_steps,
                    device=device, torch_version=str(torch.__version__))
    manifest = output / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError(f"{output} already contains a different protocol. Choose a new output directory.")
    write_json(manifest, protocol)
    gates = {}
    for seed in args.seeds:
        folder = output / f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        checkpoint = folder / "model.pt"
        if checkpoint.exists():
            model = load_world_model(checkpoint, device=device)
        else:
            def log(record):
                if record["epoch"] == 1 or record["epoch"] % 10 == 0:
                    print(f"seed {seed}  epoch {record['epoch']}  validation MSE {record['validation_mse']:.5f}", flush=True)

            model = fit_world_model(train, seed=seed, device=device, config=config, log=log)
            model.save(checkpoint)
        write_json(folder / "validation.json", model.validation)
        gates[str(seed)] = {key: value for key, value in model.validation.items()
                            if key not in ("history", "heldout_episodes", "cube_diagnostics")}
        print(f"seed {seed}  best epoch {model.validation['best_epoch']}  "
              f"validation gate {'passed' if model.validation['passed'] else 'failed'}", flush=True)
        del model

    passed_seeds = [seed for seed in args.seeds if gates[str(seed)]["passed"]]
    write_json(output / "validation.json", {"passed": len(passed_seeds) == len(args.seeds),
                                             "passing_seeds": passed_seeds, "seeds": gates})
    if not passed_seeds:
        print("No seed passed validation. Held-out cube evaluation stops here.", flush=True)
        return 2

    test = load_rows(args.data, "test")
    if np.intersect1d(train["cube_id"], test["cube_id"]).size:
        raise ValueError("Train and test cube IDs overlap")
    results = []
    for seed in passed_seeds:
        folder = output / f"seed_{seed}"
        model = load_world_model(folder / "model.pt", device=device)
        results.extend(evaluate_seed(model, train, test, seed=seed,
                                     output_dir=folder, fit_steps=args.fit_steps))
        print(f"seed {seed}  completed held-out shove evaluation and retrieval", flush=True)
        del model
    results_path = output / "results.csv"
    write_results(results, results_path)
    plot_results(results_path, output)
    print(f"Wrote {len(results)} results to {results_path}", flush=True)
    return 0 if len(passed_seeds) == len(args.seeds) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="directory created by the collection command")
    parser.add_argument("output", type=Path, help="study outputs and resumable per-seed checkpoints")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--fit-steps", type=int, default=100)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    if min(args.fit_steps, args.max_epochs, args.patience, args.threads) < 1:
        parser.error("fit steps, epochs, patience and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
