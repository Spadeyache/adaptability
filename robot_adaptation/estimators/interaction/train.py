"""Train a small sequence estimator; choose checkpoints on validation only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from . import ARCHITECTURE
from .data import FEATURE_NAMES, OUTCOME_NAMES, sha256, write_json
from .model import InteractionEstimator


class PreparedData:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.episodes = json.loads((self.root / "episodes.json").read_text())
        self.normalization = json.loads((self.root / "normalization.json").read_text())
        n = self.normalization
        with np.load(self.root / "snippets.npz", allow_pickle=False) as z:
            x = ((z["x"] - n["mean"]) / n["std"]).astype(np.float32)
            self.clips = [torch.from_numpy(x[a:b]) for a, b in zip(z["offsets"][:-1], z["offsets"][1:])]
            self.outcomes = torch.tensor((z["outcomes"] - n["outcome_mean"]) / n["outcome_std"], dtype=torch.float32)
        self.groups = {s: defaultdict(list) for s in ("train", "validation", "test")}
        for episode in self.episodes:
            self.groups[episode["split"]][episode["condition_id"]].append(episode)

    def batch(self, clip_ids):
        clips = [self.clips[i] for i in clip_ids]
        return (pad_sequence(clips, batch_first=True), torch.tensor([len(x) for x in clips]),
                self.outcomes[clip_ids])

    def standardized_targets(self, values):
        n = self.normalization
        return (np.log(values) - n["target_log_mean"]) / n["target_log_std"]

    def physical_targets(self, standardized):
        n = self.normalization
        return np.exp(standardized * n["target_log_std"] + n["target_log_mean"])


@torch.inference_mode()
def episode_embeddings(model, data, split):
    model.eval()
    ids = sorted({i for rows in data.groups[split].values() for row in rows for i in row["snippets"]})
    embeddings = {}
    for start in range(0, len(ids), 64):
        block = ids[start:start + 64]
        result = model.encode(*data.batch(block)).numpy()
        embeddings.update(zip(block, result))
    return {row["file"]: (np.mean([embeddings[i] for i in row["snippets"]], axis=0)
                            if row["snippets"] else None)
            for rows in data.groups[split].values() for row in rows}


@torch.inference_mode()
def validation_score(model, data):
    embeddings = episode_embeddings(model, data, "validation")
    errors = []
    scale = np.asarray(data.normalization["prior"])
    for rows in data.groups["validation"].values():
        vectors = [embeddings[row["file"]] for row in rows if embeddings[row["file"]] is not None]
        if not vectors:
            continue
        # Validation uses the predeclared 1- and 5-attempt objectives.
        for count in (1, 5):
            for start in range(0, len(vectors), count):
                evidence = torch.tensor(np.mean(vectors[start:start + count], axis=0), dtype=torch.float32)
                pred = data.physical_targets(model.predict_standard_log(evidence).numpy())
                errors.append(np.abs(pred - rows[0]["target"]) / scale)
    return float(np.mean(errors))


def train(dataset, output, *, epochs=300, seed=101, hidden=48, batch_groups=16,
          attempts=5, steps_per_epoch=8, learning_rate=.002, patience=60, threads=2):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    data = PreparedData(dataset)
    train_targets = np.asarray([r["target"] for rows in data.groups["train"].values() for r in rows])
    data.normalization.update(target_min=train_targets.min(axis=0).tolist(),
                              target_max=train_targets.max(axis=0).tolist(),
                              prior_median=np.median(train_targets, axis=0).tolist())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = InteractionEstimator(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_groups = [rows for rows in data.groups["train"].values() if any(r["snippets"] for r in rows)]
    settings = dict(dataset=str(data.root), dataset_manifest_sha256=sha256(data.root / "manifest.json"),
                    architecture=ARCHITECTURE, epochs=epochs, seed=seed, hidden=hidden,
                    batch_groups=batch_groups, attempts=attempts, steps_per_epoch=steps_per_epoch,
                    learning_rate=learning_rate, patience=patience, threads=threads,
                    objective="MSE of train-standardized log mass/friction at every evidence prefix",
                    validation="mean relative MAE at 1 and 5 attempts; test never used for checkpoint selection",
                    status="running")
    source = output / "source"
    source.mkdir()
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copy2(p, source / p.name)
    settings["source_sha256"] = {p.name: sha256(p) for p in source.glob("*.py")}
    write_json(output / "manifest.json", settings)
    best, best_epoch, history = float("inf"), 0, []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            ids, targets, episode_indices = [], [], []
            for group_index in rng.integers(len(train_groups), size=batch_groups):
                rows = [r for r in train_groups[group_index] if r["snippets"]]
                chosen = rng.choice(len(rows), size=attempts, replace=len(rows) < attempts)
                for j in chosen:
                    episode_index = (episode_indices[-1] + 1) if episode_indices else 0
                    ids.extend(rows[j]["snippets"])
                    episode_indices.extend([episode_index] * len(rows[j]["snippets"]))
                targets.append(rows[0]["target"])
            prediction = model(*data.batch(ids), attempts, torch.tensor(episode_indices))
            target = torch.tensor(data.standardized_targets(np.asarray(targets)), dtype=torch.float32)[:, None, :]
            loss = ((prediction - target) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
        if epoch == 1 or epoch % 5 == 0:
            score = validation_score(model, data)
            row = dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_relative_mae=score,
                       elapsed_s=time.monotonic() - started)
            history.append(row)
            write_json(output / "history.json", history)
            print(json.dumps(row), flush=True)
            if score < best:
                best, best_epoch = score, epoch
                torch.save(dict(architecture=ARCHITECTURE, hidden=hidden, state_dict=model.state_dict(),
                                normalization=data.normalization, alignment=data.manifest["alignment"],
                                feature_names=list(FEATURE_NAMES), outcome_names=list(OUTCOME_NAMES),
                                best_epoch=epoch), output / "best.pt")
            if epoch - best_epoch >= patience:
                break
    settings.update(status="complete", completed_epochs=epoch, best_epoch=best_epoch,
                    best_validation_relative_mae=best, elapsed_s=time.monotonic() - started,
                    checkpoint_sha256=sha256(output / "best.pt"))
    write_json(output / "manifest.json", settings)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    print(train(args.dataset, args.output, epochs=args.epochs, seed=args.seed, threads=args.threads))


if __name__ == "__main__":
    main()
