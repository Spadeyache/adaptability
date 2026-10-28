"""Condition-held-out errors, prior baselines and independent-attempt curves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import sha256, write_json
from .model import InteractionEstimator
from .train import PreparedData, episode_embeddings


def bootstrap_mean_ci(values, *, seed=73518, draws=10000):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = np.asarray([values[rng.integers(len(values), size=len(values))].mean(axis=0) for _ in range(draws)])
    return np.quantile(means, [.025, .975], axis=0).T.tolist()


@torch.inference_mode()
def evaluate(dataset, checkpoint, output=None, *, split="test", seed=202, orders=64, threads=2):
    torch.set_num_threads(threads)
    data = PreparedData(dataset)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = InteractionEstimator(saved["hidden"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    embeddings = episode_embeddings(model, data, split)
    rng = np.random.default_rng(seed)
    train_targets = np.asarray([r["target"] for group in data.groups["train"].values() for r in group])
    median, mean = np.median(train_targets, axis=0), train_targets.mean(axis=0)
    conditions = sorted(data.groups[split])
    counts = (1, 2, 3, 5, 10)
    errors = {k: [] for k in counts}
    shuffled_errors = {k: [] for k in counts}
    prior_errors, mean_errors, records = [], [], []
    all_informative = [value for value in embeddings.values() if value is not None]
    for condition in conditions:
        rows = data.groups[split][condition]
        target = np.asarray(rows[0]["target"])
        prior_errors.append(np.abs(median - target))
        mean_errors.append(np.abs(mean - target))
        condition_errors = {k: [] for k in counts}
        shuffled_condition = {k: [] for k in counts}
        for _ in range(orders):
            permutation = rng.permutation(len(rows))
            evidence, accumulated = np.zeros(model.hidden, dtype=np.float32), 0
            for k in range(1, max(counts) + 1):
                row = rows[permutation[(k - 1) % len(rows)]]
                if embeddings[row["file"]] is not None:
                    evidence += embeddings[row["file"]]
                    accumulated += 1
                if k in counts:
                    pred = (data.physical_targets(model.predict_standard_log(torch.tensor(evidence / accumulated)).numpy())
                            if accumulated else mean)
                    condition_errors[k].append(np.abs(pred - target))
                    wrong = np.mean([all_informative[i] for i in rng.integers(len(all_informative), size=k)], axis=0)
                    wrong_pred = data.physical_targets(model.predict_standard_log(torch.tensor(wrong)).numpy())
                    shuffled_condition[k].append(np.abs(wrong_pred - target))
        entry = dict(condition_id=condition, target=target.tolist(), episodes=len(rows),
                     informative_episodes=sum(embeddings[row["file"]] is not None for row in rows), attempts={})
        for k in counts:
            error = np.mean(condition_errors[k], axis=0)
            errors[k].append(error)
            shuffled_errors[k].append(np.mean(shuffled_condition[k], axis=0))
            entry["attempts"][k] = error.tolist()
        records.append(entry)
    prior_errors = np.asarray(prior_errors)
    metrics = {}
    for k in counts:
        values = np.asarray(errors[k])
        improvement = prior_errors - values
        metrics[k] = dict(mae=values.mean(axis=0).tolist(), mae_condition_bootstrap_95ci=bootstrap_mean_ci(values),
                          improvement_over_median_prior=improvement.mean(axis=0).tolist(),
                          improvement_condition_bootstrap_95ci=bootstrap_mean_ci(improvement),
                          shuffled_evidence_mae=np.mean(shuffled_errors[k], axis=0).tolist(),
                          relative_error_reduction_vs_median_prior=(1 - values.mean(axis=0) / prior_errors.mean(axis=0)).tolist())
        more_improvement = np.asarray(errors[1]) - values
        metrics[k]["improvement_over_one_attempt"] = more_improvement.mean(axis=0).tolist()
        metrics[k]["improvement_over_one_attempt_condition_bootstrap_95ci"] = bootstrap_mean_ci(more_improvement)
    report = dict(split=split, held_conditions=len(conditions), units=["mass_kg", "object_friction_coefficient"],
                  checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256(checkpoint),
                  dataset=str(data.root), seed=seed, orderings_per_condition=orders,
                  bootstrap=dict(draws=10000, seed=73518, unit="physical condition"),
                  prior_median=median.tolist(), prior_mean=mean.tolist(),
                  prior_median_mae=prior_errors.mean(axis=0).tolist(), prior_mean_mae=np.mean(mean_errors, axis=0).tolist(),
                  attempts=metrics, conditions=records,
                  interpretation="Independent fixed-policy attempts pooled under identical physical conditions, not online adaptive retries",
                  checkpoint_selection="Validation-only; this report does not select training duration",
                  test_target_range=dict(min=np.min([r["target"] for r in records], axis=0).tolist(),
                                         max=np.max([r["target"] for r in records], axis=0).tolist()),
                  training_target_range=dict(min=train_targets.min(axis=0).tolist(), max=train_targets.max(axis=0).tolist()))
    if output:
        write_json(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("test", "validation", "train"), default="test")
    args = p.parse_args()
    print(json.dumps(evaluate(args.dataset, args.checkpoint, args.output, split=args.split), indent=2))


if __name__ == "__main__":
    main()
