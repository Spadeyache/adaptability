"""Plot transfer, belief formation and held-out linear decoding from one results CSV."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _seed_summary(rows, condition, n):
    per_seed = {}
    for row in rows:
        if row["condition"] == condition and int(row["number_of_lifts"]) == n:
            per_seed.setdefault(row["seed"], []).append(float(row["slide_error"]))
    if not per_seed:
        raise ValueError(f"Missing condition {condition} at lift count {n}")
    means = np.array([np.mean(group) for group in per_seed.values()])
    sem = means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else np.nan
    return means.mean(), sem


def _style(ax):
    ax.grid(axis="y", color="#d7dfe3", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)


def _save(fig, path):
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_results(results_path, output_dir):
    with Path(results_path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    seeds = len({r["seed"] for r in rows})
    note = (f"Mean over cubes within each of {seeds} seeds. Error bars show ±1 seed SEM."
            if seeds > 1 else "Mean over cubes in one seed.")
    ylabel = "Moving-step slide error (normalized MSE)"
    transfer = [("zero", 0), ("own_lifts", 5), ("other_lifts", 5), ("slide_ceiling", 0)]
    means, errors = np.array([_seed_summary(rows, name, n) for name, n in transfer]).T
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    ax.bar(range(4), means, color=["#8c969a", "#227a79", "#9d7952", "#4d6890"], width=.64)
    if seeds > 1:
        ax.errorbar(range(4), means, yerr=errors, fmt="none", color="#263238", capsize=4)
    ax.set(title="Lift-to-slide transfer", ylabel=ylabel, xticks=range(4),
           xticklabels=["Zero z", "Own lifts", "Another cube’s lifts", "Slide ceiling"], ylim=(0, None))
    _style(ax)
    fig.supxlabel(note, fontsize=9)
    p1 = _save(fig, dest / "transfer.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    counts = np.arange(6)
    means, errors = np.array([_seed_summary(rows, "own_lifts" if n else "zero", n) for n in counts]).T
    ax.errorbar(counts, means, yerr=errors if seeds > 1 else None, marker="o",
                color="#227a79", linewidth=2, capsize=4)
    ax.set(title="Belief formation across lifts", xlabel="Lifts observed", xticks=counts, ylabel=ylabel)
    _style(ax)
    fig.supxlabel(note, fontsize=9)
    p2 = _save(fig, dest / "formation.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.7), layout="constrained")
    own = [r for r in rows if r["condition"] == "own_lifts" and int(r["number_of_lifts"]) == 5]
    for ax, key, label in zip(axes, ["mass", "friction"], ["mass (kg)", "friction coefficient"]):
        actual = np.array([float(r["true_" + key]) for r in own])
        predicted = np.array([float(r["predicted_" + key]) for r in own])
        low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        ax.plot([low, high], [low, high], linestyle="--", linewidth=1, color="#718086")
        ax.scatter(actual, predicted, s=18, alpha=.6, color="#227a79", edgecolors="none")
        # Identical coordinate limits keep the identity line at 45 degrees.
        ax.set(xlabel=f"True {label}", ylabel=f"Predicted {label}", aspect="equal")
        pad = (high - low) * .05 or .01
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        _style(ax)
    fig.suptitle("What five-lift beliefs encode")
    fig.supxlabel("Linear decoders fitted on training cubes. Each point represents a held-out cube and seed.",
                  fontsize=9)
    p3 = _save(fig, dest / "meaning.png")
    return p1, p2, p3
