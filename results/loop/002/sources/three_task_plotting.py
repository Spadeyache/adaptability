"""Plot shove transfer, belief formation and train-fitted linear decoding."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LEARNING = (("lifts_only", 5, "Lifts", "#227a79"), ("slides_only", 5, "Slow slides", "#b7783d"),
            ("both", 10, "Both", "#525da8"))
METRICS = (("shove_error", "Moving-step shove MSE (normalized)"),
           ("endpoint_error_m", "Shove endpoint XY error (m)"))


def _seed_summary(rows, condition, n, metric):
    selected = [row for row in rows if row["condition"] == condition and int(row["learning_episodes"]) == n]
    if not selected:
        raise ValueError(f"Missing condition {condition} at episode count {n}")
    per_seed = {}
    valid = 0
    if metric == "shove_error":
        availability = {}
        for row in rows:
            key = (row["seed"], row["cube_id"])
            present = row[metric] != ""
            if key in availability and availability[key] != present:
                raise ValueError("Moving-step score availability differs across conditions for the same cube")
            availability[key] = present
    for row in selected:
        if row[metric] != "":
            value = float(row[metric])
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite {metric} in results")
            per_seed.setdefault(row["seed"], []).append(value)
            valid += 1
    means = np.array([np.mean(group) for group in per_seed.values()])
    if valid != len(selected) and metric != "shove_error":
        return np.nan, np.nan, valid, len(selected)
    mean = means.mean() if len(means) else np.nan
    sem = means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else np.nan
    return mean, sem, valid, len(selected)


def _style(ax):
    ax.grid(axis="y", color="#d7dfe3", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)


def _save(fig, path):
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_transfer(rows, dest, note):
    conditions = [("train_mean", 0), ("lifts_only", 5), ("slides_only", 5),
                  ("both", 10), ("other_both", 10), ("shove_ceiling", 5)]
    labels = ["Train mean", "5 lifts", "5 slides", "5 lifts +\n5 slides", "Another cube’s\nboth", "5-shove\nceiling"]
    colors = ["#8c969a", "#227a79", "#b7783d", "#525da8", "#9d7952", "#4d6890"]
    has_targets = any(row.get("target_landing_error_m", "") != "" for row in rows)
    metrics = list(METRICS)
    if has_targets:
        metrics.append(("target_landing_error_m", "Actual target landing error (m)"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.8), layout="constrained")
    missing = False
    for ax, (metric, ylabel) in zip(axes, metrics):
        stats = [_seed_summary(rows, name, n, metric) for name, n in conditions]
        means, errors, valid, total = np.array(stats).T
        ax.bar(range(6), means, color=colors, width=.65)
        finite = np.isfinite(errors)
        ax.errorbar(np.arange(6)[finite], means[finite], yerr=errors[finite],
                    fmt="none", color="#263238", capsize=3)
        ticklabels = list(labels)
        if np.any(valid < total):
            missing = True
            suffix = "moving" if metric == "shove_error" else "invalid"
            counts = valid if metric == "shove_error" else total - valid
            ticklabels = [f"{label}\n{int(v)}/{int(t)} {suffix}" for label, v, t in zip(labels, counts, total)]
        ax.set(ylabel=ylabel, xticks=range(6), xticklabels=ticklabels, ylim=(0, None))
        if not np.isfinite(means).any():
            ax.set_yticks([])
            message = "No moving-step scores" if metric == "shove_error" else "Endpoint means undefined"
            ax.text(.5, .6, message + "\nSee fractions below",
                    transform=ax.transAxes, ha="center", va="center", color="#52616b")
        ax.tick_params(axis="x", labelsize=8)
        _style(ax)
    fig.suptitle("Transfer to held-out shoves")
    footnote = note
    if missing:
        footnote += ("\nMoving-step means use the same moving cubes in every condition. "
                     "Endpoint means require five valid endpoints for every cube/seed.")
    if has_targets:
        footnote += "\nTarget error comes from simulator executions of model-selected forces. All attempts and declared fallbacks are included."
    fig.supxlabel(footnote, fontsize=8)
    return _save(fig, dest / "transfer.png")


def _plot_formation(rows, dest, note):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    missing = False
    for ax, (metric, ylabel) in zip(axes, METRICS):
        has_estimate = False
        for line_index, (condition, count, label, color) in enumerate(LEARNING):
            stats = [_seed_summary(rows, "train_mean" if n == 0 else condition, n, metric)
                     for n in range(count + 1)]
            means, errors, valid, total = np.array(stats).T
            has_estimate |= np.isfinite(means).any()
            x = np.arange(count + 1)
            ax.plot(x, means, marker="o", label=label, color=color, linewidth=1.8)
            finite = np.isfinite(errors)
            ax.errorbar(x[finite], means[finite], yerr=errors[finite], fmt="none", color=color, capsize=3)
            for n in np.flatnonzero(valid < total):
                missing = True
                count = valid[n] if metric == "shove_error" else total[n] - valid[n]
                ax.annotate(f"{int(count)}/{int(total[n])}", (n, .03 + .06 * line_index),
                            xycoords=("data", "axes fraction"), ha="center", fontsize=6, color=color)
        ax.set(xlabel="Learning episodes observed", xticks=np.arange(11), ylabel=ylabel)
        if not has_estimate:
            ax.set_yticks([])
            message = "No moving-step scores" if metric == "shove_error" else "Endpoint means undefined"
            ax.text(.5, .6, message + "\nSee fractions below",
                    transform=ax.transAxes, ha="center", va="center", color="#52616b")
        _style(ax)
    axes[0].legend(frameon=False)
    fig.suptitle("Belief formation")
    footnote = "Both alternates lift, slide, lift, slide. Every prefix uses the same fitting budget.\n" + note
    if missing:
        footnote += ("\nMoving-step means use matched moving cubes; labels give moving/total. "
                     "Incomplete endpoint means are omitted; labels give invalid/total.")
    fig.supxlabel(footnote, fontsize=8)
    return _save(fig, dest / "formation.png")


def _plot_meaning(rows, dest):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6), layout="constrained")
    for column, (condition, count, label, color) in enumerate(LEARNING):
        own = [r for r in rows if r["condition"] == condition and int(r["learning_episodes"]) == count]
        for row, (key, quantity) in enumerate((("mass", "mass (kg)"), ("friction", "friction coefficient"))):
            ax = axes[row, column]
            actual = np.array([float(r["true_" + key]) for r in own])
            predicted = np.array([float(r["predicted_" + key]) for r in own])
            low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
            ax.plot([low, high], [low, high], linestyle="--", linewidth=1, color="#718086")
            ax.scatter(actual, predicted, s=17, alpha=.6, color=color, edgecolors="none")
            ax.set(xlabel=f"True {quantity}", ylabel=f"Predicted {quantity}", aspect="equal")
            if row == 0:
                ax.set_title(label)
            pad = (high - low) * .05 or .01
            ax.set_xlim(low - pad, high + pad)
            ax.set_ylim(low - pad, high + pad)
            _style(ax)
    fig.suptitle("What the learned beliefs encode")
    fig.supxlabel("Each seed’s linear decoders are fitted on its training z table and evaluated on held-out cubes.",
                  fontsize=9)
    return _save(fig, dest / "meaning.png")


def plot_results(results_path, output_dir):
    with Path(results_path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    seeds = len({r["seed"] for r in rows})
    note = (f"Mean over cubes within each of {seeds} seeds. Error bars show ±1 seed SEM."
            if seeds > 1 else "Mean over cubes in one seed.")
    return _plot_transfer(rows, dest, note), _plot_formation(rows, dest, note), _plot_meaning(rows, dest)
