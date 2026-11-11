"""Render the three held-out PickCube figures and a short measured-results report."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


LABELS = {"all": "All episodes", "policy_noise": "Noisy policy", "scripted_push": "Scripted push",
          "random_poke": "Random poke", "push_and_poke": "Push and poke pooled"}
COLORS = {"policy_noise": "#17677b", "scripted_push": "#bd5f32", "random_poke": "#70609a"}
PHYSICS = {"mass_kg": "Cube mass", "cube_friction": "Cube friction", "table_friction": "Table friction",
           "grip_scale": "Gripper force scale", "damping_scale": "Arm damping scale",
           "stiffness_scale": "Arm stiffness scale", "delay_steps": "Command delay",
           "com_x_m": "Cube COM x", "com_y_m": "Cube COM y", "com_z_m": "Cube COM z"}


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dce1e3", linewidth=.7)
    ax.set_axisbelow(True)


def plot_horizons(run, rows, names):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True, layout="constrained")
    for ax, name in zip(axes.flat, ["all", *names]):
        group = [r for r in rows if r["source"] == name]
        x = [r["horizon"] for r in group]
        for key, label, color in (("W", "W with true z", "#17677b"),
                                   ("alternative", "W_noz", "#b46437")):
            ax.plot(x, [r[key] for r in group], marker="o", color=color, label=label)
            ax.fill_between(x, [r[key + "_ci_lo"] for r in group],
                            [r[key + "_ci_hi"] for r in group], color=color, alpha=.16, linewidth=0)
        ax.set(title=f"{LABELS[name]} · {group[0]['episode_count']:,}", xticks=x, ylim=(0, None))
        tidy(ax)
    axes[0, 0].legend(frameon=False)
    fig.supylabel("Normalized state prediction MSE")
    fig.supxlabel("Steps from the observed state")
    fig.suptitle("Held-out error over a five-step rollout", weight="bold")
    fig.savefig(run / "horizon.png", dpi=170)
    plt.close(fig)


def plot_ablations(run, results, levels, curves):
    fig, (bar, sweep) = plt.subplots(1, 2, figsize=(10.8, 4.8), layout="constrained")
    swap, mean = results["ablations"]["swapped"], results["ablations"]["mean"]
    values = np.array([swap["W"], swap["alternative"], mean["alternative"]])
    intervals = np.array([swap["W_ci"], swap["alternative_ci"], mean["alternative_ci"]]).T
    bar.bar(range(3), values, color=["#17677b", "#b46437", "#8a8298"], width=.62)
    bar.errorbar(range(3), values, yerr=[values - intervals[0], intervals[1] - values],
                 fmt="none", capsize=4, color="#29343d")
    for i, value in enumerate(values):
        bar.annotate(f"{value:.3f}", (i, intervals[1, i]), xytext=(0, 5),
                     textcoords="offset points", ha="center")
    bar.set(title="Context supplied to frozen W", xticks=range(3),
            xticklabels=["Own z", "Swapped z", "Training mean z"],
            ylabel="Normalized MSE averaged over horizons", ylim=(0, max(intervals[1]) * 1.16))
    tidy(bar)
    for i, (row, curve) in enumerate(zip(results["sweeps"], curves)):
        ratios = curve / curve[0]
        color = COLORS[row["source"]]
        sweep.plot(levels, ratios, label=f"{LABELS[row['source']]} {row['episode']}",
                   color=color, linestyle="--" if i % 2 else "-", linewidth=2)
        low = int(np.argmin(curve))
        sweep.scatter(levels[low], ratios[low], color=color, s=26, zorder=3)
    sweep.axhline(1., color="#788086", linewidth=.8)
    sweep.set(title="Six preselected held-out episodes",
              xlabel="Fraction from own z (0) to donor z (1)",
              ylabel="MSE divided by MSE at own z", xticks=np.linspace(0, 1, 6))
    tidy(sweep)
    sweep.legend(fontsize=8, frameon=False, ncols=2)
    failures = [i for i, row in enumerate(results["sweeps"]) if not row["true_z_minimum"]]
    if len(failures) == 1:
        i = failures[0]
        row = results["sweeps"][i]
        low = int(np.argmin(curves[i]))
        ratios = curves[i] / curves[i, 0]
        stop = min(low + 2, len(levels))
        inset = sweep.inset_axes([.08, .34, .32, .25])
        inset.plot(levels[:stop], ratios[:stop], marker=".", color=COLORS[row["source"]])
        inset.scatter(levels[low], ratios[low], s=16, color=COLORS[row["source"]])
        inset.set(title=f"{LABELS[row['source']]} {row['episode']} · detail", xticks=levels[:stop])
        inset.tick_params(labelsize=7)
        inset.title.set_fontsize(8)
        inset.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    fig.suptitle("Context replacement and interpolation", weight="bold")
    fig.supxlabel("Bars show 95% episode-bootstrap intervals. Dots mark the lowest sampled error on each curve.",
                  fontsize=9)
    fig.savefig(run / "context_checks.png", dpi=170)
    plt.close(fig)


def plot_probe(run, results, threshold):
    fig, ax = plt.subplots(figsize=(7.8, 4.7), layout="constrained")
    items = list(results["probe_r2"].items())[::-1]
    for i, (name, value) in enumerate(items):
        ax.barh(i, value, height=.65, color="#17677b" if value >= threshold else "#a1856a")
        ax.annotate(f"{value:.3f}", (max(0, value), i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9)
    ax.axvline(threshold, ls=":", lw=1, color="#766b61")
    ax.axvline(0, color="#39434a", lw=.8)
    ax.set(yticks=range(len(items)), yticklabels=[PHYSICS[k] for k, _ in items],
           xlim=(min(-.025, min(v for _, v in items) - .025), 1.1), xlabel="Held-out linear R²",
           title="Which physics labels can a linear model read from the encoder?")
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(run / "linear_probe.png", dpi=170)
    plt.close(fig)


def finish_report(run, cfg, comparison, results, levels, curves, names):
    plot_ablations(run, results, levels, curves)
    plot_probe(run, results, cfg["low_probe_r2"])
    with (run / "by_horizon.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summaries = comparison["sources"]
    gains = summaries["all"]
    ranges = json.loads((Path(cfg["dataset"]) / "dataset.json").read_text())["config"]
    range_text = {}
    for name in PHYSICS:
        if name.startswith("com_"):
            lo, hi = ranges["com_m"]
            range_text[name] = f"{100*lo:g} to +{100*hi:g} cm"
        elif name == "mass_kg":
            lo, hi = ranges[name]
            range_text[name] = f"{1000*lo:g}–{1000*hi:g} g (log-uniform)"
        elif name == "delay_steps":
            range_text[name] = f"{ranges[name][0]}–{ranges[name][1]} steps, integer"
        else:
            range_text[name] = f"{ranges[name][0]:g}–{ranges[name][1]:g}" + ("×" if "scale" in name else "")
    lines = [
        "# PickCube five-step world model",
        "",
        f"Trained on 36,000 episodes and evaluated on 4,000 untouched episodes. With the correct context, "
        f"mean held-out MSE was **{gains['W']:.4f}**. The independently trained zero-context baseline "
        f"scored **{gains['alternative']:.4f}**. The relative reduction was **{gains['gain_pct']:.1f}%**.",
        "",
        "## Step 3 · Forecast comparison",
        "",
        "Each entry gives W / W_noz. Lower is better. The right column averages the five horizon errors equally.",
        "",
        "| Action source | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["all", *names]:
        cells = [f"{float(r['W']):.3f} / {float(r['alternative']):.3f}" for r in rows if r["source"] == name]
        lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + f" | {summaries[name]['gain_pct']:.1f}% |")
    lines.extend([
        "",
        f"The overall paired 95% bootstrap interval for the MSE reduction was "
        f"[{gains['difference_ci'][0]:.4f}, {gains['difference_ci'][1]:.4f}]. "
        f"Relative gain on pooled pushes and pokes minus gain on noisy-policy episodes was "
        f"{comparison['gain_push_poke_minus_policy_pp']:+.2f} percentage points "
        f"(95% interval {comparison['source_difference_ci_pp'][0]:+.2f} to "
        f"{comparison['source_difference_ci_pp'][1]:+.2f}). Relative gain on horizons 4–5 minus "
        f"gain on horizons 1–2 was {comparison['gain_late_minus_early_pp']:+.2f} points "
        f"({comparison['horizon_difference_ci_pp'][0]:+.2f} to {comparison['horizon_difference_ci_pp'][1]:+.2f}).",
        "",
        "![Forecast error by action source and horizon](horizon.png)",
        "",
        "## Step 4 · Frozen context checks",
        "",
        "| Context | Mean MSE | Increase over own z | Paired 95% interval for MSE increase |",
        "| --- | ---: | ---: | ---: |",
        f"| Own z | {gains['W']:.4f} | — | — |",
    ])
    for key, label in (("swapped", "Another episode of the same action type"), ("mean", "Training-mean z")):
        r = results["ablations"][key]
        lines.append(f"| {label} | {r['alternative']:.4f} | {r['increase_over_W_pct']:.1f}% | "
                     f"[{r['difference_ci'][0]:.4f}, {r['difference_ci'][1]:.4f}] |")
    count = sum(r["true_z_minimum"] for r in results["sweeps"])
    minima = ", ".join(f"{r['episode']} at {r['minimizing_alpha']:.1f}" for r in results["sweeps"])
    lines.extend([
        "",
        f"The episode's own z minimized the sampled error for **{count}/{len(curves)} episodes**. "
        f"The minimizing fractions along the line to the donor were {minima}. "
        "Each curve uses eleven fractions and all 46 window starts in that episode.",
        "",
        "![Replacement errors and episode interpolation curves](context_checks.png)",
        "",
        "| True episode constant | Collection range | Held-out linear R² |",
        "| --- | --- | ---: |",
        *[f"| {PHYSICS[k]} | {range_text[k]} | {v:.4f} |" for k, v in results["probe_r2"].items()],
        "",
        f"Readouts below R² = {cfg['low_probe_r2']:.2f} were "
        + (", ".join(PHYSICS[k] for k in results["low_readout"]) or "none")
        + ". The probe fits on training encodings and scores physical labels in their original units. "
          "It measures linear information in E(true constants). Observability from motion remains an open question.",
        "",
        "![Held-out linear readouts of the ten constants](linear_probe.png)",
        "",
        "## Training and evaluation details",
        "",
        "The encoder has two 64-unit hidden layers and an eight-dimensional output. The dynamics MLP has four "
        "512-unit hidden layers. Both models start from identical shared dynamics weights and receive the same "
        f"{cfg['updates']:,} minibatches of {cfg['batch']} five-step windows. Training uses mean squared rollout "
        "error, equal weight across horizons, and AdamW with a cosine learning-rate schedule. The final update "
        "is the fixed checkpoint for both models. Normalizers use only the 36,000 training episodes. Cube mass "
        "is log-transformed before encoding.",
        "",
        "The 39 logged state coordinates comprise joint positions and velocities, tool pose, gripper width, "
        "and cube pose and velocities. The model converts both quaternions to two rotation columns each "
        "(43 features). W receives the current state, the issued command and z. W_noz has eight zero context "
        "columns. Only the first state of each window is observed. Each horizon error compares the predicted "
        "state to the logged one in training one-step delta-SD units. All 46 five-step starts per held-out "
        "episode are averaged before resampling whole episodes. Intervals measure episode variability for "
        "this trained model pair.",
        "",
        "[Settings](settings.json) · [Per-horizon errors and intervals](by_horizon.csv) · "
        "[Comparison](comparison.json) · [Frozen checks](checks.json) · "
        "[Interpolation measurements](interpolation.csv) · [Probe scores](probe.csv)",
    ])
    if not results["all_collapse_checks_pass"]:
        lines.extend(["", "## Decision after the frozen checks", ""])
        for row in results["sweeps"]:
            if not row["true_z_minimum"]:
                lines.append(f"{LABELS[row['source']]} episode {row['episode']} scored "
                             f"{row['error_true']:.5f} at its own z and {row['error_min']:.5f} "
                             f"at interpolation fraction {row['minimizing_alpha']:.1f}.")
        if comparison["source_difference_ci_pp"][0] <= 0 <= comparison["source_difference_ci_pp"][1]:
            lines.extend(["", "The advantage on pushes and pokes over policy actions remains inconclusive."])
        lines.extend(["", "Stopped after the requested checks. The proposed follow-up is a single change "
                      "to a ten-step rollout, keeping z at eight dimensions and preserving the action mix. "
                      "The measured advantage grew with horizon in this run. Fresh held-out episodes would "
                      "test whether a longer rollout also improves recovery of the episode's own z."])
    (run / "RESULTS.md").write_text("\n".join(lines) + "\n")
