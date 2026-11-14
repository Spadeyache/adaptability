"""Reformat archived experiments into figures. Run with the project's Python."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent
BLUE, RUST, GRAY = "#14678a", "#984521", "#65717f"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sources(*paths):
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def canvas(title):
    fig, ax = plt.subplots(figsize=(7.8, 4.1), layout="constrained")
    ax.set_title(title, loc="left", fontsize=15, pad=10, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9aa2a8")
    ax.tick_params(axis="both", length=0, pad=5)
    ax.grid(axis="y", color="#dde2e5", linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def save(name, figure, data):
    folder = OUTPUT / name
    folder.mkdir(exist_ok=True)
    figure.savefig(folder / f"{name}.png", dpi=240, facecolor="white")
    plt.close(figure)
    (folder / f"{name}.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def forecast():
    run = ROOT / "runs/context_dynamics/pick_gpu_2m/world_model_z8"
    scores_path = run / "intervention_errors.npz"
    summary_path = run / "checks.json"
    config_path = run / "settings.json"
    config = read_json(config_path)
    archived = read_json(summary_path)["ablations"]
    curves = [
        ("W", "Own encoded\nconstants", archived["swapped"]["W"], archived["swapped"]["W_ci"]),
        ("swapped", "Swapped\ncontext", archived["swapped"]["alternative"], archived["swapped"]["alternative_ci"]),
        ("mean", "Training-mean\ncontext", archived["mean"]["alternative"], archived["mean"]["alternative_ci"]),
    ]
    series = []
    with np.load(scores_path, allow_pickle=False) as saved:
        n = len(saved["episode_ids"])
        if len(np.unique(saved["episode_ids"])) != n or n != archived["swapped"]["episode_count"]:
            raise ValueError("Held-out episode counts disagree")
        for key, label, center, ci in curves:
            values = saved[key]
            if values.shape != (n, config["horizon"]):
                raise ValueError("Forecast array has the wrong shape")
            np.testing.assert_allclose(values.mean(), center, rtol=1e-12)
            series.append(dict(key=key, label=label.replace("\n", " "), mse=center, ci95=ci))
    data = dict(
        metric="Mean normalized squared state prediction error over horizons 1 through 5",
        population="All 46 five-step windows of each held-out PickCube episode, including free motion",
        held_out_episodes=n, context="Eight coordinates encoded from ten true simulator constants",
        swapped="A different held-out episode with the same action source",
        training_mean="Mean of encoded training contexts",
        interval=dict(method="Percentile bootstrap resampling whole paired episodes",
                      level=0.95, draws=config["bootstrap_draws"], seed=config["bootstrap_seed"]),
        series=series, paired_alternative_minus_own={
            k: dict(mse_difference=v["difference"], ci95=v["difference_ci"])
            for k, v in archived.items()
        }, source_sha256=sources(scores_path, summary_path, config_path),
    )
    fig, ax = canvas("Five-step forecasts on 4,000 held-out episodes")
    # The zero baseline preserves the magnitude comparison between the three bars.
    for i, ((_, label, center, ci), color) in enumerate(zip(curves, [BLUE, GRAY, GRAY])):
        ax.bar(i, center, width=.59, color=color)
        ax.errorbar(i, center, yerr=[[center-ci[0]], [ci[1]-center]], fmt="none",
                    color="#202a30", linewidth=1.5, capsize=5)
        ax.text(i, ci[1] + .055, f"{center:.3f}", ha="center", va="bottom", weight="bold")
    ax.set_xticks(range(len(curves)), [row[1] for row in curves])
    ax.set_ylabel("Normalized forecast error (MSE)")
    ax.set_ylim(0, 2.28)
    ax.set_xlim(-.58, 2.58)
    save("forecast", fig, data)


def retention():
    run = ROOT / "runs/mass_belief/nov17_v1"
    trials_path, summary_path, config_path = (run / f for f in ("trials.csv", "summary.json", "config.json"))
    with trials_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    summary, config = read_json(summary_path), read_json(config_path)
    n, k = config["sequences"], config["trials"]
    x = np.arange(1, k + 1)
    draws = np.random.default_rng(config["seed"]+1).integers(n, size=(10000, n))
    fig, ax = canvas("Mass estimates over five attempts on a 400 g cube")
    series = []
    for arm, name, color, marker in [
        ("Filter reset", "Reset each trial", RUST, "s"),
        ("B: filter carried", "Carried across trials", BLUE, "o"),
    ]:
        subset = [r for r in rows if r["arm"] == arm]
        errors = np.full((n, k), np.nan)
        for r in subset:
            err = abs(float(r["truth_g"])-float(r["estimate_g"]))
            np.testing.assert_allclose(err, float(r["abs_error_g"]), atol=1e-10)
            errors[int(r["sequence"]), int(r["trial"])-1] = err
        if len(subset) != n*k or not np.isfinite(errors).all():
            raise ValueError("Incomplete sequence / trial data")
        means = errors.mean(axis=0)
        lower, upper = np.quantile(errors[draws].mean(axis=1), [.025, .975], axis=0)
        np.testing.assert_allclose(np.column_stack([means, lower, upper]), summary[arm]["mae_g_mean_lo_hi"])
        ax.fill_between(x, lower, upper, color=color, alpha=.15, linewidth=0)
        ax.plot(x, means, color=color, marker=marker, linestyle="--" if marker=="s" else "-",
                linewidth=2.3, markersize=6, label=name)
        ax.annotate(f"{means[-1]:.1f} g", (k, means[-1]), xytext=(8,0),
                    textcoords="offset points", va="center", color=color, weight="bold")
        series.append(dict(label=name, source_arm=arm, trial=x.tolist(), mean_mae_g=means.tolist(),
                           ci95_lower_g=lower.tolist(), ci95_upper_g=upper.tolist()))
    ax.set(xticks=x, xlim=(.83, 5.67), ylim=(0, 125), xlabel="Lift attempt",
           ylabel="Mean absolute mass error (g)")
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=11,
              handlelength=2, columnspacing=1.5)
    paired = summary["carried_minus_comparator_mae_g"]["Filter reset"][-1]
    data = dict(metric="End-of-trial mean absolute mass error in grams", sequence_count=n,
                trials_per_sequence=k, nominal_cube_mass_g=config["heavy_g"], cube_friction=config["friction"],
                fixed_policy_guess_g=config["probe_g"], horizon_steps=config["horizon"],
                observations="Both filters process identical observations from independently reset lift attempts",
                interval=dict(method="Percentile bootstrap resampling paired five-trial sequences",
                              level=0.95, draws=len(draws), seed=config["seed"]+1),
                series=series, trial_five_carried_minus_reset_g=dict(mean=paired[0], ci95=paired[1:]),
                source_sha256=sources(trials_path, summary_path, config_path))
    save("retention", fig, data)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "text.color": "#202a30",
                         "axes.labelcolor": "#202a30", "xtick.color": "#202a30", "ytick.color": "#202a30"})
    forecast()
    retention()
