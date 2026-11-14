"""Audit existing shove predictions and decoding without fitting model or beliefs."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import torch

from .collect import load_rows
from .model import load_world_model
from .target_control import CANDIDATE_FORCES_N, FINAL_COUNTS, _episodes, _final_beliefs


LABELS = dict(train_mean="Train mean", lifts_only="Five lifts", slides_only="Five slow slides",
              both="Five lifts + five slides", other_both="Another cube’s combined belief",
              shove_ceiling="Five-shove ceiling")


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Nonfinite values require explicit handling before diagnostic aggregation")
    error = predicted - actual
    denominator = np.sum((actual - actual.mean()) ** 2)
    correlation = float(np.corrcoef(actual, predicted)[0, 1]) if np.std(actual) > 0 and np.std(predicted) > 0 else None
    return dict(count=len(actual), pearson_r=correlation,
                prediction_r2=float(1 - np.sum(error ** 2) / denominator) if denominator > 0 else None,
                mae=float(np.abs(error).mean()), bias=float(error.mean()),
                rmse=float(np.sqrt(np.mean(error ** 2))))


def seed_average(values, seeds):
    means = np.array([np.mean(values[seeds == seed]) for seed in np.unique(seeds)])
    return dict(mean=float(means.mean()), seed_sem=float(means.std(ddof=1) / np.sqrt(len(means))),
                seed_means=dict(zip(map(str, np.unique(seeds)), means.tolist())))


def paired_cube_bootstrap(cube_values, conditions):
    """Resample whole cube IDs, retaining their five seed results and both conditions."""
    seed_ids, cube_ids = np.unique(cube_values["seed"]), np.unique(cube_values["cube_id"])
    rng = np.random.default_rng(20261114)
    draws = rng.integers(len(cube_ids), size=(10000, len(cube_ids)))
    matrices = {}
    for condition in conditions:
        chosen = cube_values["condition"] == condition
        keyed = {(int(seed), int(cube)): value for seed, cube, value in
                 zip(cube_values["seed"][chosen], cube_values["cube_id"][chosen], cube_values["value"][chosen])}
        matrices[condition] = np.array([[keyed[int(seed), int(cube)] for cube in cube_ids] for seed in seed_ids])
    contrasts = {}
    for comparator in ("train_mean", "other_both", "lifts_only"):
        delta = matrices["both"] - matrices[comparator]
        difference_by_cube = delta.mean(0)
        ci = np.quantile(difference_by_cube[draws].mean(1), [.025, .975])
        contrasts[comparator] = dict(
            both_minus_comparator=float(delta.mean()), difference_ci95=ci.tolist(),
            relative_reduction=float(1 - matrices["both"].mean() / matrices[comparator].mean()),
            seed_mean_differences=delta.mean(1).tolist(),
        )
    return dict(bootstrap_draws=len(draws), bootstrap_seed=20261114, cube_count=len(cube_ids),
                seed_count=len(seed_ids),
                estimand="Equal cube and seed weights. Negative both-minus-comparator favors both.",
                uncertainty="Cube sampling uncertainty conditional on the five fitted seeds. Whole cubes retain all seed results.",
                comparisons=contrasts)


def csv_diagnostics(rows):
    final = [r for r in rows if int(r["learning_episodes"]) == FINAL_COUNTS[r["condition"]]]
    expected = set(itertools.product({int(r["seed"]) for r in final},
                                     {int(r["cube_id"]) for r in final}, FINAL_COUNTS))
    keys = [(int(r["seed"]), int(r["cube_id"]), r["condition"]) for r in final]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("CSV final conditions lack one-to-one paired seed/cube coverage")
    values = dict(seed=np.array([k[0] for k in keys]), cube_id=np.array([k[1] for k in keys]),
                  condition=np.array([k[2] for k in keys]),
                  value=np.array([float(r["shove_error"]) for r in final]))
    one_step = {}
    for condition in FINAL_COUNTS:
        selected = values["condition"] == condition
        one_step[condition] = seed_average(values["value"][selected], values["seed"][selected])
    decoding = {}
    for condition in ("lifts_only", "slides_only", "both"):
        selected = [r for r in final if r["condition"] == condition]
        seeds = np.array([int(r["seed"]) for r in selected])
        decoding[condition] = {}
        for key, scale, unit in (("mass", 1000., "g"), ("friction", 1., "dimensionless")):
            actual = np.array([float(r["true_" + key]) for r in selected]) * scale
            predicted = np.array([float(r["predicted_" + key]) for r in selected]) * scale
            decoding[condition][key] = dict(
                unit=unit, pooled=regression_metrics(actual, predicted),
                by_seed={str(seed): regression_metrics(actual[seeds == seed], predicted[seeds == seed])
                         for seed in np.unique(seeds)})
    return dict(one_step_full_state=one_step, paired_full_state=paired_cube_bootstrap(values, FINAL_COUNTS),
                decoding=decoding), values


def verify_trials(study, episodes, trials):
    """Check complete trial keys and every join to the executed simulator cache."""
    expected_seeds = sorted(json.loads((study / "protocol.json").read_text())["optimizer_seeds"])
    np.testing.assert_array_equal(np.unique(trials["seed"]), expected_seeds)
    keys = list(zip(trials["seed"].tolist(), trials["cube_id"].tolist(),
                    trials["episode_id"].tolist(), trials["condition"].tolist()))
    expected = {(int(seed), cube, ep, condition) for seed in np.unique(trials["seed"])
                for cube, ep in episodes for condition in FINAL_COUNTS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Trial keys are duplicated or incomplete")
    origins = np.stack([episodes[cube, ep]["initial_state"][19:21] for _, cube, ep, _ in keys]).astype(float)
    directions = np.stack([episodes[cube, ep]["direction"] for _, cube, ep, _ in keys])
    expected_targets = origins + directions * trials["target_distance_m"][:, None]
    np.testing.assert_allclose(trials["target_xy"], expected_targets, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(trials["force_n"], CANDIDATE_FORCES_N[trials["candidate_index"]])
    if not np.isfinite(trials["predicted_endpoint_xy"]).all():
        raise ValueError("Nonfinite saved predictions must be retained with explicit invalid marks")
    trajectories = {}
    maximum_ready_difference = 0.
    for cube, ep, candidate in sorted(set(zip(trials["cube_id"], trials["episode_id"], trials["candidate_index"]))):
        key = int(cube), int(ep), int(candidate)
        path = study / "target_control" / "trajectories" / f"cube_{cube:04}_episode_{ep}_candidate_{candidate:02}.npz"
        episode = episodes[cube, ep]
        with np.load(path, allow_pickle=False) as cached:
            if (int(cached["cube_id"][0]), int(cached["episode_id"][0]), int(cached["candidate_index"])) != key:
                raise ValueError("Cached trajectory identity disagrees with its lookup key")
            start, stop = int(cached["force_start_step"][0]), int(cached["stop_step"][0])
            if (start, stop) != (episode["force_start"], episode["stop"]):
                raise ValueError("Cached execution horizon differs from the planned horizon")
            ready_error = np.max(np.abs(cached["state"][start] - episode["initial_state"]))
            maximum_ready_difference = max(maximum_ready_difference, float(ready_error))
            np.testing.assert_allclose(cached["state"][start], episode["initial_state"], atol=1e-5, rtol=0)
            trajectories[key] = (cached["next_state"][stop, 19:21].astype(float), bool(cached["observed_stopped"][0]))
    for i, (_, cube, ep, _) in enumerate(keys):
        endpoint, stopped = trajectories[cube, ep, int(trials["candidate_index"][i])]
        np.testing.assert_array_equal(trials["actual_endpoint_xy"][i], endpoint)
        if bool(trials["actual_stopped"][i]) != stopped:
            raise ValueError("Actual stopped flag differs from the paired simulator cache")
    predicted_signed = np.sum((trials["predicted_endpoint_xy"] - origins) * directions, axis=1) * 1000.
    actual_signed = np.sum((trials["actual_endpoint_xy"] - origins) * directions, axis=1) * 1000.
    derived = dict(trials, origin_xy=origins, direction=directions,
                   predicted_signed_distance_mm=predicted_signed, actual_signed_distance_mm=actual_signed)
    return derived, dict(trials=len(keys), unique_trajectories=len(trajectories),
                         seeds=expected_seeds, cube_count=len({cube for cube, _ in episodes}),
                         conditions=list(FINAL_COUNTS), episodes_per_cube=5,
                         max_ready_state_absolute_difference=maximum_ready_difference,
                         identity_keys="seed, cube_id, episode_id, condition",
                         execution_join_keys="cube_id, episode_id, candidate_index",
                         complete_cartesian_product=True, all_cached_endpoint_joins_exact=True)


def rollout_summary(trials):
    result = {}
    speeds = trials["predicted_final_speed_m_s"] * 1000.
    for condition in FINAL_COUNTS:
        selected = trials["condition"] == condition
        actual = trials["actual_signed_distance_mm"][selected]
        predicted = trials["predicted_signed_distance_mm"][selected]
        speed = speeds[selected]
        seeds = trials["seed"][selected]
        xy_difference = (trials["predicted_endpoint_xy"][selected] - trials["actual_endpoint_xy"][selected]) * 1000.
        xy_error = np.hypot(xy_difference[:, 0], xy_difference[:, 1])
        result[condition] = dict(
            signed_distance_mm=regression_metrics(actual, predicted),
            endpoint_xy_disagreement_mm=dict(mean=float(xy_error.mean()),
                                             rmse=float(np.sqrt(np.mean(xy_error ** 2))),
                                             median=float(np.median(xy_error)),
                                             p90=float(np.quantile(xy_error, .9))),
            actual_signed_distance_mm_quantiles=dict(zip(("min", "p25", "median", "p75", "max"), np.quantile(actual, [0, .25, .5, .75, 1]).tolist())),
            predicted_signed_distance_mm_quantiles=dict(zip(("min", "p25", "median", "p75", "max"), np.quantile(predicted, [0, .25, .5, .75, 1]).tolist())),
            predicted_terminal_speed_mm_s=dict(mean=float(speed.mean()), median=float(np.median(speed)),
                                               p90=float(np.quantile(speed, .9)), maximum=float(speed.max()),
                                               above_10_mm_s=int(np.sum(speed > 10)),
                                               above_10_mm_s_fraction=float(np.mean(speed > 10))),
            actual_stopped=int(trials["actual_stopped"][selected].sum()),
            by_seed={str(seed): regression_metrics(actual[seeds == seed], predicted[seeds == seed]) for seed in np.unique(seeds)},
        )
    return result


def scatter_plot(trials, summary, output):
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 9), layout="constrained", sharex=True, sharey=True)
    actual = trials["actual_signed_distance_mm"]
    predicted = trials["predicted_signed_distance_mm"]
    speed = trials["predicted_final_speed_m_s"] * 1000.
    xpad = max(np.ptp(actual) * .04, .1)
    ypad = max(np.ptp(predicted) * .04, .1)
    limits_x = (min(actual.min(), 0) - xpad, actual.max() + xpad)
    limits_y = (min(predicted.min(), limits_x[0]) - ypad, max(predicted.max(), limits_x[1]) + ypad)
    norm = colors.LogNorm(vmin=max(float(speed.min()), .001), vmax=float(speed.max()))
    for ax, condition in zip(axes.flat, FINAL_COUNTS):
        selected = trials["condition"] == condition
        dots = ax.scatter(actual[selected], predicted[selected], c=speed[selected], norm=norm,
                          cmap="viridis", s=12, alpha=.65, edgecolors="none", rasterized=True)
        ax.plot(limits_x, limits_x, "--", color="#b44d41", linewidth=1.2, label="Predicted = actual")
        ax.set(title=LABELS[condition], xlim=limits_x, ylim=limits_y)
        ax.grid(alpha=.15)
        metrics = summary[condition]["signed_distance_mm"]
        s = summary[condition]["predicted_terminal_speed_mm_s"]
        ax.text(.03, .97, f"n = {metrics['count']}   r = {metrics['pearson_r']:.3f}\n"
                         f"MAE {metrics['mae']:.1f} mm   bias {metrics['bias']:+.1f} mm\n"
                         f"Speed median {s['median']:.1f} mm/s",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    fig.colorbar(dots, ax=axes, shrink=.88, label="Predicted terminal speed (mm/s, logarithmic scale)")
    fig.supxlabel("Actual signed resting distance along the commanded shove direction (mm)")
    fig.supylabel("Predicted signed distance at the fixed rollout horizon (mm)")
    fig.suptitle("Executed pushes and their model predictions\n"
                 "All 4,500 trial records retained; red dashed line is identity", fontsize=14)
    fig.savefig(output / "predicted_vs_actual_distance.png", dpi=180)
    fig.savefig(output / "predicted_vs_actual_distance.pdf")
    plt.close(fig)


@torch.no_grad()
def cube_one_step(data, study, csv_values, output, device):
    """Fresh frozen-model scores expose cube coordinates within the full 43D loss."""
    rows = load_rows(data, "test")
    moving = ((rows["task"] == 2) & (rows["episode_id"] < 5) & rows["cube_moving"]
              & (rows["step"] >= rows["force_start_step"]) & (rows["step"] <= rows["stop_step"]))
    state, action, following = [torch.as_tensor(rows[k][moving], dtype=torch.float32, device=device)
                                for k in ("state", "action", "next_state")]
    cubes = rows["cube_id"][moving]
    ids = np.unique(cubes)
    records = []
    reference_differences = []
    for seed in np.unique(csv_values["seed"]):
        model = load_world_model(study / f"seed_{seed}" / "model.pt", device=device)
        beliefs = _final_beliefs(study / f"seed_{seed}" / "beliefs.npz")
        inputs, target = model.model.normalize(torch.cat((state, action), 1), following - state)
        for condition in FINAL_COUNTS:
            z = torch.as_tensor(np.stack([beliefs[int(cube)][condition] for cube in cubes]),
                                dtype=torch.float32, device=device)
            difference = model.model(inputs, z) - target
            squared = difference.double().square()
            raw_mm_squared = (difference.double() * model.model.delta_sd.double() * 1000).square()
            full = squared.mean(1).cpu().numpy()
            xy = squared[:, 19:21].mean(1).cpu().numpy()
            velocity = squared[:, 28:31].mean(1).cpu().numpy()
            xy_mm = raw_mm_squared[:, 19:21].mean(1).cpu().numpy()
            velocity_mm_s = raw_mm_squared[:, 28:31].mean(1).cpu().numpy()
            for cube in ids:
                selected = cubes == cube
                reference = ((csv_values["seed"] == seed) & (csv_values["cube_id"] == cube)
                             & (csv_values["condition"] == condition))
                original = csv_values["value"][reference].item()
                computed = full[selected].mean()
                # The study used CUDA high matmul precision; this diagnostic defaults to CPU.
                np.testing.assert_allclose(computed, original, rtol=1e-3, atol=1e-6)
                reference_differences.append((abs(computed-original), abs(computed-original) / max(abs(original), 1e-12)))
                records.append(dict(seed=int(seed), cube_id=int(cube), condition=condition,
                                    full_state_mse=full[selected].mean(), cube_xy_mse=xy[selected].mean(),
                                    cube_velocity_mse=velocity[selected].mean(),
                                    cube_xy_rmse_mm=np.sqrt(xy_mm[selected].mean()),
                                    cube_velocity_rmse_mm_s=np.sqrt(velocity_mm_s[selected].mean()),
                                    moving_steps=int(selected.sum())))
        del model
    arrays = {key: np.array([r[key] for r in records]) for key in records[0]}
    np.savez_compressed(output / "cube_one_step_metrics.npz", **arrays)
    summary = {}
    for metric in ("full_state_mse", "cube_xy_mse", "cube_velocity_mse", "cube_xy_rmse_mm", "cube_velocity_rmse_mm_s"):
        summary[metric] = {
            condition: seed_average(arrays[metric][arrays["condition"] == condition], arrays["seed"][arrays["condition"] == condition])
            for condition in FINAL_COUNTS
        }
        summary[metric]["paired_cube_bootstrap"] = paired_cube_bootstrap(
            dict(seed=arrays["seed"], cube_id=arrays["cube_id"], condition=arrays["condition"], value=arrays[metric]),
            FINAL_COUNTS)
    return dict(definition="Normalized MSE uses training delta scales. XY is indices 19,20; velocity is 28,29,30. Physical RMSE is calculated per cube, then cube/seed averaged.",
                moving_rows_per_seed=len(cubes), full_state_matches_existing_csv=True,
                csv_comparison_tolerance=dict(relative=.001, absolute=.000001),
                csv_max_absolute_difference=float(np.max(reference_differences, axis=0)[0]),
                csv_max_relative_difference=float(np.max(reference_differences, axis=0)[1]),
                metrics=summary)


def run(root, output, device):
    data, study = root / "data", root / "study"
    output.mkdir(parents=True, exist_ok=True)
    with (study / "results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    csv_summary, csv_values = csv_diagnostics(rows)
    cfg = json.loads((data / "dataset.json").read_text())
    episodes = _episodes(data, cfg)
    with np.load(study / "target_control" / "trials.npz", allow_pickle=False) as saved:
        trials = {key: saved[key] for key in saved.files}
    joined, join_report = verify_trials(study, episodes, trials)
    np.testing.assert_array_equal(np.unique(csv_values["seed"]), np.unique(trials["seed"]))
    np.testing.assert_array_equal(np.unique(csv_values["cube_id"]), np.unique(trials["cube_id"]))
    rollout = rollout_summary(joined)
    np.savez_compressed(output / "signed_distance_trials.npz", **joined)
    scatter_plot(joined, rollout, output)
    cube_metrics = cube_one_step(data, study, csv_values, output, device)
    sources = [Path(__file__), data / "dataset.json", study / "results.csv", study / "target_control" / "trials.npz"]
    sources += [study / f"seed_{seed}" / name for seed in np.unique(trials["seed"]) for name in ("model.pt", "beliefs.npz")]
    summary = dict(
        definitions=dict(
            one_step="Moving pulse/coast transition delta MSE averaged over all 43 observable state dimensions; equal cube and seed weights.",
            decoding="Existing train-table-fitted linear decoder predictions on held-out cube beliefs, copied from results.csv.",
            signed_distance="Dot product of endpoint minus measured force-onset cube XY with the commanded unit direction, converted to mm.",
            rollout="The predicted endpoint is at the fixed observed execution horizon. Continuing predicted motion means it is not a predicted resting location.",
            observation="Actual endpoints all passed the recorded rest criterion.",
            bias="Predicted signed distance minus actual signed distance.",
            trial_dependence="4,500 records include repeated seeds/conditions sharing executed simulator trajectories; count is not an independent sample size.",
            selected_action_conditioning="These predictions concern model-selected forces. Correlation describes those executed choices and does not estimate prediction quality over the entire candidate action grid.",
        ),
        join_validation=join_report, csv_diagnostics=csv_summary, rollout=rollout,
        cube_one_step=cube_metrics,
        source_sha256={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(output), join_validation=join_report,
                          one_step=csv_summary["one_step_full_state"],
                          signed_distance={k: v["signed_distance_mm"] for k, v in rollout.items()}), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.root, args.output or args.root / "diagnostics" / "rollout", args.device)
