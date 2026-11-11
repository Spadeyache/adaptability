"""Compare matched PickCube forecasts, then check the frozen privileged context."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .check import probe, write_csv, write_json
from .data import Trajectories, restore
from .pick_train import digest


def episode_errors(model, data, ids, contexts, horizon):
    """Average all valid rolling starts within an episode, keeping each horizon separate."""
    starts = torch.arange(data.horizon - horizon + 1, device=data.device)
    output = []
    for block, z in zip(ids.split(32), contexts.split(32)):
        states, actions = data.windows(block.repeat_interleave(len(starts)), starts.repeat(len(block)), horizon)
        errors = model.error_by_horizon(states, actions, z.repeat_interleave(len(starts), dim=0))
        output.append(errors.reshape(len(block), len(starts), horizon).double().mean(1).cpu().numpy())
    values = np.concatenate(output)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite held-out forecast")
    return values


def resampled_means(values, rng, draws):
    chunks = []
    for start in range(0, draws, 50):
        ids = rng.integers(len(values), size=(min(50, draws - start), len(values)))
        chunks.append(values[ids].mean(axis=1))
    return np.concatenate(chunks)


def interval(values):
    return np.quantile(values, [.025, .975], axis=0).tolist()


def paired_summary(reference, alternative, averaged, samples):
    difference = alternative - reference
    return dict(W=float(reference), alternative=float(alternative),
                difference=float(difference), difference_ci=interval(samples[:, 1] - samples[:, 0]),
                W_ci=interval(samples[:, 0]), alternative_ci=interval(samples[:, 1]),
                gain_pct=float(100 * difference / alternative),
                increase_over_W_pct=float(100 * difference / reference),
                episode_count=averaged)


def summarize(errors, alternative, source, names, cfg):
    summaries, rows, draws_by_source = {}, [], {}
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    for label in ["all", *names, "push_and_poke"]:
        if label == "all":
            mask = np.ones(len(source), dtype=bool)
        elif label == "push_and_poke":
            mask = source != "policy_noise"
        else:
            mask = source == label
        # One paired draw resamples whole episodes, preserving all five horizons and both predictions.
        pairs = np.stack((errors[mask], alternative[mask]), axis=-1)
        point = pairs.mean(0)
        draws = resampled_means(pairs, rng, cfg["bootstrap_draws"])
        draws_by_source[label] = draws
        summaries[label] = paired_summary(*point.mean(0), int(mask.sum()), draws.mean(1))
        for h in range(cfg["horizon"]):
            row = paired_summary(*point[h], int(mask.sum()), draws[:, h])
            for key in ("W_ci", "alternative_ci", "difference_ci"):
                row[key + "_lo"], row[key + "_hi"] = row.pop(key)
            rows.append(dict(source=label, horizon=h + 1, **row))
    return summaries, rows, draws_by_source


def percent_gain(pair):
    return 100 * (pair[..., 1] - pair[..., 0]) / pair[..., 1]


def compare(run, data, ids, source, names, models, cfg):
    w, noz = models
    z = w.encode(data.constants[ids])
    errors = episode_errors(w, data, ids, z, cfg["horizon"])
    baseline = episode_errors(noz, data, ids, torch.zeros_like(z), cfg["horizon"])
    summaries, rows, bootstrap = summarize(errors, baseline, source, names, cfg)
    pairs = np.stack((errors, baseline), axis=-1)
    all_draws = bootstrap["all"]
    long_minus_short = percent_gain(pairs.mean(0)[3:].mean(0)) - percent_gain(pairs.mean(0)[:2].mean(0))
    long_draws = percent_gain(all_draws[:, 3:].mean(1)) - percent_gain(all_draws[:, :2].mean(1))
    source_difference = summaries["push_and_poke"]["gain_pct"] - summaries["policy_noise"]["gain_pct"]
    source_draws = (percent_gain(bootstrap["push_and_poke"].mean(1))
                    - percent_gain(bootstrap["policy_noise"].mean(1)))
    overall = summaries["all"]
    result = dict(sources=summaries, gain_push_poke_minus_policy_pp=float(source_difference),
                  source_difference_ci_pp=interval(source_draws), gain_late_minus_early_pp=float(long_minus_short),
                  horizon_difference_ci_pp=interval(long_draws),
                  clear_overall=overall["gain_pct"] >= 1 and overall["difference_ci"][0] > 0,
                  greater_on_push_poke=bool(source_difference > 0), greater_at_long_horizons=bool(long_minus_short > 0))
    np.savez_compressed(run / "forecast_errors.npz", episode_ids=ids.cpu().numpy(), source=source,
                        W=errors, W_noz=baseline)
    write_csv(run / "by_horizon.csv", rows)
    write_json(run / "comparison.json", result)
    from .pick_figures import plot_horizons
    plot_horizons(run, rows, names)
    print(json.dumps(result, indent=2), flush=True)


def checks(run, data, ids, source, names, model, cfg, constants):
    reference = json.loads((run / "comparison.json").read_text())
    ids_cpu = ids.cpu().numpy()
    with np.load(run / "forecast_errors.npz", allow_pickle=False) as previous:
        np.testing.assert_array_equal(previous["episode_ids"], ids_cpu)
        np.testing.assert_array_equal(previous["source"], source)
        errors = previous["W"]
    train_x, test_x = data.constants[data.train_ids], data.constants[ids]
    train_z, z = model.encode(train_x), model.encode(test_x)
    rng = np.random.default_rng(cfg["swap_seed"])
    donors = np.arange(len(ids))
    for name in names:
        shuffled = rng.permutation(np.flatnonzero(source == name))
        donors[shuffled] = np.roll(shuffled, 1)
    assert np.all(donors != np.arange(len(ids))) and np.array_equal(source, source[donors])
    swapped_z = z[torch.as_tensor(donors, device=data.device)]
    contexts = {"swapped": swapped_z, "mean": train_z.mean(0).expand_as(z)}
    ablations, saved = {}, {}
    for name, context in contexts.items():
        values = episode_errors(model, data, ids, context, cfg["horizon"])
        saved[name] = values
        # Paired CIs for the two interventions, all horizons averaged within each episode.
        pairs = np.column_stack((errors.mean(1), values.mean(1)))
        draws = resampled_means(pairs, np.random.default_rng(cfg["bootstrap_seed"]), cfg["bootstrap_draws"])
        ablations[name] = paired_summary(*pairs.mean(0), len(ids), draws)
    np.savez_compressed(run / "intervention_errors.npz", episode_ids=ids_cpu, donor_ids=ids_cpu[donors],
                        W=errors, **saved)

    selected = np.concatenate([np.flatnonzero(source == name)[:cfg["examples_per_source"]] for name in names])
    selected_t = torch.as_tensor(selected, device=data.device)
    levels = np.linspace(0, 1, cfg["interpolation_intervals"] + 1)
    curves = []
    for alpha in levels:
        interpolated = torch.lerp(z[selected_t], swapped_z[selected_t], float(alpha))
        curves.append(episode_errors(model, data, ids[selected_t], interpolated, cfg["horizon"]).mean(1))
    curves = np.stack(curves, axis=1)
    np.testing.assert_allclose(curves[:, 0], errors[selected].mean(1), rtol=3e-5, atol=5e-7)
    np.testing.assert_allclose(curves[:, -1], saved["swapped"][selected].mean(1), rtol=3e-5, atol=5e-7)
    sweeps, sweep_rows = [], []
    for i, (pos, curve) in enumerate(zip(selected, curves)):
        row = dict(episode=int(ids_cpu[pos]), source=str(source[pos]), donor=int(ids_cpu[donors[pos]]))
        sweeps.append(dict(**row, minimizing_alpha=float(levels[np.argmin(curve)]),
                           true_z_minimum=bool(curve[0] <= curve.min() + 1e-7),
                           error_true=float(curve[0]), error_min=float(curve.min())))
        sweep_rows.extend(dict(**row, alpha=float(a), error=float(v)) for a, v in zip(levels, curves[i]))
    write_csv(run / "interpolation.csv", sweep_rows)
    r2, readout = probe(*(tensor.double().cpu().numpy() for tensor in (train_z, z, train_x, test_x)))
    np.savez(run / "readout.npz", **readout)
    probe_rows = [dict(constant=k, r2=float(value)) for k, value in zip(constants, r2)]
    write_csv(run / "probe.csv", probe_rows)
    gates = {key: v["difference_ci"][0] > 0 and v["increase_over_W_pct"] >= 1
             for key, v in ablations.items()}
    gates["interpolation"] = all(r["true_z_minimum"] for r in sweeps)
    result = dict(ablations=ablations, sweeps=sweeps, probe_r2=dict(zip(constants, r2.tolist())),
                  low_readout=[k for k, value in zip(constants, r2) if value < cfg["low_probe_r2"]],
                  collapse_checks=gates, all_collapse_checks_pass=all(gates.values()))
    write_json(run / "checks.json", result)
    from .pick_figures import finish_report
    finish_report(run, cfg, reference, result, levels, curves, names)
    print(json.dumps(result, indent=2), flush=True)


@torch.inference_mode()
def evaluate(stage, folder, run, device):
    cfg = json.loads((run / "settings.json").read_text())
    complete = json.loads((run / "training_complete.json").read_text())
    for name, expected in complete.items():
        if digest(run / name / "model.pt") != expected:
            raise ValueError(f"Wrong model weights for {name}")
    if digest(folder / "checksums.json") != cfg["checksums_sha256"]:
        raise ValueError("Evaluation dataset differs from the training protocol")
    meta = json.loads((folder / "dataset.json").read_text())
    with (folder / "episodes.csv").open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == "validation"]
    source = np.asarray([r["source"] for r in rows])
    ids_cpu = np.asarray([int(r["episode_id"]) for r in rows])
    torch.set_float32_matmul_precision("highest")
    data = Trajectories(folder, device)
    held = torch.cat((data.select_ids, *data.check_ids)).sort().values
    np.testing.assert_array_equal(held.cpu().numpy(), ids_cpu)
    assert len(held) == 4000 and len(data.train_ids) == 36000
    w, _ = restore(run / "W", device)
    data.states = w.normalize(data.states)
    if stage == "compare":
        noz, _ = restore(run / "W_noz", device)
        for name, value in w.named_buffers():
            torch.testing.assert_close(value, noz.get_buffer(name), atol=0, rtol=0)
        assert not torch.count_nonzero(noz.dynamics[0].weight[:, -cfg["latent"]:])
        compare(run, data, held, source, meta["sources"], (w, noz), cfg)
    else:
        checks(run, data, held, source, meta["sources"], w, cfg, meta["constants"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("compare", "checks"))
    parser.add_argument("data", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(args.stage, args.data, args.run, args.device)
