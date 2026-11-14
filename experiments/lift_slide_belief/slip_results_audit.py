"""Independently verify saved paired-study tables and render their readout intervals."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .collect import load_rows, physics
from .model import load_world_model
from .slip_data import project_rows


ARMS = ("fixed_grip", "random_grip")
LEARNING = ("lifts_only", "slides_only", "both")
FORECAST_CONDITIONS = ("train_mean", *LEARNING, "other_both", "shove_ceiling")
SAME_CONDITIONS = ("train_mean", "own_task", "other_cube_same_task", "both")
SAME_METRICS = ("normalized_mse", "base43_normalized_mse", "cube_xy_rmse_mm", "cube_xyz_rmse_mm")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def indexed(rows, names):
    result = {}
    for row in rows:
        key = tuple(row[name] for name in names)
        if key in result:
            raise ValueError(f"Duplicate result key {key}")
        result[key] = row
    return result


def assert_number(actual, expected, *, rtol=1e-9, atol=1e-10):
    if expected is None:
        if actual not in ("", None):
            raise AssertionError(f"Unexpected score {actual}")
    else:
        np.testing.assert_allclose(float(actual), expected, rtol=rtol, atol=atol)


def check_readouts(predictions, table, comparisons, by_seed, gates, seeds, cube_ids):
    lookup = indexed(predictions, ("arm", "seed", "cube_id", "condition"))
    expected = {(arm, str(seed), str(cube), condition) for arm in ARMS for seed in seeds
                if gates[arm][str(seed)]["passed"] for cube in cube_ids for condition in LEARNING}
    if set(lookup) != expected:
        raise ValueError("Prediction table omits, duplicates, or adds a cube/seed/condition")
    seed_table = indexed(by_seed, ("arm", "seed", "condition", "quantity"))
    expected_seed_rows = set(itertools.product(ARMS, map(str, seeds), LEARNING, ("mass", "friction")))
    if set(seed_table) != expected_seed_rows:
        raise ValueError("Failed or passing seeds are missing from the by-seed readout table")
    common = [seed for seed in seeds if all(gates[arm][str(seed)]["passed"] for arm in ARMS)]
    grouped = indexed(table, ("arm", "condition"))
    changes = indexed(comparisons, ("condition", "quantity"))
    assert set(grouped) == set(itertools.product(ARMS, LEARNING))
    assert set(changes) == set(itertools.product(LEARNING, ('mass', 'friction')))
    draws = np.random.default_rng(7741).integers(len(cube_ids), size=(2000, len(cube_ids)))
    metrics_checked = 0
    for condition in LEARNING:
        for quantity in ("mass", "friction"):
            scale = 1000. if quantity == "mass" else 1.
            mae_key = "mass_mae_g" if quantity == "mass" else "friction_mae"
            bootstraps, points = {}, {}
            for arm in ARMS:
                saved = grouped[arm, condition]
                assert int(saved['requested_seeds']) == len(seeds)
                assert int(saved["paired_seed_count"]) == len(common)
                assert saved["paired_seeds"] == "|".join(map(str, common))
                assert int(saved["test_cubes"]) == (len(cube_ids) if common else 0)
                for seed in seeds:
                    seed_row = seed_table[arm, str(seed), condition, quantity]
                    passed = gates[arm][str(seed)]["passed"]
                    assert (seed_row["gate_passed"] == "True") == passed
                    assert int(seed_row['test_cubes']) == (len(cube_ids) if passed else 0)
                    if not passed:
                        assert_number(seed_row["r2"], None)
                        assert_number(seed_row["mae"], None)
                        continue
                    entries = [lookup[arm, str(seed), str(cube), condition] for cube in cube_ids]
                    truth = np.array([float(row["true_" + quantity]) for row in entries])
                    pred = np.array([float(row["predicted_" + quantity]) for row in entries])
                    r2 = 1 - np.sum((pred-truth)**2) / np.sum((truth-truth.mean())**2)
                    mae = scale*np.mean(np.abs(pred-truth))
                    assert_number(seed_row["r2"], r2)
                    assert_number(seed_row["mae"], mae)
                if not common:
                    continue
                entries = [[lookup[arm, str(seed), str(cube), condition] for cube in cube_ids] for seed in common]
                truth = np.array([[float(row["true_" + quantity]) for row in group] for group in entries])
                pred = np.array([[float(row["predicted_" + quantity]) for row in group] for group in entries])
                if arm == ARMS[0]:
                    reference = truth
                else:
                    np.testing.assert_array_equal(truth, reference)
                point_r2 = (1 - np.sum((pred-truth)**2, 1) / np.sum((truth-truth.mean(1,keepdims=True))**2, 1)).mean()
                point_mae = scale*np.abs(pred-truth).mean()
                a, p = truth[:, draws], pred[:, draws]
                bootstrap_r2 = (1 - np.sum((p-a)**2, 2) / np.sum((a-a.mean(2,keepdims=True))**2, 2)).mean(0)
                bootstrap_mae = scale*np.abs(p-a).mean(axis=(0, 2))
                bootstraps[arm] = (bootstrap_r2, bootstrap_mae)
                points[arm] = (point_r2, point_mae)
                for metric, point, samples in ((quantity+"_r2", point_r2, bootstrap_r2),
                                               (mae_key, point_mae, bootstrap_mae)):
                    assert_number(saved[metric], point)
                    low, high = np.quantile(samples, [.025, .975])
                    assert_number(saved[metric+"_ci95_low"], low)
                    assert_number(saved[metric+"_ci95_high"], high)
                    metrics_checked += 3
            if common:
                saved = changes[condition, quantity]
                for index, metric, sign in ((0, "r2_gain", 1), (1, "mae_reduction", -1)):
                    value = sign*(points[ARMS[1]][index]-points[ARMS[0]][index])
                    samples = sign*(bootstraps[ARMS[1]][index]-bootstraps[ARMS[0]][index])
                    assert_number(saved[metric], value)
                    for bound, number in zip(("low", "high"), np.quantile(samples, [.025, .975])):
                        assert_number(saved[metric+"_ci95_"+bound], number)
                    metrics_checked += 3
    return common, dict(prediction_rows=len(predictions), by_seed_rows=len(by_seed),
                        complete_cube_ids=cube_ids, scalar_table_and_interval_checks=metrics_checked,
                        common_passing_seeds=common)


def check_same_task_tables(rows, table, comparisons, common):
    lookup = indexed(rows, ("arm", "seed", "cube_id", "task", "subset", "condition"))
    cube_ids = sorted({int(row["cube_id"]) for row in rows})
    grouped = indexed(table, ("arm", "task", "subset", "condition"))
    contrasts = indexed(comparisons, ("arm", "task", "subset", "metric", "baseline"))
    rng = np.random.default_rng(8831)
    checked = 0
    for arm, task, subset in itertools.product(ARMS, ("lift", "slow_slide"), ("active", "active_moving")):
        valid = [cube for cube in cube_ids if common and all(
            lookup[arm, str(seed), str(cube), task, subset, condition]["normalized_mse"] != ""
            for seed in common for condition in SAME_CONDITIONS)]
        draws = rng.integers(len(valid), size=(2000, len(valid))) if valid else None
        for condition in SAME_CONDITIONS:
            saved = grouped[arm, task, subset, condition]
            assert int(saved["scored_cubes"]) == len(valid)
            assert int(saved["paired_seed_count"]) == len(common)
        for metric in SAME_METRICS:
            values = {}
            for condition in SAME_CONDITIONS:
                saved = grouped[arm, task, subset, condition]
                if not valid:
                    assert_number(saved[metric], None)
                    continue
                # Averaging seeds per cube before resampling preserves the cube cluster.
                by_cube = np.array([[float(lookup[arm, str(seed), str(cube), task, subset, condition][metric])
                                     for cube in valid] for seed in common]).mean(0)
                values[condition] = by_cube
                assert_number(saved[metric], by_cube.mean())
                for bound, number in zip(("low", "high"), np.quantile(by_cube[draws].mean(1), [.025, .975])):
                    assert_number(saved[metric+"_ci95_"+bound], number)
                checked += 3
            for baseline in ("train_mean", "other_cube_same_task"):
                saved = contrasts[arm, task, subset, metric, baseline]
                if not valid:
                    assert_number(saved["improvement"], None)
                    continue
                differences = values[baseline]-values["own_task"]
                assert_number(saved["improvement"], differences.mean())
                assert_number(saved["relative_gain"], differences.mean()/values[baseline].mean())
                for bound, number in zip(("low", "high"), np.quantile(differences[draws].mean(1), [.025, .975])):
                    assert_number(saved["improvement_ci95_"+bound], number)
                checked += 4
    return dict(rows=len(rows), scalar_table_and_interval_checks=checked)


@torch.no_grad()
def check_banks_and_predictions(study, protocol, gates, seeds, predictions, forecasts, same_tasks):
    cube_ids = list(range(200, 230))
    prediction_index = indexed(predictions, ("arm", "seed", "cube_id", "condition"))
    forecast_index = indexed(forecasts, ("arm", "seed", "cube_id", "condition"))
    same_index = indexed(same_tasks, ("arm", "seed", "cube_id", "task", "subset", "condition"))
    max_absolute = max_relative = 0.
    metric_differences, initial_tolerance_exceptions = {}, []
    comparisons = 0
    restored = 0
    restored_hashes = {}
    test_arrays = {}
    for arm in ARMS:
        data = Path(protocol["data"][arm])
        cfg = json.loads((data/"dataset.json").read_text())
        rows = project_rows(load_rows(data, "test"))
        test_arrays[arm] = rows
        for seed in seeds:
            if not gates[arm][str(seed)]["passed"]:
                continue
            folder = study/arm/f"seed_{seed}"
            model = load_world_model(folder/"model.pt", device="cpu")
            with np.load(folder/"diagnostic_beliefs.npz", allow_pickle=False) as archive:
                bank = {key:archive[key] for key in archive.files}
            restored_hashes[f'{arm}/seed_{seed}'] = {
                name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                for name in ('model.pt','diagnostic_beliefs.npz')}
            np.testing.assert_array_equal(bank["cube_id"], cube_ids)
            np.testing.assert_array_equal(model.cube_ids, np.arange(200))
            np.testing.assert_array_equal(bank["prior_mean"], model.prior_mean)
            np.testing.assert_array_equal(bank["prior_sd"], model.prior_sd)
            assert set(bank['donor_cube_id'].tolist()) == set(cube_ids)
            assert np.all(bank['donor_cube_id'] != bank['cube_id'])
            for condition in (*LEARNING, 'shove_ceiling'):
                assert bank['z_'+condition].shape == (30, 4)
                assert np.isfinite(bank['z_'+condition]).all()
            # Rebuild only the post-hoc straight-line decoder from training-cube labels.
            target = np.array([physics(cfg["seed"], int(cube), cfg) for cube in model.cube_ids])
            decoder = np.linalg.lstsq(np.column_stack((model.train_z, np.ones(len(target)))), target, rcond=None)[0]
            np.testing.assert_allclose(bank["linear_decoder"], decoder, rtol=1e-8, atol=1e-9)
            index = {int(cube):i for i,cube in enumerate(bank["cube_id"])}
            for cube in cube_ids:
                i = index[cube]
                expected_truth = physics(cfg["seed"], cube, cfg)
                for condition in LEARNING:
                    saved = prediction_index[arm, str(seed), str(cube), condition]
                    vector = bank["z_"+condition][i]
                    np.testing.assert_array_equal([float(saved[f"z_{j}"]) for j in range(4)], vector)
                    decoded = np.append(vector, 1.) @ decoder
                    for q, value, truth in zip(("mass", "friction"), decoded, expected_truth):
                        assert_number(saved["predicted_"+q], value, rtol=1e-8, atol=1e-9)
                        assert_number(saved["true_"+q], truth)
            for task, task_name in ((0, "lift"), (1, "slow_slide"), (2, "shove")):
                eligible = (rows["task"] == task)
                if task == 2:
                    eligible &= ((rows["episode_id"] < 5) & rows["cube_moving"]
                                 & (rows["step"] >= rows["force_start_step"])
                                 & (rows["step"] <= rows["stop_step"]))
                    conditions = FORECAST_CONDITIONS
                else:
                    eligible &= (rows["episode_id"] >= 5) & (rows["phase"] >= 1)
                    conditions = SAME_CONDITIONS
                selected = np.flatnonzero(eligible)
                state = torch.as_tensor(rows["state"][selected], dtype=torch.float32)
                action = torch.as_tensor(rows["action"][selected], dtype=torch.float32)
                following = torch.as_tensor(rows["next_state"][selected], dtype=torch.float32)
                inputs, targets = model.model.normalize(torch.cat((state, action), 1), following-state)
                task_context = "lifts_only" if task == 0 else "slides_only"
                row_cubes = rows["cube_id"][selected]
                for condition in conditions:
                    latent = []
                    for cube in row_cubes:
                        i = index[int(cube)]
                        if condition == "train_mean":
                            z = bank["prior_mean"]
                        elif condition in ("other_both", "other_cube_same_task"):
                            donor = int(bank["donor_cube_id"][i])
                            assert donor != cube
                            z = bank["z_"+("both" if task == 2 else task_context)][index[donor]]
                        else:
                            name = task_context if condition == "own_task" else condition
                            z = bank["z_"+name][i]
                        latent.append(z)
                    output = model.model(inputs, torch.as_tensor(np.stack(latent), dtype=torch.float32))
                    squared = (output-targets).double().square()
                    physical = squared * model.model.delta_sd.double().square()
                    for cube in cube_ids:
                        subsets = ("moving",) if task == 2 else ("active", "active_moving")
                        for subset in subsets:
                            mask = row_cubes == cube
                            if subset == "active_moving":
                                mask &= rows["cube_moving"][selected]
                            if task == 2:
                                saved = forecast_index[arm,str(seed),str(cube),condition]
                                expected_donor = str(bank['donor_cube_id'][index[cube]]) if condition == 'other_both' else ''
                                count_field = "moving_steps"
                                specs = dict(shove_error=None, shove_base43_error=None, cube_xy_rmse_mm=None,
                                             cube_velocity_rmse_mm_s=None)
                            else:
                                saved = same_index[arm,str(seed),str(cube),task_name,subset,condition]
                                assert saved["support_episodes"] == "0|1|2|3|4"
                                assert saved["prove_episodes"] == "5|6|7|8|9"
                                expected_donor = str(bank['donor_cube_id'][index[cube]]) if condition == 'other_cube_same_task' else ''
                                count_field = "scored_steps"
                                specs = {key:None for key in SAME_METRICS}
                            assert saved['donor_cube_id'] == expected_donor
                            assert int(saved[count_field]) == int(mask.sum())
                            if mask.any():
                                err, raw = squared[mask], physical[mask]
                                if task == 2:
                                    specs.update(shove_error=err.mean().item(), shove_base43_error=err[:,:43].mean().item(),
                                        cube_xy_rmse_mm=1000*raw[:,19:21].sum(1).mean().sqrt().item(),
                                        cube_velocity_rmse_mm_s=1000*raw[:,28:31].sum(1).mean().sqrt().item())
                                else:
                                    specs.update(normalized_mse=err.mean().item(), base43_normalized_mse=err[:,:43].mean().item(),
                                        cube_xy_rmse_mm=1000*raw[:,19:21].sum(1).mean().sqrt().item(),
                                        cube_xyz_rmse_mm=1000*raw[:,19:22].sum(1).mean().sqrt().item())
                            for metric, value in specs.items():
                                try:
                                    assert_number(saved[metric], value, rtol=.005, atol=1e-5)
                                except AssertionError as error:
                                    raise AssertionError(f'{arm} seed{seed} cube{cube} {task_name} {subset} '
                                                         f'{condition} {metric}: {error}') from error
                                if value is not None:
                                    absolute = abs(float(saved[metric])-value)
                                    relative = absolute/max(abs(float(saved[metric])), 1e-10)
                                    max_absolute = max(max_absolute, absolute)
                                    max_relative = max(max_relative, relative)
                                    maximum = metric_differences.setdefault(metric, dict(absolute=0., relative=0.))
                                    maximum['absolute'] = max(maximum['absolute'], absolute)
                                    maximum['relative'] = max(maximum['relative'], relative)
                                    if not np.isclose(float(saved[metric]), value, rtol=.002, atol=1e-5):
                                        initial_tolerance_exceptions.append(dict(arm=arm, seed=seed, cube_id=cube,
                                            task=task_name, subset=subset, condition=condition, metric=metric,
                                            saved=float(saved[metric]), replayed=value, relative_difference=relative))
                                comparisons += 1
            restored += 1
            print(f"Independent score replay {arm} seed {seed} passed", flush=True)
    fixed, random = test_arrays[ARMS[0]],test_arrays[ARMS[1]]
    for key in ("state", "action", "next_state", "phase", "cube_moving"):
        np.testing.assert_array_equal(fixed[key][fixed["task"]==2], random[key][random["task"]==2])
    return dict(restored_models=restored, complete_cube_ids=cube_ids, checks=comparisons,
                checkpoint_and_bank_sha256=restored_hashes,
                saved_latents_and_train_only_linear_decoders_verified=True,
                same_task_support_ids=list(range(5)), same_task_evaluation_ids=list(range(5,10)),
                actual_episode_mask_counts_verified=True, shared_shove_transitions_exact=True,
                replay_device="cpu", study_device=protocol["device"],
                score_tolerance=dict(relative=.005,absolute=.00001),
                original_relative_tolerance=.002,
                original_tolerance_exception_count=len(initial_tolerance_exceptions),
                original_tolerance_exceptions=initial_tolerance_exceptions,
                numerical_note='Study CUDA high precision enables TF32. CPU highest precision differs slightly. '
                    'The initial 0.2% replay tolerance was increased to 0.5% after an isolated high/highest '
                    'GPU comparison reproduced the saved/CPU values; no model, data, or scientific metric changed.',
                maximum_differences_by_metric=metric_differences,
                maximum_absolute_score_difference=max_absolute, maximum_relative_score_difference=max_relative)


def readout_figure(table, output, common):
    lookup = indexed(table, ("arm", "condition"))
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), layout="constrained")
    panels = (("mass_r2", "Mass readout R²"), ("friction_r2", "Friction readout R²"),
              ("mass_mae_g", "Mass MAE (g)"), ("friction_mae", "Friction MAE"))
    x = np.arange(3)
    for ax, (metric,label) in zip(axes.flat, panels):
        for arm, offset, color, name in (("fixed_grip",-.1,"#9b773e","Fixed grip"),
                                         ("random_grip",.1,"#227a79","Randomized grip")):
            if common:
                mean = np.array([float(lookup[arm,c][metric]) for c in LEARNING])
                low = np.array([float(lookup[arm,c][metric+"_ci95_low"]) for c in LEARNING])
                high = np.array([float(lookup[arm,c][metric+"_ci95_high"]) for c in LEARNING])
                # Bootstrap percentile intervals need not contain the point estimate.
                ax.vlines(x+offset, low, high, color=color, linewidth=1.5)
                ax.scatter(x+offset, mean, color=color, s=40, label=name, zorder=3)
                ax.plot(x+offset, mean, color=color, alpha=.6, linewidth=1)
            else:
                ax.text(.5,.5,"No common passing seeds",transform=ax.transAxes,ha="center")
        ax.set(xticks=x, xticklabels=["Five lifts","Five slides","Five lifts +\nfive slides"], ylabel=label)
        if metric.endswith("r2"):
            ax.axhline(0,color="#89949c",linestyle="--",linewidth=.7)
        ax.grid(axis="y",alpha=.2)
        ax.spines[["right","top"]].set_visible(False)
    axes[0,0].legend(frameon=False)
    fig.suptitle("Latent readouts after matched fixed and randomized grip experience")
    fig.supxlabel(f"Means across {len(common)} common passing seeds. Intervals are paired whole-cube 95% bootstrap intervals.\n"
                  "All 30 held-out cubes included; linear decoders fitted only on training-cube latent tables.",fontsize=9)
    for suffix in ("png","pdf"):
        fig.savefig(output/f"readout_comparison.{suffix}",dpi=180)
    plt.close(fig)


def run(study):
    protocol = json.loads((study/"protocol.json").read_text())
    summary = json.loads((study/"summary.json").read_text())
    gates = json.loads((study/"validation.json").read_text())
    seeds = protocol["optimizer_seeds"]
    assert seeds == [0,1,2,3,4]
    assert summary['requested_seeds'] == seeds
    for arm in ARMS:
        assert set(gates[arm]) == set(map(str,seeds))
        for seed in seeds:
            checkpoint = study/arm/f"seed_{seed}"/"model.pt"
            assert checkpoint.exists()
            local = json.loads(checkpoint.with_name("validation.json").read_text())
            assert local["passed"] == gates[arm][str(seed)]["passed"]
    for path, expected in protocol["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    for arm in ARMS:
        path = Path(protocol['data'][arm])/'dataset.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol['dataset_manifest_sha256'][arm]
    predictions = read_csv(study/"readout_predictions.csv")
    table = read_csv(study/"readout_table.csv")
    comparisons = read_csv(study/"paired_readout_comparison.csv")
    same = read_csv(study/"same_task.csv")
    forecast = read_csv(study/"forecast.csv")
    common, readout_check = check_readouts(predictions,table,comparisons,read_csv(study/"readout_by_seed.csv"),
                                           gates,seeds,list(range(200,230)))
    assert common == summary["common_passing_seeds"]
    same_check = check_same_task_tables(same,read_csv(study/"same_task_summary.csv"),
                                       read_csv(study/"same_task_comparisons.csv"),common)
    for arm in ARMS:
        passed = [str(seed) for seed in seeds if gates[arm][str(seed)]["passed"]]
        expected = set(itertools.product((arm,),passed,map(str,range(200,230)),("lift","slow_slide"),
                                        ("active","active_moving"),SAME_CONDITIONS))
        actual = set(indexed([r for r in same if r["arm"]==arm],
                             ("arm","seed","cube_id","task","subset","condition")))
        assert actual == expected
        forecast_expected = set(itertools.product((arm,),passed,map(str,range(200,230)),FORECAST_CONDITIONS))
        forecast_actual = set(indexed([r for r in forecast if r['arm']==arm],('arm','seed','cube_id','condition')))
        assert forecast_actual == forecast_expected
    retrieval = json.loads((study/'retrieval_check.json').read_text())
    for arm in ARMS:
        expected_seeds = {str(seed) for seed in seeds if gates[arm][str(seed)]['passed']}
        assert set(retrieval[arm]) == expected_seeds
        for saved in retrieval[arm].values():
            assert saved['passed'] and saved['checkpoint_and_beliefs_reloaded']
            assert (saved['cube_ids'],saved['decoded_rows'],saved['forecast_rows'],saved['same_task_rows']) == (30,90,180,480)
    forecast_summary = indexed(read_csv(study/"forecast_summary.csv"),("arm","condition"))
    for arm,condition in itertools.product(ARMS,FORECAST_CONDITIONS):
        saved = forecast_summary[arm,condition]
        selected = [r for r in forecast if r["arm"]==arm and r["condition"]==condition and int(r["seed"]) in common]
        assert int(saved["cube_seed_scores"]) == len(common)*30
        valid = [r for r in selected if r["shove_error"] != ""]
        assert int(saved["valid_scores"]) == len(valid)
        for metric in ("shove_error","shove_base43_error","cube_xy_rmse_mm","cube_velocity_rmse_mm_s"):
            seed_means = [np.mean([float(r[metric]) for r in valid if int(r["seed"])==seed]) for seed in common]
            assert_number(saved[metric],np.mean(seed_means) if seed_means else None)
    replay = check_banks_and_predictions(study,protocol,gates,seeds,predictions,forecast,same)
    result = dict(passed=True,requested_seeds=seeds,common_passing_seeds=common,
        failed_seeds={arm:[seed for seed in seeds if not gates[arm][str(seed)]["passed"]] for arm in ARMS},
        source_hashes_unchanged=True,readouts=readout_check,same_tasks=same_check,frozen_replay=replay,
        saved_retrieval_reports_verified=True,
        numerical_precision_check=json.loads((study/'gpu_precision_check.json').read_text()),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifact_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(study.glob('*.csv'))},
        normalized_mse_comparison="Within each arm, using its own training scales.",
        physical_metrics="Euclidean RMS component errors after inverse scaling; position mm and velocity mm/s.",
        coverage_note="All 30 test cubes, including 203 and 224, retained. Acquisition coverage is in independent_data_audit.json.",
        figure="readout_comparison.png")
    assert result['failed_seeds'] == summary['failed_seeds']
    readout_figure(table,study,common)
    (study/"independent_results_audit.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.study)
