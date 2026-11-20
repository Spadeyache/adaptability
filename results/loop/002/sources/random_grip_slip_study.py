"""Compare matched fixed and randomized grip data using the same 51-state model."""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .__main__ import training_digest, write_json
from .collect import load_rows
from .evaluation import _cube_groups, _episode_rows, _episodes, _linear_decoder, _physics, _prove_episodes
from .model import TrainConfig, _transitions, fit_world_model, load_world_model
from .oracle_diagnostics import write_csv


ARMS = ('fixed_grip', 'random_grip')
LEARNING = ('lifts_only', 'slides_only', 'both')
CONDITIONS = ('train_mean', *LEARNING, 'other_both', 'shove_ceiling')
PROJECTION = dict(base_state=[0, 43], relative_position=[43, 46], relative_velocity=[46, 49],
                  finger_position_error=[49, 51])
MATCHED_FIELDS = ('cube_id', 'task', 'episode_id', 'step', 'true_mass', 'true_friction',
                  'reset_seed', 'motion_direction', 'lift_speed_m_s', 'lift_height_m', 'slide_speed_m_s')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_pair(fixed, random):
    """Check matched exogenous draws while allowing grip-dependent trajectories."""
    for key in MATCHED_FIELDS:
        if key not in fixed or key not in random:
            raise ValueError(f'Missing matching field {key}')
        np.testing.assert_array_equal(fixed[key], random[key], err_msg=f'Paired {key} differs')
    shoves = fixed['task'] == 2
    for key in ('state', 'action', 'next_state', 'cube_moving', 'force_start_step', 'stop_step'):
        np.testing.assert_array_equal(fixed[key][shoves], random[key][shoves],
                                      err_msg=f'The shared shove observations differ in {key}')
    for rows in (fixed, random):
        if rows['state'].shape[1] != 51 or rows['next_state'].shape != rows['state'].shape:
            raise ValueError('Both arms require the same 51-dimensional observed state')
        if not np.array_equal(np.unique(rows['task']), [0, 1, 2]):
            raise ValueError('Both arms require all three motion tasks')
        for cube_indices in _cube_groups(rows).values():
            for task in range(3):
                _episodes(rows, cube_indices, task)


def exposure(rows):
    keys = np.column_stack((rows['cube_id'], rows['task'], rows['episode_id']))
    episodes, first = np.unique(keys, axis=0, return_index=True)
    lift = first[episodes[:, 1] == 0]
    result = dict(cubes=len(np.unique(rows['cube_id'])), episodes=len(episodes), rows=len(rows['state']),
                  episodes_by_task={str(task):int(np.sum(episodes[:, 1] == task)) for task in range(3)})
    if 'grasp_acquired' in rows and 'slip_after_acquisition' in rows:
        acquired = np.asarray(rows['grasp_acquired'][lift], dtype=bool)
        slipped = np.asarray(rows['slip_after_acquisition'][lift], dtype=bool)
        rotated = np.asarray(rows['rotating_loss_after_acquisition'][lift], dtype=bool)
        if np.any((slipped | rotated) & ~acquired) or np.any(slipped & rotated):
            raise ValueError('Slip-after-acquisition flag without acquisition')
        result['lift_outcomes'] = dict(failed_pickup=int((~acquired).sum()),
            slip_after_acquisition=int(slipped.sum()), rotating_loss_after_acquisition=int(rotated.sum()),
            held_after_acquisition=int((acquired & ~slipped & ~rotated).sum()))
    return result


@torch.no_grad()
def transition_errors(model, rows, indices, z):
    inputs, target = model.model.normalize(*_transitions(_episode_rows(rows, indices), model.device))
    predicted = model.model(inputs, model._latent(z).expand(len(inputs), -1))
    squared = (predicted - target).double().square()
    if not torch.isfinite(squared).all():
        raise ValueError('Nonfinite one-step prediction')
    physical_squared = squared * model.model.delta_sd.double().square()
    return squared, physical_squared


def shove_errors(model, rows, indices, z):
    moving = (rows['cube_moving'][indices]
              & (rows['step'][indices] >= rows['force_start_step'][indices])
              & (rows['step'][indices] <= rows['stop_step'][indices]))
    selected = indices[moving]
    if not len(selected):
        return dict(shove_error=None, shove_base43_error=None, cube_xy_rmse_mm=None,
                    cube_velocity_rmse_mm_s=None, moving_steps=0)
    squared, physical_squared = transition_errors(model, rows, selected, z)
    return dict(shove_error=squared.mean().item(), shove_base43_error=squared[:, :43].mean().item(),
                cube_xy_rmse_mm=1000 * physical_squared[:, 19:21].sum(1).mean().sqrt().item(),
                cube_velocity_rmse_mm_s=1000 * physical_squared[:, 28:31].sum(1).mean().sqrt().item(),
                moving_steps=len(selected))


def evaluate_same_tasks(model, test, *, seed, bank):
    """Use episodes 5–9 to test beliefs fitted only to episodes 0–4."""
    groups = _cube_groups(test)
    ids = bank['cube_id'].tolist()
    donor_index = {cube:index for index,cube in enumerate(ids)}
    records = []
    for index, cube in enumerate(ids):
        for task, name, learned in ((0, 'lift', 'lifts_only'), (1, 'slow_slide', 'slides_only')):
            prove = np.concatenate(_episodes(test, groups[cube], task)[5:])
            active = prove[test['phase'][prove] >= 1]
            donor = int(bank['donor_cube_id'][index])
            contexts = dict(train_mean=bank['prior_mean'], own_task=bank['z_'+learned][index],
                            other_cube_same_task=bank['z_'+learned][donor_index[donor]],
                            both=bank['z_both'][index])
            for condition, latent in contexts.items():
                errors = transition_errors(model, test, active, latent) if len(active) else None
                masks = dict(active=np.ones(len(active), dtype=bool),
                             active_moving=np.asarray(test['cube_moving'][active], dtype=bool))
                for subset, mask in masks.items():
                    metrics = dict(normalized_mse=None, base43_normalized_mse=None,
                                   cube_xy_rmse_mm=None, cube_xyz_rmse_mm=None)
                    if mask.any():
                        normalized, physical = (value[torch.as_tensor(mask, device=value.device)] for value in errors)
                        metrics.update(normalized_mse=normalized.mean().item(),
                            base43_normalized_mse=normalized[:, :43].mean().item(),
                            cube_xy_rmse_mm=1000*physical[:, 19:21].sum(1).mean().sqrt().item(),
                            cube_xyz_rmse_mm=1000*physical[:, 19:22].sum(1).mean().sqrt().item())
                    records.append(dict(seed=seed, cube_id=int(cube), task=name, subset=subset,
                        condition=condition, support_episodes='0|1|2|3|4', prove_episodes='5|6|7|8|9',
                        scored_steps=int(mask.sum()), donor_cube_id=donor if condition=='other_cube_same_task' else None,
                        **metrics))
    return records


def evaluate_readouts(model, train, test, *, seed, fit_steps=100):
    """Fit final task-specific beliefs, then decode physics and score held-out shoves."""
    if not model.validation['passed']:
        raise ValueError('The unchanged shuffled-z validation gate must pass')
    groups = _cube_groups(test)
    ids = np.array(sorted(groups))
    if np.intersect1d(ids, model.cube_ids).size:
        raise ValueError('Held-out cube IDs overlap the training latent table')
    supports = {condition: [] for condition in (*LEARNING, 'shove_ceiling')}
    for cube in ids:
        lifts = _episodes(test, groups[cube], 0)[:5]
        slides = _episodes(test, groups[cube], 1)[:5]
        both = [episode for pair in zip(lifts, slides) for episode in pair]
        ceiling = _episodes(test, groups[cube], 2)[5:]
        for condition, episodes in zip(supports, (lifts, slides, both, ceiling)):
            supports[condition].append(_episode_rows(test, np.concatenate(episodes)))
    beliefs = {condition:model.fit_many_z(observations, steps=fit_steps)
               for condition, observations in supports.items()}
    order = np.random.default_rng(seed).permutation(len(ids))
    donors = np.empty_like(order)
    donors[order] = np.roll(order, 1)
    if len(ids) < 2 or np.any(donors == np.arange(len(ids))):
        raise ValueError('A wrong-cube baseline requires at least two distinct held-out cubes')
    decoder = _linear_decoder(model, train)
    predictions, forecasts = [], []
    for index, cube in enumerate(ids):
        truth = _physics(test, groups[cube])
        for condition in LEARNING:
            z = beliefs[condition][index]
            decoded = np.append(z, 1.) @ decoder
            predictions.append(dict(seed=seed, cube_id=int(cube), condition=condition,
                true_mass=float(truth[0]), true_friction=float(truth[1]),
                predicted_mass=float(decoded[0]), predicted_friction=float(decoded[1]),
                **{f'z_{i}':float(value) for i,value in enumerate(z)}))
        prove = np.concatenate(_prove_episodes(test, groups[cube]))
        contexts = dict(train_mean=model.prior_mean,
                        **{condition:beliefs[condition][index] for condition in (*LEARNING, 'shove_ceiling')},
                        other_both=beliefs['both'][donors[index]])
        for condition in CONDITIONS:
            forecasts.append(dict(seed=seed, cube_id=int(cube), condition=condition,
                donor_cube_id=int(ids[donors[index]]) if condition == 'other_both' else None,
                **shove_errors(model, test, prove, contexts[condition])))
    bank = dict(cube_id=ids, prior_mean=model.prior_mean, prior_sd=model.prior_sd,
                donor_cube_id=ids[donors], linear_decoder=decoder, **{'z_'+key:value for key,value in beliefs.items()})
    return predictions, forecasts, bank


def seed_readouts(predictions, arm, seed, passed):
    records = []
    for condition in LEARNING:
        selected = [row for row in predictions if row['condition'] == condition]
        for quantity in ('mass', 'friction'):
            record = dict(arm=arm, seed=seed, condition=condition, quantity=quantity,
                          gate_passed=passed, test_cubes=len(selected), r2=None, mae=None,
                          mae_unit='g' if quantity == 'mass' else 'friction_coefficient')
            if selected:
                actual = np.array([row['true_'+quantity] for row in selected])
                predicted = np.array([row['predicted_'+quantity] for row in selected])
                scale = 1000 if quantity == 'mass' else 1
                denominator = np.sum((actual-actual.mean())**2)
                record.update(r2=float(1-np.sum((predicted-actual)**2)/denominator) if denominator > 0 else None,
                              mae=float(scale*np.mean(np.abs(predicted-actual))))
            records.append(record)
    return records


def verify_retrieval(model, test, path, predictions, forecasts):
    """Reload final vectors by cube ID and reproduce decoding and prediction scores."""
    with np.load(path, allow_pickle=False) as saved:
        ids = saved['cube_id']
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Duplicate cube IDs in the diagnostic belief bank')
        beliefs = {}
        for condition in (*LEARNING, 'shove_ceiling'):
            values = saved['z_' + condition]
            if values.shape != (len(ids), 4):
                raise ValueError('Diagnostic beliefs must contain one final four-vector per cube')
            beliefs[condition] = dict(zip(ids.tolist(), values.copy()))
        decoder = saved['linear_decoder'].copy()
        prior = saved['prior_mean'].copy()
        donors = dict(zip(ids.tolist(), saved['donor_cube_id'].tolist()))
    np.testing.assert_array_equal(prior, model.prior_mean)
    groups = _cube_groups(test)
    if set(ids.tolist()) != set(groups):
        raise ValueError('Stored cube IDs disagree with the held-out observations')
    largest_difference = 0.
    for row in predictions:
        value = np.append(beliefs[row['condition']][row['cube_id']], 1.) @ decoder
        expected = [row['predicted_mass'], row['predicted_friction']]
        np.testing.assert_allclose(value, expected, rtol=1e-7, atol=1e-10)
        largest_difference = max(largest_difference, float(np.max(np.abs(value - expected))))
    for row in forecasts:
        cube, condition = row['cube_id'], row['condition']
        if condition == 'train_mean':
            latent = prior
        elif condition == 'other_both':
            latent = beliefs['both'][donors[cube]]
        else:
            latent = beliefs[condition][cube]
        prove = np.concatenate(_prove_episodes(test, groups[cube]))
        observed = shove_errors(model, test, prove, latent)
        for metric, value in observed.items():
            if value is None or row[metric] is None:
                if value != row[metric]:
                    raise AssertionError(f'Retrieved {metric} validity changed')
            else:
                np.testing.assert_allclose(value, row[metric], rtol=1e-7, atol=1e-10)
                largest_difference = max(largest_difference, float(abs(value - row[metric])))
    return dict(passed=True, cube_ids=len(ids), decoded_rows=len(predictions), forecast_rows=len(forecasts),
                maximum_absolute_difference=largest_difference, checkpoint_and_beliefs_reloaded=True)


def paired_readout_tables(predictions, gates, seeds, *, bootstrap_draws=2000):
    """Use common passing seeds and resample whole cubes jointly across both arms."""
    common = [seed for seed in seeds if all(gates[arm][str(seed)]['passed'] for arm in ARMS)]
    lookup = {(r['arm'],r['seed'],r['cube_id'],r['condition']):r for r in predictions}
    table, comparisons = [], []
    cube_ids = sorted({r['cube_id'] for r in predictions})
    rng = np.random.default_rng(7741)
    draws = rng.integers(len(cube_ids), size=(bootstrap_draws, len(cube_ids))) if common else None
    for condition in LEARNING:
        records = {arm:dict(arm=arm, condition=condition, requested_seeds=len(seeds), paired_seed_count=len(common),
                           paired_seeds='|'.join(map(str,common)), test_cubes=len(cube_ids) if common else 0)
                   for arm in ARMS}
        for quantity in ('mass', 'friction'):
            estimates, bootstraps = {}, {}
            unit = 'g' if quantity == 'mass' else 'coefficient'
            for arm in ARMS:
                prefix = 'mass_mae_g' if quantity == 'mass' else 'friction_mae'
                values = {quantity+'_r2':None, prefix:None}
                intervals = {key+'_ci95_'+bound:None for key in values for bound in ('low','high')}
                if common:
                    rows = [[lookup[arm,seed,cube,condition] for cube in cube_ids] for seed in common]
                    actual = np.array([[row['true_'+quantity] for row in group] for group in rows])
                    predicted = np.array([[row['predicted_'+quantity] for row in group] for group in rows])
                    if arm == ARMS[0]:
                        reference = actual
                    else:
                        np.testing.assert_array_equal(actual, reference)
                    variance = np.sum((actual-actual.mean(1,keepdims=True))**2, axis=1)
                    scale = 1000 if quantity == 'mass' else 1
                    r2 = 1-np.sum((predicted-actual)**2, axis=1)/variance
                    mae = scale*np.mean(np.abs(predicted-actual),axis=1)
                    values = {quantity+'_r2':float(r2.mean()), prefix:float(mae.mean())}
                    a, p = actual[:,draws], predicted[:,draws]
                    denom = np.sum((a-a.mean(2,keepdims=True))**2,axis=2)
                    b = np.stack(((1-np.sum((p-a)**2,axis=2)/denom).mean(0),
                                  (scale*np.mean(np.abs(p-a),axis=2)).mean(0)),axis=1)
                    if not np.isfinite(b).all():
                        raise ValueError('Degenerate resampled labels in the paired readout bootstrap')
                    for column,key in enumerate(values):
                        lo,hi = np.quantile(b[:,column],[.025,.975])
                        intervals[key+'_ci95_low'],intervals[key+'_ci95_high'] = float(lo),float(hi)
                    estimates[arm],bootstraps[arm] = np.array(list(values.values())),b
                records[arm].update(**values,**intervals)
            comparison = dict(condition=condition,quantity=quantity,paired_seed_count=len(common),
                              r2_gain=None,r2_gain_ci95_low=None,r2_gain_ci95_high=None,
                              mae_reduction=None,mae_unit=unit,mae_reduction_ci95_low=None,mae_reduction_ci95_high=None)
            if common:
                gain = (estimates[ARMS[1]]-estimates[ARMS[0]]) * np.array([1,-1])
                bootstrap = (bootstraps[ARMS[1]]-bootstraps[ARMS[0]]) * np.array([1,-1])
                for column,key in enumerate(('r2_gain','mae_reduction')):
                    lo,hi = np.quantile(bootstrap[:,column],[.025,.975])
                    comparison.update({key:float(gain[column]),key+'_ci95_low':float(lo),key+'_ci95_high':float(hi)})
            comparisons.append(comparison)
        table.extend(records.values())
    return table,comparisons,common


def forecast_table(forecasts, common):
    table = []
    for arm in ARMS:
        for condition in CONDITIONS:
            selected = [row for row in forecasts if row['arm'] == arm and row['condition'] == condition
                        and row['seed'] in common]
            valid = [row for row in selected if row['shove_error'] is not None]
            record = dict(arm=arm, condition=condition, paired_seed_count=len(common),
                          cube_seed_scores=len(selected), valid_scores=len(valid))
            for metric in ('shove_error', 'shove_base43_error', 'cube_xy_rmse_mm', 'cube_velocity_rmse_mm_s'):
                seed_means = [np.mean([row[metric] for row in valid if row['seed'] == seed])
                              for seed in common if any(row['seed'] == seed for row in valid)]
                record[metric] = float(np.mean(seed_means)) if seed_means else None
            table.append(record)
    return table


def same_task_tables(records, common, bootstrap_draws=2000):
    """Pair baseline and own-belief errors on complete test cubes within each arm."""
    metrics = ('normalized_mse', 'base43_normalized_mse', 'cube_xy_rmse_mm', 'cube_xyz_rmse_mm')
    conditions = ('train_mean', 'own_task', 'other_cube_same_task', 'both')
    lookup = {(row['arm'], row['seed'], row['cube_id'], row['task'], row['subset'], row['condition']):row
              for row in records}
    cube_ids = sorted({row['cube_id'] for row in records})
    rng = np.random.default_rng(8831)
    summaries, comparisons = [], []
    for arm in ARMS:
        for task in ('lift', 'slow_slide'):
            for subset in ('active', 'active_moving'):
                eligible = [cube for cube in cube_ids if common and all(
                    lookup[arm,seed,cube,task,subset,condition]['normalized_mse'] is not None
                    for seed in common for condition in conditions)]
                draws = rng.integers(len(eligible), size=(bootstrap_draws, len(eligible))) if eligible else None
                summary = {condition:dict(arm=arm, task=task, subset=subset, condition=condition,
                    primary=subset=='active', paired_seed_count=len(common), total_cubes=len(cube_ids),
                    scored_cubes=len(eligible)) for condition in conditions}
                for metric in metrics:
                    means, intervals = {}, {}
                    for condition in conditions:
                        point, low, high = None, None, None
                        if eligible:
                            values = np.array([[lookup[arm,seed,cube,task,subset,condition][metric]
                                                for cube in eligible] for seed in common])
                            point = float(values.mean())
                            intervals[condition] = values[:,draws].mean(axis=(0,2))
                            low, high = np.quantile(intervals[condition],[.025,.975]).tolist()
                            means[condition] = point
                        summary[condition].update({metric:point, metric+'_ci95_low':low, metric+'_ci95_high':high})
                    for baseline in ('train_mean', 'other_cube_same_task'):
                        record = dict(arm=arm,task=task,subset=subset,primary=subset=='active',metric=metric,
                                      baseline=baseline,paired_seed_count=len(common),scored_cubes=len(eligible),
                                      improvement=None,improvement_ci95_low=None,improvement_ci95_high=None,
                                      relative_gain=None)
                        if eligible:
                            difference = means[baseline]-means['own_task']
                            low, high = np.quantile(intervals[baseline]-intervals['own_task'],[.025,.975])
                            record.update(improvement=difference,improvement_ci95_low=float(low),improvement_ci95_high=float(high),
                                          relative_gain=difference/means[baseline] if means[baseline]>0 else None)
                        comparisons.append(record)
                summaries.extend(summary.values())
    return summaries, comparisons


def run(args):
    from .slip_data import project_rows

    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    torch.set_num_threads(args.threads)
    if device.startswith('cuda'):
        torch.set_float32_matmul_precision('high')
    data_dirs = dict(fixed_grip=args.fixed_data.resolve(), random_grip=args.random_data.resolve())
    manifests = {arm:json.loads((folder/'dataset.json').read_text()) for arm,folder in data_dirs.items()}
    for arm, label in zip(ARMS, ('fixed', 'random')):
        if manifests[arm].get('grip_arm') != label:
            raise ValueError(f'{arm} requires the {label} acquisition dataset')
    protocols = [{key:value for key,value in manifests[arm].items() if key != 'grip_arm'} for arm in ARMS]
    if protocols[0] != protocols[1]:
        raise ValueError('Fixed and randomized acquisition protocols differ beyond grip-arm assignment')
    rows = {arm:{split:project_rows(load_rows(folder, split)) for split in ('train','test')}
            for arm,folder in data_dirs.items()}
    for split in ('train','test'):
        validate_pair(rows[ARMS[0]][split],rows[ARMS[1]][split])
    for arm in ARMS:
        if np.intersect1d(rows[arm]['train']['cube_id'],rows[arm]['test']['cube_id']).size:
            raise ValueError('Train and test cube IDs overlap')
    config = TrainConfig(max_epochs=args.max_epochs,patience=args.patience)
    files = [Path(__file__), *[Path(__file__).with_name(name) for name in
             ('model.py','evaluation.py','slip_data.py','scripted.py','collect.py','__main__.py','oracle_diagnostics.py')]]
    protocol = dict(data={arm:str(path) for arm,path in data_dirs.items()},
        dataset_manifest_sha256={arm:digest(path/'dataset.json') for arm,path in data_dirs.items()},
        transition_sha256={arm:{split:training_digest(data) for split,data in arm_rows.items()} for arm,arm_rows in rows.items()},
        source_sha256={str(path.resolve()):digest(path) for path in files},
        optimizer_seeds=args.seeds,training=asdict(config),fit_steps=args.fit_steps,state_projection=PROJECTION,
        state_dim=51,validation_required_controls=['shuffled'],device=device,torch_version=str(torch.__version__),
        belief_bank_schema='One N-by-4 final z array per condition plus cube IDs, donor IDs, prior and linear decoder. The prefix-bank target-control reader requires a schema adapter.',
        paired_seed_rule='Primary task readout tables use only seeds passing the unchanged gate in both arms',
        normalized_forecast_metrics='Each arm uses its own training delta scales. Compare these metrics among latent conditions within an arm.',
        physical_forecast_metrics='Cube XY and XYZ velocity RMSE are square roots of mean summed squared component errors, in mm and mm/s. Average cube RMSEs within each seed, then average seeds.',
        same_task_protocol=dict(support_episode_ids=list(range(5)),prove_episode_ids=list(range(5,10)),
            primary_slice='All rows with phase >= 1, retaining stationary holds and failed-motion outcomes',
            secondary_slice='Rows with phase >= 1 and cube_moving',
            conditions=['train_mean','own_task','other_cube_same_task','both'],
            physical_position_metric='Euclidean RMS cube XY/XYZ position error in mm; XYZ is the primary physical lift metric',
            comparison_scope='Lift/slide action trajectories differ across arms. Evaluate own-versus-mean/wrong belief effects within each arm. Shared-shove physical errors and true-property readouts support cross-arm comparisons.',
            uncertainty='2000 paired whole-cube bootstrap draws, seed8831, common passing seeds averaged',
            adaptation='Reuse the same final support-fitted beliefs; no further fitting on prove episodes'),
        uncertainty='Paired 95% percentile bootstrap of whole test cubes, averaged over common passing training seeds; 2000 draws, seed7741',
        exposure={arm:{split:exposure(data) for split,data in arm_rows.items()} for arm,arm_rows in rows.items()})
    output = args.output.resolve()
    manifest = output/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('Output folder already contains a different paired diagnostic protocol')
    write_json(manifest,protocol)
    gates = {arm:{} for arm in ARMS}
    for arm in ARMS:
        for seed in args.seeds:
            folder = output/arm/f'seed_{seed}'
            folder.mkdir(parents=True,exist_ok=True)
            checkpoint = folder/'model.pt'
            if checkpoint.exists():
                model = load_world_model(checkpoint,device=device)
            else:
                def log(record):
                    if record['epoch']==1 or record['epoch']%10==0:
                        print(f"{arm} seed{seed} epoch{record['epoch']} validation{record['validation_mse']:.5f}",flush=True)
                model = fit_world_model(rows[arm]['train'],seed=seed,device=device,config=config,log=log)
                model.save(checkpoint)
            write_json(folder/'validation.json',model.validation)
            gates[arm][str(seed)] = {key:value for key,value in model.validation.items() if key not in ('history','heldout_episodes')}
            print(f"{arm} seed{seed} validation gate {'passed' if model.validation['passed'] else 'failed'}",flush=True)
            del model
    write_json(output/'validation.json',gates)
    predictions,forecasts,readouts,same_tasks = [],[],[],[]
    retrieval = {arm:{} for arm in ARMS}
    for arm in ARMS:
        for seed in args.seeds:
            passed = gates[arm][str(seed)]['passed']
            pred, forecast = [],[]
            if passed:
                folder = output/arm/f'seed_{seed}'
                model = load_world_model(folder/'model.pt',device=device)
                pred,forecast,bank = evaluate_readouts(model,rows[arm]['train'],rows[arm]['test'],seed=seed,fit_steps=args.fit_steps)
                same_task = evaluate_same_tasks(model, rows[arm]['test'], seed=seed, bank=bank)
                temporary = folder/'diagnostic_beliefs.tmp'
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream,**bank)
                temporary.replace(folder/'diagnostic_beliefs.npz')
                del model
                restored = load_world_model(folder/'model.pt', device=device)
                retrieval[arm][str(seed)] = verify_retrieval(restored, rows[arm]['test'],
                    folder/'diagnostic_beliefs.npz', pred, forecast)
                with np.load(folder/'diagnostic_beliefs.npz', allow_pickle=False) as saved:
                    restored_bank = {key:saved[key] for key in saved.files}
                repeated = evaluate_same_tasks(restored, rows[arm]['test'], seed=seed, bank=restored_bank)
                if repeated != same_task:
                    raise AssertionError('Reloaded same-task predictions changed')
                retrieval[arm][str(seed)]['same_task_rows'] = len(same_task)
                del restored
                predictions.extend(dict(arm=arm,**record) for record in pred)
                forecasts.extend(dict(arm=arm,**record) for record in forecast)
                same_tasks.extend(dict(arm=arm,**record) for record in same_task)
            readouts.extend(seed_readouts(pred,arm,seed,passed))
    table,comparisons,common = paired_readout_tables(predictions,gates,args.seeds)
    forecasts_summary = forecast_table(forecasts, common)
    same_task_summary,same_task_comparisons = same_task_tables(same_tasks, common)
    if predictions:
        write_csv(output/'readout_predictions.csv',predictions)
        write_csv(output/'forecast.csv',forecasts)
        write_csv(output/'same_task.csv',same_tasks)
    write_csv(output/'readout_by_seed.csv',readouts)
    write_csv(output/'readout_table.csv',table)
    write_csv(output/'paired_readout_comparison.csv',comparisons)
    write_csv(output/'forecast_summary.csv',forecasts_summary)
    write_csv(output/'same_task_summary.csv',same_task_summary)
    write_csv(output/'same_task_comparisons.csv',same_task_comparisons)
    write_json(output/'retrieval_check.json', retrieval)
    write_json(output/'summary.json',dict(requested_seeds=args.seeds,common_passing_seeds=common,
               failed_seeds={arm:[seed for seed in args.seeds if not gates[arm][str(seed)]['passed']] for arm in ARMS},
               readout_table=table,paired_readout_comparison=comparisons,
               forecast_table=forecasts_summary,same_task_summary=same_task_summary,
               same_task_comparisons=same_task_comparisons,exposure=protocol['exposure']))
    print(json.dumps(dict(common_passing_seeds=common,readout_table=table),indent=2),flush=True)
    return 0 if len(common)==len(args.seeds) else 2


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('fixed_data',type=Path)
    cli.add_argument('random_data',type=Path)
    cli.add_argument('output',type=Path)
    cli.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    cli.add_argument('--device',default='auto')
    cli.add_argument('--threads',type=int,default=4)
    cli.add_argument('--max-epochs',type=int,default=1000)
    cli.add_argument('--patience',type=int,default=12)
    cli.add_argument('--fit-steps',type=int,default=100)
    args = cli.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or any(seed<0 for seed in args.seeds):
        cli.error('seeds must be distinct and nonnegative')
    if min(args.threads,args.max_epochs,args.patience,args.fit_steps)<1:
        cli.error('threads, epochs, patience and adaptation steps must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.study.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return run(args)


if __name__=='__main__':
    raise SystemExit(main())
