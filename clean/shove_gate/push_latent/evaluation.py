"""One frozen fresh-cube evaluation with a mandatory C1 stop.

This module never trains dynamics weights or chooses adaptation settings.
Observed test query outcomes cannot enter latent fitting or selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import models

ROOT = Path(__file__).resolve().parent
SEEDS = [101, 211, 307, 401, 503]
TEST_CUBES = list(range(260, 290))
QUERY_EPISODES = list(range(5, 10))
BASELINES = ['plain', 'oracle', 'history']
CONTROLS = ['fitted_z', 'mean_z', 'wrong_z']
GROUPS = ['push', 'moving_coast', 'all_steps']
PRIMARY = 'horizontal_velocity_delta_nmse'
METRICS = [PRIMARY, 'one_step_nmse', 'rollout10_nmse',
           'rollout10_final_nmse', 'stopping_position_error_m']


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def write_atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_manifest(root, manifest):
    for name, expected in manifest.items():
        require(digest(Path(root) / name) == expected, f'Changed sealed file: {name}')


def aggregate(values, row_episode, cube_ids, mask):
    """Equal step within episode, episode within cube, and cube weighting."""
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(np.isfinite(values[mask]).all(), 'Nonfinite predictive error: stop, do not drop rows')
    by_cube = {}
    per_episode = []
    for eid, cid in enumerate(cube_ids):
        chosen = values[(row_episode == eid) & mask]
        if len(chosen):
            mean = float(chosen.mean())
            by_cube.setdefault(int(cid), []).append(mean)
            per_episode.append(dict(episode_row=eid, cube_id=int(cid), mean=mean, starts=len(chosen)))
    per_cube = [dict(cube_id=cid, mean=float(np.mean(v)), episodes=len(v))
                for cid, v in sorted(by_cube.items())]
    return dict(mean=float(np.mean([x['mean'] for x in per_cube])) if per_cube else None,
                eligible_starts=int(mask.sum()), nonfinite_starts=0,
                cubes=len(per_cube), episodes=len(per_episode), per_cube=per_cube,
                per_episode=per_episode,
                missing_cube_ids=sorted(set(map(int, cube_ids)) - set(by_cube)))


def moments(values):
    require(len(values) == 5 and all(v is not None and math.isfinite(v) for v in values),
            'Five finite paired seed values are required')
    a = np.asarray(values, dtype=np.float64)
    return dict(seed_values=a.tolist(), mean=float(a.mean()), sample_sd=float(a.std(ddof=1)))


def comparison(baseline, candidate):
    p = np.asarray(baseline['seed_values']); q = np.asarray(candidate['seed_values'])
    return dict(mean_gap=float(p.mean() - q.mean()),
                relative_mean_improvement=float((p.mean() - q.mean()) / p.mean()) if p.mean() > 0 else None,
                sum_seed_sds=float(p.std(ddof=1) + q.std(ddof=1)),
                beyond_seed_spread=bool(q.mean() + q.std(ddof=1) < p.mean() - p.std(ddof=1)),
                paired_seed_improvements=[float((a-b)/a) if a > 0 else None for a, b in zip(p, q)],
                paired_gap_sample_sd=float((p-q).std(ddof=1)))


def c1_decision(plain, oracle, coverage=True):
    comp = comparison(plain, oracle)
    reduction = plain['mean'] > 0 and oracle['mean'] <= .8 * plain['mean']
    return dict(passed=bool(coverage and reduction and comp['beyond_seed_spread']),
                at_least_twenty_percent=bool(reduction), coverage_complete=bool(coverage),
                plain=plain, oracle=oracle, **comp)


def c2_c3_decisions(plain, oracle, fitted, mean_z, wrong_z):
    denominator = plain['mean'] - oracle['mean']
    require(denominator > 0, 'C2 requires the positive C1 Plain-to-Oracle gap')
    recovery = (plain['mean'] - fitted['mean']) / denominator
    per_seed = [(p-z)/(p-o) if p > o else None for p, o, z in
                zip(plain['seed_values'], oracle['seed_values'], fitted['seed_values'])]
    c2 = dict(passed=bool(recovery >= .5), recovered_fraction=float(recovery),
              plain_to_oracle_gap=denominator, recovered_error=plain['mean']-fitted['mean'],
              per_seed_recovered_fractions=per_seed)
    mean_comparison = comparison(mean_z, fitted)
    wrong_comparison = comparison(wrong_z, fitted)
    c3 = dict(passed=bool(mean_comparison['beyond_seed_spread'] and wrong_comparison['beyond_seed_spread']),
              fitted_vs_mean_z=mean_comparison, fitted_vs_wrong_z=wrong_comparison)
    return c2, c3


def wrong_z_mapping(cube_ids):
    ordered = sorted(map(int, cube_ids))
    require(len(ordered) > 1 and len(set(ordered)) == len(ordered), 'Wrong-z requires unique multiple cubes')
    return {cid: ordered[(i+1) % len(ordered)] for i, cid in enumerate(ordered)}


def make_query_pack(root, normalizer):
    episodes = models.load_episodes(root, TEST_CUBES, QUERY_EPISODES, allow_test=True)
    full = []; force = []; stop_xy = []; release = []; full_lengths = []
    for ep in episodes:
        path = root / 'data/raw' / f"cube_{ep['cube_id']:03d}_episode_{ep['episode_id']:02d}.npz"
        with np.load(path, allow_pickle=False) as raw:
            states = np.asarray(raw['state'], dtype=np.float64)
            actions = np.asarray(raw['action'], dtype=np.float64)
            r = int(raw['release_index']); dt = float(raw['physics_dt_s'])
            require(dt == .001 and r % 5 == 0 and len(actions) == r + 1500,
                    'Unexpected raw timing or preplanned horizon')
            sampled = states[::5].astype(np.float32)
            n = len(ep['state'])
            require(np.array_equal(sampled[:n], ep['state']) and
                    np.array_equal(sampled[1:n+1], ep['next_state']), 'Raw/model state alignment mismatch')
            require(np.array_equal(actions[::5][:n].astype(np.float32), ep['action']), 'Raw/model action mismatch')
            resting = ((np.linalg.norm(states[:, 9:12], axis=1) < .001) &
                       (np.linalg.norm(states[:, 12:15], axis=1) < .05))
            candidates = np.flatnonzero(np.convolve(resting[r+1:].astype(int), np.ones(100, dtype=int), 'valid') == 100)
            require(len(candidates) > 0, 'No confirmed rest in fixed raw horizon')
            stop = r + 1 + int(candidates[0])
            if stop == r+1 and resting[r]: stop = r
            full.append(sampled); force.append(actions[0].astype(np.float32))
            stop_xy.append(states[stop, :2]); release.append(r//5); full_lengths.append(len(sampled)-1)
    lengths = np.asarray([len(e['state']) for e in episodes], dtype=np.int64)
    row_episode = np.repeat(np.arange(len(episodes)), lengths)
    row_step = np.concatenate([np.arange(n) for n in lengths])
    state = np.concatenate([e['state'] for e in episodes])
    nxt = np.concatenate([e['next_state'] for e in episodes])
    action = np.concatenate([e['action'] for e in episodes])
    phase = np.concatenate([e['phase'] for e in episodes])
    moving = (np.linalg.norm(state[:, 9:12], axis=1) > .001) | (np.linalg.norm(state[:, 12:15], axis=1) > .05)
    release = np.asarray(release); full_lengths = np.asarray(full_lengths)
    remaining = full_lengths[row_episode] - row_step
    push_remaining = release[row_episode] - row_step
    masks = dict(push=phase == 'push', moving_coast=(phase == 'coast') & moving,
                 all_steps=np.ones(len(state), dtype=bool))
    rollout_masks = dict(push=masks['push'] & (push_remaining >= 10),
                         moving_coast=masks['moving_coast'] & (remaining >= 10), all_steps=remaining >= 10)
    padded = np.empty((len(full), max(full_lengths)+1, 15), dtype=np.float32)
    for i, trace in enumerate(full):
        padded[i, :len(trace)] = trace; padded[i, len(trace):] = trace[-1]
    return dict(episodes=episodes, state=state, next_state=nxt, action=action,
                history=models.build_history(episodes, normalizer),
                row_episode=row_episode, row_step=row_step,
                cube_ids=np.array([e['cube_id'] for e in episodes]),
                episode_ids=np.array([e['episode_id'] for e in episodes]),
                full_state=padded, full_lengths=full_lengths, force=np.asarray(force),
                stop_xy=np.asarray(stop_xy), remaining=remaining, push_remaining=push_remaining,
                masks=masks, rollout_masks=rollout_masks)


@torch.no_grad()
def predict_batch(model, state, force, context, history, z, norm, push_remaining,
                  remaining, targets10, next_state, stop_xy, primary_only=False):
    """Roll recursively; History receives predicted transitions after the start."""
    prediction = state.clone(); history = history.clone() if history is not None else None
    total10 = torch.zeros(len(state), dtype=torch.float64, device=state.device)
    final10 = total10.clone(); values = {}
    horizon = 1 if primary_only else int(remaining.max().item())
    for step in range(horizon):
        active = remaining > step
        command = torch.where((push_remaining[active] > step)[:, None], force[active], torch.zeros_like(force[active]))
        x = (prediction[active] - norm['state_mean']) / norm['state_std']
        a = (command - norm['action_mean']) / norm['action_std']
        delta = model(x, a, context=context[active] if model.variant == 'oracle' else None,
                      history=history[active] if model.variant == 'history' else None,
                      z=None if z is None else z[active])
        require(bool(torch.isfinite(delta).all()), 'Nonfinite predictive output: protocol stop')
        proposed = prediction[active] + delta * norm['state_std']
        require(bool(torch.isfinite(proposed).all()), 'Nonfinite predicted state: protocol stop')
        if model.variant == 'history':
            token = torch.cat([x, a, delta], dim=-1)
            history[active] = torch.cat([history[active, 1:], token[:, None, :]], dim=1)
        prediction[active] = proposed
        if step == 0:
            predicted_delta = delta[:, 9:11].double() * norm['state_std'][9:11].double()
            true_delta = next_state[:, 9:11].double() - state[:, 9:11].double()
            values[PRIMARY] = ((predicted_delta-true_delta).square().mean(-1) /
                               norm['horizontal_velocity_delta_mse_scale']).cpu().numpy()
            values['one_step_nmse'] = (((prediction-next_state)/norm['state_std']).double().square().mean(-1)).cpu().numpy()
        if not primary_only and step < 10:
            error = ((prediction-targets10[:, step])/norm['state_std']).double().square().mean(-1)
            total10 += error / 10
            if step == 9: final10 = error
    if not primary_only:
        values.update(rollout10_nmse=total10.cpu().numpy(), rollout10_final_nmse=final10.cpu().numpy(),
                      stopping_position_error_m=torch.linalg.vector_norm(prediction[:, :2].double()-stop_xy, dim=-1).cpu().numpy())
    require(all(np.isfinite(v).all() for v in values.values()), 'Nonfinite forecast metric: protocol stop')
    return values


def evaluate_model(root, model, checkpoint, checkpoint_path, pack, properties, device,
                   model_name, z_by_cube=None, primary_only=False, batch_size=2048):
    started = time.perf_counter()
    norm = models.tensor_normalizer(checkpoint['normalizer'], device)
    ep_context = None
    if model.variant == 'oracle':
        ep_context = torch.tensor(np.stack([properties[int(c)] for c in pack['cube_ids']]), device=device)
        ep_context = (ep_context-norm['context_mean'])/norm['context_std']
    names = [PRIMARY, 'one_step_nmse'] if primary_only else METRICS
    values = {name: [] for name in names}
    for start in range(0, len(pack['state']), batch_size):
        end = min(start+batch_size, len(pack['state'])); sl = slice(start, end)
        ep = pack['row_episode'][sl]; row = pack['row_step'][sl]
        horizons = np.minimum(row[:, None]+np.arange(1, 11)[None, :], pack['full_lengths'][ep, None])
        targets = pack['full_state'][ep[:, None], horizons]
        z = None if z_by_cube is None else torch.tensor(np.stack([z_by_cube[int(pack['cube_ids'][e])] for e in ep]), dtype=torch.float32, device=device)
        chunk = predict_batch(model, torch.tensor(pack['state'][sl], device=device),
            torch.tensor(pack['force'][ep], device=device),
            None if ep_context is None else ep_context[torch.tensor(ep, device=device)],
            torch.tensor(pack['history'][sl], device=device) if model.variant == 'history' else None,
            z, norm, torch.tensor(pack['push_remaining'][sl], device=device),
            torch.tensor(pack['remaining'][sl], device=device), torch.tensor(targets, device=device),
            torch.tensor(pack['next_state'][sl], device=device),
            torch.tensor(pack['stop_xy'][ep], dtype=torch.float64, device=device), primary_only)
        for name in names: values[name].append(chunk[name])
    groups = {}
    for group in GROUPS:
        groups[group] = {}
        for metric in names:
            mask = pack['rollout_masks'][group] if metric.startswith('rollout10') else pack['masks'][group]
            aggregate_result = aggregate(np.concatenate(values[metric]), pack['row_episode'], pack['cube_ids'], mask)
            require(aggregate_result['cubes'] == 30 and aggregate_result['episodes'] == 150,
                    f'Incomplete fresh query coverage: {group}/{metric}')
            if metric == PRIMARY:
                mse = aggregate_result['mean']*checkpoint['normalizer']['horizontal_velocity_delta_mse_scale']
                aggregate_result.update(raw_mse_m2_s2=mse, rmse_m_s=math.sqrt(mse))
            groups[group][metric] = aggregate_result
    return dict(model_name=model_name, variant=model.variant, seed=checkpoint['seed'],
                checkpoint_sha256=digest(checkpoint_path), selected_update=checkpoint['selected_update'],
                completed_updates=checkpoint['completed_updates'], groups=groups,
                test_cubes=30, test_episodes=150, query_episodes=QUERY_EPISODES,
                elapsed_seconds=time.perf_counter()-started, primary_only=primary_only,
                no_future_history=True, no_rollout_clipping=True,
                stopping_horizon='Planned release +1.5s, never observed stop/length')


def model_moments(records, names):
    result = {}
    for name in names:
        result[name] = {group: {metric: moments([records[name, seed]['groups'][group][metric]['mean']
                        for seed in SEEDS]) for metric in METRICS} for group in GROUPS}
    return result


def mass_score(z_values, cube_ids, decoder, true_masses):
    z_values = np.asarray(z_values, dtype=np.float64)
    coef = np.asarray(decoder['coefficients'], dtype=np.float64)
    predicted = np.column_stack([np.ones(len(z_values)), z_values]) @ coef
    truth = np.asarray([true_masses[int(c)] for c in cube_ids], dtype=np.float64)
    baseline = np.full(len(truth), decoder['training_mean_mass_kg'])
    require(np.isfinite(predicted).all(), 'Nonfinite mass readout: protocol stop')
    denominator = float(np.sum((truth-truth.mean())**2))
    require(denominator > 0, 'Undefined test mass variance')
    def scores(p):
        return dict(r2=1-float(np.sum((p-truth)**2))/denominator,
                    mae_kg=float(np.mean(np.abs(p-truth))), rmse_kg=float(np.sqrt(np.mean((p-truth)**2))))
    return dict(**scores(predicted), baseline=scores(baseline), cubes=len(cube_ids),
                predicted_mass_by_cube=[dict(cube_id=int(c), prediction_kg=float(p)) for c, p in zip(cube_ids, predicted)],
                label_storage='True masses read only from cubes.json; not copied here')


def freeze_context(root):
    require(read(root/'results/full_audit.json')['status'] == 'passed', 'Collection audit failed')
    require(not (root/'results/STOP.json').exists(), 'Prior protocol STOP exists')
    seal = read(root/'results/checkpoint_seal.json')
    expected = {f'{v}_seed_{s}.pt' for v in models.VARIANTS for s in SEEDS}
    require(set(seal['checkpoints']) == expected, 'All35 candidate checkpoints must be frozen before test')
    verify_manifest(root/'checkpoints', seal['checkpoints']); verify_manifest(root, seal['manifest'])
    adaptation = read(root/'results/adaptation_seal.json')
    require(adaptation['checkpoint_seal_sha256'] == digest(root/'results/checkpoint_seal.json'), 'Adaptation/checkpoint seal mismatch')
    verify_manifest(root, adaptation['files'])
    models.freeze_test_data(root)
    return dict(checkpoint_seal_sha256=digest(root/'results/checkpoint_seal.json'),
                adaptation_seal_sha256=digest(root/'results/adaptation_seal.json'),
                test_data_seal_sha256=digest(root/'results/test_data_seal.json'),
                data_seal_sha256=digest(root/'data/seal.json'),
                source_sha256=digest(Path(__file__)), protocol_sha256=digest(root/'PROTOCOL.md'))


def run(root=ROOT, device_name='cpu', batch_size=2048):
    root = Path(root); context = freeze_context(root)
    ledgerpath = root/'results/final_evaluation.json'
    if ledgerpath.exists():
        ledger = read(ledgerpath)
        require(ledger['context'] == context, 'Frozen evaluation inputs changed')
        require(ledger['status'] != 'stopped', 'Prior integrity/computation stop: no automatic rerun')
        if ledger['status'] == 'complete': return read(root/'results/summary.json')
    else:
        write_new(ledgerpath, dict(status='begun', context=context, started_unix_seconds=time.time(),
                                  note='All35 checkpoints, z settings and mass readouts frozen before test'))
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = torch.device(device_name)
    require(device.type != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable')
    # Command-only probe was explicitly deferred until this frozen test ledger.
    # It cannot alter dynamics checkpoints or adaptation choices.
    if not (root/'results/action_probe.json').exists():
        from . import collection
        require(root.resolve() == collection.ROOT.resolve(), 'Command probe root must match frozen collection root')
        collection.probe()
    normalizer = read(root/'results/normalizer.json')
    pack = make_query_pack(root, normalizer); properties = models.property_lookup(root)
    records = {}
    # Complete the entire preregistered Stage A before recording the C1 decision.
    for seed in SEEDS:
        for name in BASELINES:
            path = root/'results/test_metrics'/f'{name}_seed_{seed}.json'
            cp_path = root/'checkpoints'/f'{name}_seed_{seed}.pt'
            if path.exists():
                result = read(path); require(result['checkpoint_sha256'] == digest(cp_path), 'Saved baseline checkpoint mismatch')
            else:
                model, checkpoint = models.load_checkpoint(cp_path, device)
                result = evaluate_model(root, model, checkpoint, cp_path, pack, properties, device, name, batch_size=batch_size)
                write_new(path, result)
            records[name, seed] = result
            print(json.dumps(dict(stage='stage_a', model=name, seed=seed,
                                  primary=result['groups']['push'][PRIMARY]['mean'])), flush=True)
    by_model = model_moments(records, BASELINES)
    c1 = c1_decision(by_model['plain']['push'][PRIMARY], by_model['oracle']['push'][PRIMARY])
    c1path = root/'results/c1.json'
    if c1path.exists(): require(read(c1path) == c1, 'C1 changed during frozen resume')
    else: write_new(c1path, c1)
    summary = dict(c1=c1, c2=None, c3=None, stopped_after_c1=not c1['passed'],
                   by_model=by_model, comparisons={}, gap_recovery=None, prefix_curve=None, mass_readout=None,
                   seeds=SEEDS, test_cubes=TEST_CUBES, query_episodes=QUERY_EPISODES,
                   primary_group='push', primary_metric=PRIMARY,
                   frozen_context=context, history_reported_regardless_of_c1=True)
    if c1['passed']:
        config = read(root/'results/selected_config.json')
        readouts = read(root/'results/mass_readouts.json')['by_seed']
        prefixes = {}; mass_results = {}
        for seed in SEEDS:
            cp_path = root/'checkpoints'/f"{config['variant']}_seed_{seed}.pt"
            model, checkpoint = models.load_checkpoint(cp_path, device)
            initial = model.embedding.weight.detach().mean(0)
            prefixpath = root/'results'/f'prefix_scores_seed_{seed}.json'
            if prefixpath.exists():
                prefix = read(prefixpath)
                require(prefix['checkpoint_sha256'] == digest(cp_path), 'Saved prefix checkpoint mismatch')
            else:
                prefix = dict(seed=seed, checkpoint_sha256=digest(cp_path), support_counts=list(range(1, 6)),
                              selected_config=config, scores={}, z_vectors={}, fit_logs={}, query_episodes=QUERY_EPISODES)
                for count in range(1, 6):
                    support = models.load_episodes(root, TEST_CUBES, range(count), allow_test=True)
                    fitted = models.fit_zs(model, support, checkpoint['normalizer'], initial,
                                          config['learning_rate'], config['updates'], device)
                    require(fitted['cube_ids'] == TEST_CUBES and fitted['weights_unchanged'], 'Latent fit violated cube/weight invariants')
                    zs = fitted['snapshots'][config['updates']]
                    zmap = {cid: zs[i] for i, cid in enumerate(TEST_CUBES)}
                    score = evaluate_model(root, model, checkpoint, cp_path, pack, properties, device,
                                           f'fitted_z_{count}', zmap, primary_only=True, batch_size=batch_size)
                    prefix['scores'][str(count)] = score['groups']['push'][PRIMARY]
                    prefix['z_vectors'][str(count)] = np.asarray(zs).tolist()
                    prefix['fit_logs'][str(count)] = fitted['log']
                    print(json.dumps(dict(stage='adaptation_prefix', seed=seed, support_shoves=count,
                                          primary=prefix['scores'][str(count)]['mean'])), flush=True)
                write_new(prefixpath, prefix)
            prefixes[seed] = prefix
            fitted_z = np.asarray(prefix['z_vectors']['5'], dtype=np.float32)
            fitted_map = {cid: fitted_z[i] for i, cid in enumerate(TEST_CUBES)}
            mean = initial.detach().cpu().numpy()
            mapping = wrong_z_mapping(TEST_CUBES)
            z_maps = dict(fitted_z=fitted_map, mean_z={cid: mean for cid in TEST_CUBES},
                          wrong_z={cid: fitted_map[mapping[cid]] for cid in TEST_CUBES})
            for name, zmap in z_maps.items():
                path = root/'results/test_metrics'/f'{name}_seed_{seed}.json'
                if path.exists():
                    result = read(path); require(result['checkpoint_sha256'] == digest(cp_path), 'Saved latent-control checkpoint mismatch')
                else:
                    result = evaluate_model(root, model, checkpoint, cp_path, pack, properties, device,
                                            name, zmap, batch_size=batch_size)
                    if name == 'wrong_z': result['source_cube_mapping'] = mapping
                    write_new(path, result)
                records[name, seed] = result
            require(math.isclose(records['fitted_z', seed]['groups']['push'][PRIMARY]['mean'],
                                 prefix['scores']['5']['mean'], rel_tol=1e-10, abs_tol=1e-12),
                    'Five-support primary changed between prefix and full scoring')
            masspath = root/'results'/f'mass_scores_seed_{seed}.json'
            if masspath.exists(): mass = read(masspath)
            else:
                truth = {int(c['cube_id']): c['mass_kg'] for c in read(root/'data/cubes.json')['cubes']}
                mass = dict(seed=seed, **mass_score(fitted_z, TEST_CUBES, readouts[str(seed)], truth))
                write_new(masspath, mass)
            mass_results[seed] = mass
        by_model = model_moments(records, BASELINES + CONTROLS)
        p, o, z, mean, wrong = [by_model[name]['push'][PRIMARY] for name in ['plain', 'oracle', 'fitted_z', 'mean_z', 'wrong_z']]
        c2, c3 = c2_c3_decisions(p, o, z, mean, wrong)
        summary.update(by_model=by_model, c2=c2, c3=c3, gap_recovery=c2,
            prefix_curve={str(k): moments([prefixes[s]['scores'][str(k)]['mean'] for s in SEEDS]) for k in range(1, 6)},
            mass_readout={metric: moments([mass_results[s][metric] for s in SEEDS]) for metric in ['r2', 'mae_kg', 'rmse_kg']},
            mass_baseline={metric: moments([mass_results[s]['baseline'][metric] for s in SEEDS]) for metric in ['r2', 'mae_kg', 'rmse_kg']},
            selected_config=config)
    # C1 failure performs no Stage B fitting or scoring; only reports saved Stage A.
    summary['comparisons'] = {name: {g: {m: comparison(summary['by_model']['plain'][g][m], summary['by_model'][name][g][m])
                                   for m in METRICS} for g in GROUPS}
                              for name in summary['by_model'] if name != 'plain'}
    summary['all_applicable_criteria_passed'] = bool(c1['passed'] and summary['c2']['passed'] and summary['c3']['passed'])
    require(freeze_context(root) == context, 'Sealed inputs changed during final evaluation')
    path = root/'results/summary.json'
    if path.exists(): require(read(path) == summary, 'Saved final summary changed')
    else: write_new(path, summary)
    ledger = read(ledgerpath)
    ledger.update(status='complete', completed_unix_seconds=time.time(), summary_sha256=digest(path),
                  c1_passed=c1['passed'], stopped_after_c1=summary['stopped_after_c1'],
                  all_applicable_criteria_passed=summary['all_applicable_criteria_passed'])
    write_atomic(ledgerpath, ledger)
    print(json.dumps(dict(stage='final_complete', c1=c1['passed'],
                          c2=None if summary['c2'] is None else summary['c2']['passed'],
                          c3=None if summary['c3'] is None else summary['c3']['passed'])), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=2048)
    args = parser.parse_args()
    try:
        run(args.root.resolve(), args.device, args.batch_size)
    except Exception as error:
        root = args.root.resolve(); ledgerpath = root/'results/final_evaluation.json'
        stop = dict(status='stopped', exception_type=type(error).__name__, reason=str(error),
                    unix_seconds=time.time(), no_automatic_retry=True)
        destination = root/'results/STOP.json'
        if not destination.exists(): write_new(destination, stop)
        if ledgerpath.exists():
            ledger = read(ledgerpath)
            if ledger.get('status') != 'complete':
                ledger.update(status='stopped', failure=stop); write_atomic(ledgerpath, ledger)
        raise


if __name__ == '__main__': main()
