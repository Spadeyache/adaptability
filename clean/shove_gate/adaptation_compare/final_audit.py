"""Independent Step 3 audit of saved observations, scores, and file hashes.

This module never imports an experiment driver, performs a model forward pass,
or runs an optimizer.  The arithmetic here is deliberately separate from the
evaluation implementation.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
STEP2 = ROOT.parent / 'push_latent'
SEEDS = [101, 211, 307, 401, 503]
TRAIN = list(range(180))
VALIDATION = list(range(180, 200))
TEST = list(range(290, 320))
PREFIXES = [1, 2, 3, 4, 5]
SNAPSHOTS = [1, 5, 6, 10, 15, 16, 20, 25, 30]
RETURNS = [6, 16, 30]
LRS = [0.00001, 0.0001]
COMPONENTS = ['push_velocity_nmse', 'coast_velocity_nmse', 'full_state_nmse']
METRICS = COMPONENTS + ['composite']
PROTOCOL_SHA = 'b7c290b7e65cad349cb22cacd278e8f0c4b81d7316b65195f0237d1cb3053675'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same(observed, expected, label, rtol=1e-10, atol=1e-12):
    a = np.asarray(observed, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    require(a.shape == b.shape, f'{label}: shape differs')
    require(np.isfinite(a).all() and np.isfinite(b).all(), f'{label}: nonfinite')
    require(np.allclose(a, b, rtol=rtol, atol=atol), f'{label}: value differs')


def hashes(base, manifest):
    for name, expected in manifest.items():
        require(sha(Path(base) / name) == expected, f'Changed sealed artifact: {name}')
    return len(manifest)


def moments(values):
    a = np.asarray(values, dtype=np.float64)
    require(a.shape == (5,) and np.isfinite(a).all(), 'Need five finite paired seeds')
    return {'seed_values': a.tolist(), 'mean': float(a.mean()), 'sample_sd': float(a.std(ddof=1))}


def verify_moments(observed, values, label):
    expected = moments(values)
    for key, value in expected.items():
        same(observed[key], value, f'{label}/{key}')
    return expected


def verify_aggregate(value, cube_ids, episodes_per_cube, label):
    require(value['cubes'] == len(cube_ids), f'{label}: cube count')
    require(value['episodes'] == len(cube_ids) * episodes_per_cube, f'{label}: episode count')
    require(value['nonfinite_starts'] == 0, f'{label}: finite coverage')
    if 'eligible_starts' in value:
        require(value['eligible_starts'] > 0, f'{label}: nonempty step coverage')
    require(not value.get('missing_cube_ids', []), f'{label}: missing cubes')
    rows = value['per_cube']
    require([r['cube_id'] for r in rows] == cube_ids, f'{label}: cube identities')
    require(all(r['episodes'] == episodes_per_cube for r in rows), f'{label}: per-cube episode coverage')
    same(value['mean'], np.mean([r['mean'] for r in rows]), f'{label}: equal-cube mean')
    if 'per_episode' in value:
        episodes = value['per_episode']
        require(len(episodes) == len(cube_ids) * episodes_per_cube, f'{label}: episode rows')
        if 'eligible_starts' in value:
            require(sum(r['starts'] for r in episodes) == value['eligible_starts'], f'{label}: step counts')
        for row in rows:
            chosen = [e['mean'] for e in episodes if e['cube_id'] == row['cube_id']]
            require(len(chosen) == episodes_per_cube, f'{label}: episode grouping')
            same(row['mean'], np.mean(chosen), f'{label}: equal-episode mean')


def p1_arithmetic(a_initial, a_returns, b_initial, b_returns):
    """Compute the exact preregistered per-return conditions independently."""
    a0 = np.asarray(a_initial, dtype=np.float64)
    b0 = np.asarray(b_initial, dtype=np.float64)
    moments(a0); moments(b0)
    by_return = {}
    for count in RETURNS:
        av = np.asarray(a_returns[count], dtype=np.float64)
        bv = np.asarray(b_returns[count], dtype=np.float64)
        moments(av); moments(bv)
        tolerance = 1e-12 + 1e-10 * np.abs(a0)
        a_nonrise = bool(np.all(av <= a0 + tolerance))
        b_rise = bool(bv.mean() - bv.std(ddof=1) > b0.mean() + b0.std(ddof=1))
        by_return[count] = {
            'a_nonrise': a_nonrise,
            'b_rise_beyond_seed_spread': b_rise,
            'passed': a_nonrise and b_rise,
            'a_change': moments(av - a0),
            'b_change': moments(bv - b0),
            'a_initial': moments(a0), 'a_return': moments(av),
            'b_initial': moments(b0), 'b_return': moments(bv),
        }
    return {'passed': all(x['passed'] for x in by_return.values()), 'by_return': by_return}


def p2_arithmetic(a_prefixes, b_five):
    b5 = np.asarray(b_five, dtype=np.float64)
    moments(b5)
    earliest = None
    differences = {}
    for count in PREFIXES:
        av = np.asarray(a_prefixes[count], dtype=np.float64)
        moments(av)
        differences[count] = moments(av - b5)
        if earliest is None and av.mean() <= b5.mean():
            earliest = count
    return {'passed': earliest is not None, 'earliest_prefix': earliest, 'paired_differences': differences}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_score(record, cube_ids, episode_ids, scales, label, oracle=False):
    require(record['cube_ids'] == cube_ids and record['episode_ids'] == episode_ids, f'{label}: score identities')
    require(record['properties_used'] is oracle, f'{label}: property access')
    expected_pairs = {(c, e) for c in cube_ids for e in episode_ids}
    for metric, value in record['aggregates'].items():
        verify_aggregate(value, cube_ids, len(episode_ids), f'{label}/{metric}')
        same(record['scores'][metric], value['mean'], f'{label}/{metric}: scalar/aggregate')
        pairs = [(r['cube_id'], r['episode_id']) for r in value['per_episode']]
        require(len(pairs) == len(expected_pairs) and set(pairs) == expected_pairs, f'{label}/{metric}: episode identities')
    same(record['scores']['composite'], np.mean([record['scores'][m] for m in COMPONENTS]), f'{label}: composite')
    for i, row in enumerate(record['aggregates']['composite']['per_cube']):
        same(row['mean'], np.mean([record['aggregates'][m]['per_cube'][i]['mean'] for m in COMPONENTS]), f'{label}: per-cube composite')
    for i, row in enumerate(record['aggregates']['composite']['per_episode']):
        same(row['mean'], np.mean([record['aggregates'][m]['per_episode'][i]['mean'] for m in COMPONENTS]), f'{label}: per-episode composite')
    for metric, raw, scale in [
        ('push_velocity_nmse', 'raw_push_velocity_mse', 'push_velocity_energy'),
        ('coast_velocity_nmse', 'raw_coast_velocity_mse', 'coast_velocity_energy'),
        ('all_coast_velocity_nmse', 'raw_all_coast_velocity_mse', 'coast_velocity_energy'),
        ('full_state_nmse', 'legacy_full_state_mse', 'full_state_energy'),
    ]:
        same(record['scores'][metric] * scales[scale], record['scores'][raw], f'{label}: {metric} units')
        for r, v in zip(record['aggregates'][metric]['per_cube'], record['aggregates'][raw]['per_cube']):
            same(r['mean'] * scales[scale], v['mean'], f'{label}: per-cube {metric} units')


def verify_fit(fit, cube_ids, prefixes, label, kind, base_state=None, penalty=None, lr=None):
    require(fit['updates'] == 50, f'{label}: adaptation updates')
    require(fit['support_episode_ids'] == list(range(prefixes)), f'{label}: support/query separation')
    require([r['update'] for r in fit['loss_log']] == list(range(1, 51)), f'{label}: complete loss log')
    for row in fit['loss_log']:
        require(all(math.isfinite(float(v)) for v in row.values()), f'{label}: nonfinite support log')
    if kind == 'A':
        require(fit['cube_ids'] == cube_ids and fit['weights_unchanged'], f'{label}: independent z/weights')
        require(fit['learning_rate'] == .01 and fit['penalty'] == penalty, f'{label}: fixed z settings')
        require(np.isfinite(fit['z_values']).all(), f'{label}: finite z')
        if base_state is not None:
            require(fit['model_state_sha256'] == base_state, f'{label}: frozen A state')
        for row in fit['loss_log']:
            same(row['mean_cube_data_loss'] + row['mean_cube_prior'], row['mean_cube_total_loss'], f'{label}: latent loss terms', rtol=1e-6, atol=1e-9)
            require(row['mean_cube_prior'] >= 0, f'{label}: nonnegative prior')
    else:
        require(fit['cube_id'] == cube_ids[0] and len(cube_ids) == 1, f'{label}: single-cube adaptation')
        require(fit['optimizer_reset'] and fit['all_parameter_tensors_receive_gradients'], f'{label}: all weights/fresh Adam')
        require(fit['learning_rate'] == lr, f'{label}: fixed B rate')
        require(len(fit['after_state_sha256']) == 64 and len(fit['before_state_sha256']) == 64, f'{label}: B state hashes missing')
        if base_state is not None:
            require(fit['before_state_sha256'] == base_state, f'{label}: B initialization')


def state_sha(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def vector_sha(values):
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()


def derive_scales_independently(root, normalizer):
    std = np.asarray(normalizer['state_std'], dtype=np.float32).astype(np.float64)
    by_cube = {m: [] for m in ['push_velocity_energy', 'coast_velocity_energy', 'full_state_energy']}
    counts = {m: 0 for m in by_cube}
    for cube in TRAIN:
        by_episode = {m: [] for m in by_cube}
        for episode in range(8):
            path = root / f'data/episodes/cube_{cube:03d}_episode_{episode:02d}.npz'
            with np.load(path, allow_pickle=False) as trace:
                require(np.all(trace['cube_id'] == cube) and np.all(trace['episode_id'] == episode), 'Scale trace identities')
                state = trace['state'].astype(np.float32).astype(np.float64)
                nxt = trace['next_state'].astype(np.float32).astype(np.float64)
                phase = trace['phase']
            moving = (np.linalg.norm(state[:, 9:12], axis=1) > .001) | (np.linalg.norm(state[:, 12:15], axis=1) > .05)
            push, coast = phase == 'push', (phase == 'coast') & moving
            require(push.any() and coast.any(), 'Every fitting shove must cover declared score groups')
            delta = nxt - state
            values = {'push_velocity_energy': ((delta[push, 9:11] ** 2).mean(1)),
                      'coast_velocity_energy': ((delta[coast, 9:11] ** 2).mean(1)),
                      'full_state_energy': (((delta / std) ** 2).mean(1))}
            for name, rows in values.items():
                require(np.isfinite(rows).all(), 'Scale observations nonfinite')
                by_episode[name].append(float(rows.mean())); counts[name] += len(rows)
        for name in by_cube:
            by_cube[name].append(float(np.mean(by_episode[name])))
    return {name: float(np.mean(values)) for name, values in by_cube.items()}, counts


def verify_validation(root, scales, selected, checkpoint_hashes, checkpoint_states):
    grid = read(root / 'results/validation_grid.json')
    aa, bb = grid['a_records'], grid['b_records']
    a_grid = {(d, p, s) for d in [4, 8, 16] for p in [.0001, .001, .01, .1, 1.] for s in SEEDS}
    b_grid = {(lr, s) for lr in LRS for s in SEEDS}
    require(len(aa) == 75 and {(r['z_dim'], r['penalty'], r['seed']) for r in aa} == a_grid, 'Complete A validation grid')
    require(len(bb) == 10 and {(r['learning_rate'], r['seed']) for r in bb} == b_grid, 'Complete B validation grid')
    for row in aa:
        variant = f'z{row["z_dim"]}'; seed = row['seed']
        require(row['learning_rate'] == .01 and row['updates'] == 50 and row['weights_unchanged'], 'A validation fixed settings')
        require(row['checkpoint_sha256'] == checkpoint_hashes[f'{variant}_seed_{seed}.pt'], 'A validation checkpoint')
        require(row['model_state_sha256'] == checkpoint_states[variant, seed], 'A validation shared weights')
        require(row['query']['model_variant'] == variant, 'A validation scored model variant')
        require(row['cube_ids'] == VALIDATION and np.asarray(row['z_values']).shape == (20, row['z_dim']), 'A validation latent dimensions/identities')
        require(np.isfinite(row['z_values']).all(), 'A validation latent finite')
        require([r['update'] for r in row['loss_log']] == list(range(1, 51)), 'A validation loss coverage')
        for loss in row['loss_log']:
            same(loss['mean_cube_data_loss'] + loss['mean_cube_prior'], loss['mean_cube_total_loss'], 'A validation prior arithmetic', rtol=1e-6, atol=1e-9)
        verify_score(row['query'], VALIDATION, list(range(5, 10)), scales, 'A validation')
        path = root / f'results/validation_runs/z{row["z_dim"]}_seed{seed}_penalty{row["penalty"]:g}.json'
        require(read(path) == row, 'A validation individual/grid record')
    for row in bb:
        seed, lr = row['seed'], row['learning_rate']
        require(row['updates'] == 50 and row['checkpoint_sha256'] == checkpoint_hashes[f'plain_seed_{seed}.pt'], 'B validation fixed setting/checkpoint')
        require(row['query']['model_variant'] == 'plain', 'B validation scored model variant')
        require([r['cube_id'] for r in row['fit_logs']] == VALIDATION, 'B validation adaptation coverage')
        for fit in row['fit_logs']:
            verify_fit(fit, [fit['cube_id']], 5, 'B validation', 'B', checkpoint_states['plain', seed], lr=lr)
        verify_score(row['query'], VALIDATION, list(range(5, 10)), scales, 'B validation')
        require(read(root / f'results/validation_runs/plain_seed{seed}_lr{lr:g}.json') == row, 'B validation individual/grid record')
    candidates = []
    for dim in [4, 8, 16]:
        for penalty in [.0001, .001, .01, .1, 1.]:
            rows = [next(r for r in aa if (r['z_dim'], r['penalty'], r['seed']) == (dim, penalty, seed)) for seed in SEEDS]
            candidates.append(dict(z_dim=dim, variant=f'z{dim}', penalty=penalty, learning_rate=.01, updates=50,
                                   validation={m: moments([r['query']['scores'][m] for r in rows]) for m in METRICS}))
    b_candidates = []
    for lr in LRS:
        rows = [next(r for r in bb if (r['learning_rate'], r['seed']) == (lr, seed)) for seed in SEEDS]
        b_candidates.append(dict(learning_rate=lr, updates=50,
                                validation={m: moments([r['query']['scores'][m] for r in rows]) for m in METRICS}))
    chosen_a = min(candidates, key=lambda r: (r['validation']['composite']['mean'], r['z_dim'], -r['penalty']))
    chosen_b = min(b_candidates, key=lambda r: (r['validation']['composite']['mean'], r['learning_rate']))
    require(selected['A'] == chosen_a and selected['B'] == chosen_b, 'Validation selection or tie-break differs')
    require(selected['a_candidates'] == candidates and selected['b_rates'] == b_candidates, 'Saved candidate summaries differ')
    require(selected == grid['selection'], 'Grid/selected setting differs')
    require(selected['selection_population'] == VALIDATION and selected['selection_metric'] == 'composite', 'Validation-only selection')
    require(selected['seeds'] == SEEDS and selected['support_episodes'] == list(range(5)) and selected['query_episodes'] == list(range(5, 10)), 'Validation split/seeds')
    return 85


def verify_extracted_cube(record, original, cube_id, label):
    require(record['cube_id'] == cube_id and record['query_shoves'] == list(range(5, 10)), f'{label}: extracted identity')
    require(record['derived_from_saved_per_cube_aggregate'], f'{label}: extraction provenance')
    for metric in record['scores']:
        matches = [r for r in original['aggregates'][metric]['per_cube'] if r['cube_id'] == cube_id]
        require(len(matches) == 1, f'{label}: unique source cube')
        same(record['scores'][metric], matches[0]['mean'], f'{label}: extracted {metric}')


def verify_test_records(root, scales, selected, checkpoint_hashes, checkpoint_states, summary):
    import torch  # Loading saved tensor bytes only; no model is constructed.
    a_records, b_records, sequences, baseline_records = {}, {}, {}, {}
    counts = {'score_populations': 0, 'independent_b_fit_events': 0, 'sequential_b_fit_events': 0, 'sequence_snapshots': 0}
    avariant = selected['A']['variant']; penalty = selected['A']['penalty']; dim = selected['A']['z_dim']
    for seed in SEEDS:
        for model in ['plain', 'history', 'oracle']:
            rec = read(root / f'results/baselines/{model}_seed{seed}.json')
            baseline_records[model, seed] = rec
            require(rec['seed'] == seed and rec['model'] == model, 'Baseline identity')
            require(rec['checkpoint_sha256'] == checkpoint_hashes[f'{model}_seed_{seed}.pt'], 'Baseline checkpoint')
            require(rec['query']['model_variant'] == model and rec['training_validation']['model_variant'] == model, 'Baseline scored model variant')
            verify_score(rec['query'], TEST, list(range(5, 10)), scales, 'Baseline query', oracle=model == 'oracle')
            verify_score(rec['training_validation'], TRAIN, [8, 9], scales, 'Baseline training validation', oracle=model == 'oracle')
            verify_extracted_cube(rec['first_cube'], rec['query'], 290, 'Baseline first cube')
            counts['score_populations'] += 2
        for count in PREFIXES:
            rec = read(root / f'results/prefix/A_seed{seed}_k{count}.json'); a_records[seed, count] = rec
            require(rec['seed'] == seed and rec['support_shoves'] == count, 'A prefix identity')
            require(rec['checkpoint_sha256'] == checkpoint_hashes[f'{avariant}_seed_{seed}.pt'], 'A prefix checkpoint')
            require(rec['query']['model_variant'] == avariant, 'A prefix scored model variant')
            verify_fit(rec['fit'], TEST, count, 'A prefix', 'A', checkpoint_states[avariant, seed], penalty=penalty)
            require(np.asarray(rec['fit']['z_values']).shape == (30, dim), 'A prefix latent shape')
            verify_score(rec['query'], TEST, list(range(5, 10)), scales, 'A prefix query')
            counts['score_populations'] += 1
        for lr in LRS:
            key = format(lr, '.0e')
            for count in PREFIXES:
                rec = read(root / f'results/prefix/B_lr{key}_seed{seed}_k{count}.json'); b_records[seed, key, count] = rec
                require(rec['seed'] == seed and rec['support_shoves'] == count and rec['learning_rate'] == lr, 'B prefix identity')
                require(rec['checkpoint_sha256'] == checkpoint_hashes[f'plain_seed_{seed}.pt'], 'B prefix checkpoint')
                require([r['cube_id'] for r in rec['per_cube']] == TEST, 'B prefix cube coverage')
                for row in rec['per_cube']:
                    require(row['query']['model_variant'] == 'plain', 'B prefix scored model variant')
                    verify_fit(row['fit'], [row['cube_id']], count, 'B prefix', 'B', checkpoint_states['plain', seed], lr=lr)
                    verify_score(row['query'], [row['cube_id']], list(range(5, 10)), scales, 'B prefix query')
                    counts['score_populations'] += 1; counts['independent_b_fit_events'] += 1
                for metric, mean in rec['scores'].items():
                    same(mean, np.mean([r['query']['scores'][metric] for r in rec['per_cube']]), 'B prefix equal-cube aggregation')
        aseq = read(root / f'results/sequence/A_seed{seed}.json'); sequences['A', seed] = aseq
        require(aseq['seed'] == seed and aseq['checkpoint_sha256'] == checkpoint_hashes[f'{avariant}_seed_{seed}.pt'], 'A sequence checkpoint')
        require([r['cube_id'] for r in aseq['visits']] == TEST and [r['visit'] for r in aseq['visits']] == list(range(1, 31)), 'A sequence visit order')
        require([r['visits'] for r in aseq['snapshots']] == SNAPSHOTS, 'A sequence observation schedule')
        require(set(aseq['stored_z']) == set(map(str, TEST)), 'A stored vector coverage')
        fit5 = a_records[seed, 5]['fit']; aq = a_records[seed, 5]['query']
        verify_score(aseq['initial_training_validation'], TRAIN, [8, 9], scales, 'A initial training validation')
        counts['score_populations'] += 1
        for i, row in enumerate(aseq['visits']):
            cube = TEST[i]
            same(aseq['stored_z'][str(cube)], fit5['z_values'][i], 'A cached vector equals independent 5-shove fit', rtol=0, atol=0)
            require(row['z_sha256'] == vector_sha(aseq['stored_z'][str(cube)]), 'A visit latent hash')
            require(row['cached_from_independent_five_shove_fit'], 'A cached-fit provenance')
            verify_extracted_cube(row['query'], aq, cube, 'A currently visited query')
        for snap in aseq['snapshots']:
            n = snap['visits']; require(snap['other_cubes_seen'] == n - 1, 'A other-cube return count')
            require(snap['weights_sha256'] == checkpoint_states[avariant, seed], 'A shared weights changed')
            require(snap['z_first_sha256'] == vector_sha(aseq['stored_z']['290']), 'A first latent changed')
            require(snap['stored_cube_ids'] == TEST[:n] and snap['cached_predictions_verified_by_unchanged_hashes'], 'A stored/cached provenance')
            require(snap['training_validation'] == aseq['initial_training_validation'], 'A training-validation retention')
            verify_extracted_cube(snap['first_cube'], aq, 290, 'A first-cube retention')
            counts['sequence_snapshots'] += 1
        for lr in LRS:
            key = format(lr, '.0e'); seq = read(root / f'results/sequence/B_lr{key}_seed{seed}.json')
            sequences[key, seed] = seq
            require(seq['seed'] == seed and seq['learning_rate'] == lr, 'B sequence identity')
            require(seq['checkpoint_sha256'] == checkpoint_hashes[f'plain_seed_{seed}.pt'], 'B sequence checkpoint')
            require([r['cube_id'] for r in seq['visits']] == TEST and [r['visit'] for r in seq['visits']] == list(range(1, 31)), 'B sequence visit order')
            require([r['visits'] for r in seq['snapshots']] == SNAPSHOTS, 'B sequence observation schedule')
            verify_score(seq['initial_training_validation'], TRAIN, [8, 9], scales, 'B initial training validation')
            for m in seq['initial_training_validation']['scores']:
                same(seq['initial_training_validation']['scores'][m], baseline_records['plain', seed]['training_validation']['scores'][m], 'B initial/base Plain agreement')
            previous = checkpoint_states['plain', seed]
            for row in seq['visits']:
                verify_fit(row['fit'], [row['cube_id']], 5, 'B sequence', 'B', previous, lr=lr)
                require(row['before_weights_sha256'] == previous == row['fit']['before_state_sha256'], 'B carried previous weights')
                require(row['after_weights_sha256'] == row['fit']['after_state_sha256'], 'B saved updated weights')
                verify_score(row['query'], [row['cube_id']], list(range(5, 10)), scales, 'B current-cube query')
                previous = row['after_weights_sha256']; counts['score_populations'] += 1; counts['sequential_b_fit_events'] += 1
            first_independent = b_records[seed, key, 5]['per_cube'][0]
            require(seq['visits'][0]['after_weights_sha256'] == first_independent['fit']['after_state_sha256'], 'B first sequence fit differs from same independent fit')
            for snap in seq['snapshots']:
                n = snap['visits']; require(snap['other_cubes_seen'] == n - 1, 'B other-cube return count')
                require(snap['weights_sha256'] == seq['visits'][n-1]['after_weights_sha256'], 'B snapshot/visit state mismatch')
                path = root / snap['checkpoint_file']; require(sha(path) == snap['checkpoint_file_sha256'], 'B snapshot file changed')
                state = torch.load(path, map_location='cpu', weights_only=True)
                require(state_sha(state) == snap['weights_sha256'], 'B snapshot tensor hash')
                require(all(np.isfinite(t.cpu().numpy()).all() for t in state.values()), 'B saved snapshot nonfinite')
                verify_score(snap['first_cube'], [290], list(range(5, 10)), scales, 'B first-cube return')
                verify_score(snap['training_validation'], TRAIN, [8, 9], scales, 'B training-validation forgetting')
                counts['score_populations'] += 2; counts['sequence_snapshots'] += 1
            for metric in METRICS:
                same(seq['snapshots'][0]['first_cube']['scores'][metric], first_independent['query']['scores'][metric], 'B first-visit query reproducibility')
    for count in PREFIXES:
        for metric in METRICS:
            verify_moments(summary['curves']['A'][str(count)][metric], [a_records[s, count]['query']['scores'][metric] for s in SEEDS], 'A prefix moments')
            for lr in LRS:
                key = format(lr, '.0e')
                verify_moments(summary['curves']['B'][key][str(count)][metric], [b_records[s, key, count]['scores'][metric] for s in SEEDS], 'B prefix moments')
    for model in ['plain', 'history', 'oracle']:
        for population in ['query', 'first_cube', 'training_validation']:
            for metric in METRICS:
                verify_moments(summary['baselines'][model][population][metric], [baseline_records[model, s][population]['scores'][metric] for s in SEEDS], 'Baseline moments')
    for label in ['A'] + [format(lr, '.0e') for lr in LRS]:
        dest = summary['sequence']['A'] if label == 'A' else summary['sequence']['B'][label]
        for metric in METRICS:
            verify_moments(dest['0']['training_validation'][metric], [sequences[label, s]['initial_training_validation']['scores'][metric] for s in SEEDS], 'Sequence initial moments')
        for n in SNAPSHOTS:
            for population in ['first_cube', 'training_validation']:
                for metric in METRICS:
                    values = [next(r for r in sequences[label, s]['snapshots'] if r['visits'] == n)[population]['scores'][metric] for s in SEEDS]
                    verify_moments(dest[str(n)][population][metric], values, 'Sequence snapshot moments')
    return counts


def verify_criteria(summary, selected):
    criteria = summary['criteria']
    primary = format(selected['B']['learning_rate'], '.0e')
    require(criteria['primary_lr'] == primary, 'Primary B rate changed after validation')
    for lr in LRS:
        key = format(lr, '.0e')
        aseq, bseq = summary['sequence']['A'], summary['sequence']['B'][key]
        acurve, bcurve = summary['curves']['A'], summary['curves']['B'][key]
        for metric in METRICS:
            record = criteria['by_lr'][key][metric]
            expected_p1 = p1_arithmetic(
                aseq['1']['first_cube'][metric]['seed_values'],
                {n: aseq[str(n)]['first_cube'][metric]['seed_values'] for n in RETURNS},
                bseq['1']['first_cube'][metric]['seed_values'],
                {n: bseq[str(n)]['first_cube'][metric]['seed_values'] for n in RETURNS})
            require(record['p1']['passed'] == expected_p1['passed'], 'P1 overall arithmetic')
            require(set(record['p1']['by_other_cubes']) == {'5', '15', '29'}, 'P1 return identities')
            for n in RETURNS:
                actual = record['p1']['by_other_cubes'][str(n - 1)]; expected = expected_p1['by_return'][n]
                require(actual['total_visits'] == n, 'P1 return indexing')
                for field in ['a_nonrise', 'b_rise_beyond_seed_spread', 'passed']:
                    require(actual[field] == expected[field], 'P1 component decision')
                for afield, efield in [('a_initial', 'a_initial'), ('a_return', 'a_return'),
                                      ('b_initial', 'b_initial'), ('b_return', 'b_return'),
                                      ('a_paired_change', 'a_change'), ('b_paired_change', 'b_change')]:
                    verify_moments(actual[afield], expected[efield]['seed_values'], 'P1 moments')
            expected_p2 = p2_arithmetic({n: acurve[str(n)][metric]['seed_values'] for n in PREFIXES}, bcurve['5'][metric]['seed_values'])
            require(record['p2']['passed'] == expected_p2['passed'] and record['p2']['earliest_prefix'] == expected_p2['earliest_prefix'], 'P2 crossing decision')
            for n in PREFIXES:
                row = record['p2']['comparisons'][str(n)]
                require(row['a'] == acurve[str(n)][metric] and row['b_five'] == bcurve['5'][metric], 'P2 population substitution')
                require(row['reaches'] == (row['a']['mean'] <= row['b_five']['mean']), 'P2 exact threshold')
                verify_moments(row['paired_difference'], expected_p2['paired_differences'][n]['seed_values'], 'P2 paired differences')
    require(criteria['p1'] == criteria['by_lr'][primary]['composite']['p1'], 'Primary P1 changed')
    require(criteria['p2'] == criteria['by_lr'][primary]['composite']['p2'], 'Primary P2 changed')
    for m in COMPONENTS:
        require(criteria['components'][m] == criteria['by_lr'][primary][m], 'Component verdict differs')
    passed = criteria['p1']['passed'] and criteria['p2']['passed']
    require(summary['all_criteria_passed'] == passed, 'Overall P1/P2 verdict')
    return {'p1_passed': criteria['p1']['passed'], 'p2_passed': criteria['p2']['passed'], 'all_criteria_passed': passed}


def verify_saved_validation(root):
    """Audit completed tuning, including fitting-only energies, without scoring."""
    import torch
    require(sha(root / 'PROTOCOL.md') == PROTOCOL_SHA, 'Written protocol changed')
    counts = {}
    for label, path in [('collection_source', 'results/source_seal.json'),
                        ('retired', 'results/retired_experiment_seal.json'),
                        ('prepared_training', 'results/prepared_training_seal.json')]:
        counts[label] = hashes(root, read(root / path))
    preparation = read(root / 'results/preparation.json')
    require(preparation['status'] == 'passed' and preparation['protocol_sha256'] == PROTOCOL_SHA, 'Preparation failed/changed')
    require(preparation['reused_training_cubes'] == 200 and preparation['reused_training_episodes'] == 2000 and preparation['reused_trace_files'] == 4000, 'Original training coverage')
    require(preparation['retired_artifacts_preserved'] == counts['retired'] and preparation['base_checkpoints_preserved'] == 35, 'Retirement/checkpoint coverage')
    review = read(root / 'results/prevalidation_review.json')
    require(review['status'] == 'passed', 'Prevalidation contracts failed')
    manifest = review.get('files', review.get('hashes', review.get('source_sha256')))
    require(isinstance(manifest, dict), 'Prevalidation source manifest missing')
    counts['prevalidation'] = hashes(root, manifest)
    seal = read(root / 'results/selection_seal.json')
    counts['selection'] = hashes(root, seal['files'])
    require(seal['selected_config_sha256'] == sha(root / 'results/selected_config.json'), 'Selected configuration seal')
    cpseal = read(STEP2 / 'results/checkpoint_seal.json')['checkpoints']
    require(set(cpseal) == {f'{v}_seed_{s}.pt' for v in ['plain', 'oracle', 'history', 'z2', 'z4', 'z8', 'z16'] for s in SEEDS}, 'All 35 frozen checkpoints required')
    require(seal['frozen_step2_checkpoints'] == cpseal, 'Checkpoint selection changed')
    counts['checkpoints'] = hashes(STEP2 / 'checkpoints', cpseal)
    normalizer = read(STEP2 / 'results/normalizer.json')
    require(normalizer['fit_cubes'] == TRAIN and normalizer['fit_episodes'] == list(range(8)), 'Frozen scaler population')
    checkpoint_states = {}
    for name in cpseal:
        cp = torch.load(STEP2 / 'checkpoints' / name, map_location='cpu', weights_only=False)
        require(cp['normalizer'] == normalizer, 'Checkpoint normalization changed')
        checkpoint_states[cp['variant'], cp['seed']] = state_sha(cp['model_state'])
    scales = read(root / 'results/metric_scales.json')
    require(scales['fit_cube_ids'] == TRAIN and scales['fit_episode_ids'] == list(range(8)), 'Scoring-scale fitting population')
    independently, rows = derive_scales_independently(root, normalizer)
    for metric, value in independently.items():
        same(scales[metric], value, f'Independent {metric}')
    same(scales['push_velocity_energy'], normalizer['horizontal_velocity_delta_mse_scale'], 'Preserved push denominator')
    for metric, group in [('push_velocity_energy', 'push'), ('coast_velocity_energy', 'moving_coast'), ('full_state_energy', 'full_state')]:
        value = scales['coverage'][group]
        verify_aggregate(value, TRAIN, 8, f'Metric energy/{group}')
        same(value['mean'], scales[metric], 'Metric energy aggregate')
        require(value['eligible_starts'] == rows[metric], 'Metric energy row coverage')
    require(scales['zero_change_fitting_scores'] == {m: 1. for m in METRICS}, 'Energy zero-change reference')
    selected = read(root / 'results/selected_config.json')
    counts['validation_records'] = verify_validation(root, scales, selected, cpseal, checkpoint_states)
    require((root / 'PROTOCOL.md').stat().st_mtime <= preparation['prepared_unix_seconds'] <= (root / 'results/metric_scales.json').stat().st_mtime <= seal['completed_unix_seconds'], 'Protocol/preparation/scales/selection chronology')
    return dict(counts=counts, scales=scales, selected=selected, checkpoint_hashes=cpseal,
                checkpoint_states=checkpoint_states, selection=seal, preparation=preparation)


def main(validation_only=False):
    root = ROOT; results = root / 'results'
    output = results / ('validation_audit.json' if validation_only else 'final_audit.json')
    require(not output.exists(), 'Audit already exists; do not overwrite evidence')
    if validation_only:
        require(not (results / 'fresh_initialization.json').exists(), 'Validation audit must precede fresh cube generation')
        require(not (results / 'STOP.json').exists(), 'Stopped validation cannot receive a passing audit')
        checked = verify_saved_validation(root)
        result = dict(status='passed', audited_unix_seconds=time.time(), source_sha256=sha(__file__),
                      selected_config=checked['selected'], metric_scales={k: checked['scales'][k] for k in ['push_velocity_energy', 'coast_velocity_energy', 'full_state_energy']},
                      hash_counts=checked['counts'], no_fresh_test_generation=True, no_predictions_or_optimizers_run=True)
        write_new(output, result); print(json.dumps(result, indent=2)); return
    ledger = read(results / 'final_evaluation.json') if (results / 'final_evaluation.json').exists() else None
    stop = read(results / 'STOP.json') if (results / 'STOP.json').exists() else None
    require((ledger and ledger.get('status') in ['complete', 'stopped']) or stop, 'Final audit requires a completed or explicitly stopped experiment')
    if not ledger or ledger.get('status') != 'complete':
        write_new(output, dict(status='incomplete_protocol_stop', audited_unix_seconds=time.time(), source_sha256=sha(__file__),
                               stop=stop, ledger=ledger, no_missing_results_inferred=True, no_predictions_or_optimizers_run=True))
        print('Incomplete experiment recorded; no missing scores inferred.'); return
    require(not stop, 'Completed ledger conflicts with a STOP artifact')
    checked = verify_saved_validation(root); counts = checked['counts']
    selected, scales = checked['selected'], checked['scales']
    require(ledger['selected_config_sha256'] == sha(results / 'selected_config.json'), 'Final setting differs')
    require(ledger['source_sha256'] == sha(root / 'evaluation.py') and ledger['protocol_sha256'] == PROTOCOL_SHA, 'Final evaluator/protocol changed')
    pretest = read(results / 'pretest_review.json')
    require(pretest['status'] == 'passed' and ledger['pretest_review_sha256'] == sha(results / 'pretest_review.json'), 'Pretest review changed or failed')
    counts['pretest'] = hashes(root, pretest['files'])
    counts['data_definition'] = hashes(root / 'data', read(root / 'data/seal.json'))
    require(read(results / 'full_audit.json')['status'] == 'passed', 'Collection audit did not pass')
    announced = read(results / 'settings_announced.json'); fresh = read(results / 'fresh_initialization.json')
    require(announced['selected_config_sha256'] == ledger['selected_config_sha256'], 'Setting announcement mismatch')
    require(fresh['selected_config_sha256'] == ledger['selected_config_sha256'], 'Fresh generation setting mismatch')
    require(fresh['selection_seal_sha256'] == sha(results / 'selection_seal.json') and fresh['settings_announced_sha256'] == sha(results / 'settings_announced.json'), 'Fresh generation freeze mismatch')
    require(fresh['fresh_test_ids'] == TEST and fresh['fresh_seed'] == 2026111613 and fresh['no_rejection_or_resampling'], 'Fresh sample identity/generator')
    require(checked['selection']['completed_unix_seconds'] <= announced['announced_unix_seconds'] <= fresh['initialized_unix_seconds'] <= (results / 'full_audit.json').stat().st_mtime <= ledger['started_unix_seconds'] <= ledger['completed_unix_seconds'], 'Selection/announcement/collection/evaluation chronology')
    split = read(root / 'data/split.json')
    require(split['train'] == TRAIN and split['validation'] == VALIDATION and split['test'] == TEST and split['retired_test'] == list(range(200, 290)), 'Preserved split/fresh-test roles')
    expected_names = {f'cube_{c:03d}_episode_{e:02d}.npz' for c in list(range(200)) + TEST for e in range(10)}
    for kind in ['raw', 'episodes']:
        require({p.name for p in (root / 'data' / kind).glob('*.npz')} == expected_names, f'{kind}: missing/retired trace identity')
    rows = read(results / 'metrics.json')
    require(len(rows) == 2300 and {(r['cube_id'], r['episode_id']) for r in rows} == {(c, e) for c in list(range(200)) + TEST for e in range(10)}, 'Collection records complete')
    counts['trace_bytes'] = 0
    for row in rows:
        for kind in ['raw', 'episodes']:
            path = root / f'data/{kind}/cube_{row["cube_id"]:03d}_episode_{row["episode_id"]:02d}.npz'
            require(sha(path) == row[kind + '_sha256'], 'Collected/reused trace bytes changed')
            counts['trace_bytes'] += 1
    summary = read(results / 'summary.json')
    require(ledger['summary_sha256'] == sha(results / 'summary.json'), 'Final summary changed')
    require(summary['step'] == 3 and summary['seeds'] == SEEDS and summary['test_cubes'] == TEST and summary['query_shoves'] == list(range(5, 10)), 'Final query population/seeds')
    require(summary['selected_config'] == selected, 'Summary selected configuration')
    counts.update(verify_test_records(root, scales, selected, checked['checkpoint_hashes'], checked['checkpoint_states'], summary))
    verdict = verify_criteria(summary, selected)
    for key, value in verdict.items():
        require(ledger[key] == value, 'Ledger criterion differs')
    result = dict(status='passed', audited_unix_seconds=time.time(), source_sha256=sha(__file__),
                  hash_counts=counts, selected_config=selected, **verdict,
                  all30fresh_cubes_and150_query_shoves_covered=True, all50update_adaptation_logs_checked=True,
                  base_checkpoints_unchanged=True, retired_results_unchanged=True,
                  source_and_data_chronology_verified=True, independent_fitting_scales_verified=True,
                  no_predictions_or_optimizers_run=True, summary_sha256=sha(results / 'summary.json'))
    write_new(output, result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-only', action='store_true')
    main(validation_only=parser.parse_args().validation_only)

