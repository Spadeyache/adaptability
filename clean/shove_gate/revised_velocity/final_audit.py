"""Independent saved-artifact audit. Never predicts, trains, or simulates."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SEEDS = [101, 211, 307, 401, 503]
KINDS = ['plain', 'oracle']
GROUPS = ['push', 'moving_coast', 'all_steps']
PRIMARY = 'horizontal_velocity_delta_nmse'
METRICS = [PRIMARY, 'one_step_nmse', 'rollout10_nmse', 'rollout10_final_nmse',
           'stopping_position_error_m']
FRESH = list(range(230, 260))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same(a, b, label):
    if a is None or b is None:
        require(a is None and b is None, label)
    else:
        require(math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-13),
                f'{label}: {a} != {b}')


def verify_hashes(mapping, root):
    for name, expected in mapping.items():
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        require(digest(path) == expected, f'Hash mismatch: {path}')
    return len(mapping)


def main():
    destination = ROOT / 'results/final_audit.json'
    if destination.exists():
        raise RuntimeError('Final audit already exists; do not overwrite evidence')
    ledger = read(ROOT / 'results/final_evaluation.json')
    require(ledger['status'] == 'complete', 'Final evaluation must be complete before audit')
    summary = read(ROOT / 'results/gate_summary.json')
    require(ledger['summary_sha256'] == digest(ROOT / 'results/gate_summary.json'),
            'Ledger summary seal mismatch')
    require(summary['primary_metric'] == PRIMARY, 'Wrong primary metric')
    require(summary['primary_group'] == 'moving_coast', 'Wrong primary group')
    checkpoint_seal = read(ROOT / 'results/checkpoint_seal.json')
    protocol = read(ROOT / 'results/model_protocol.json')
    normalizer = read(ROOT / 'results/normalizer.json')
    train_floor = read(ROOT / 'results/physics_floor.json')
    test_floor = read(ROOT / 'results/test_physics_floor.json')
    require(normalizer['fit_cubes'] == list(range(200)), 'Normalizer includes unexpected cubes')
    require(normalizer['fit_episodes'] == list(range(8)), 'Normalizer includes validation episodes')
    scale = normalizer['horizontal_velocity_delta_mse_scale']
    same(scale, train_floor['normalization']['scalar_horizontal_delta_mse'], 'Independent train scale')
    same(scale, test_floor['normalization_scale'], 'Fresh-test physics scale')
    require(test_floor['test_cubes'] == FRESH, 'Physics used wrong test IDs')
    require(protocol['test_cubes'] == FRESH, 'Model protocol used wrong test IDs')
    require(summary['test_cubes'] == 30 and summary['test_episodes'] == 300,
            'Unexpected final test dimensions')
    hash_counts = {}
    hash_counts['checkpoint_files'] = verify_hashes(checkpoint_seal['checkpoints'], ROOT / 'checkpoints')
    hash_counts['training_source_manifest'] = verify_hashes(checkpoint_seal['manifest'], ROOT)
    hash_counts['test_data_manifest'] = verify_hashes(read(ROOT / 'results/test_data_seal.json'), ROOT)
    hash_counts['collection_source_manifest'] = verify_hashes(read(ROOT / 'results/source_seal.json'), ROOT)
    hash_counts['data_definitions'] = verify_hashes(read(ROOT / 'data/seal.json'), ROOT / 'data')
    hash_counts['retired_experiment_files'] = verify_hashes(read(ROOT / 'results/retired_experiment_seal.json'), ROOT)
    hash_counts['pretraining_review_files'] = verify_hashes(read(ROOT / 'results/pretraining_review.json')['hashes'], ROOT)
    require(hash_counts['checkpoint_files'] == 10, 'Expected ten frozen checkpoints')
    require(hash_counts['retired_experiment_files'] == 626, 'Unexpected retired evidence manifest size')
    require(checkpoint_seal['protocol_sha256'] == digest(ROOT / 'results/model_protocol.json'),
            'Checkpoint model protocol seal mismatch')
    require(summary['checkpoint_seal_sha256'] == digest(ROOT / 'results/checkpoint_seal.json'),
            'Summary checkpoint seal mismatch')
    require(summary['test_physics_floor_sha256'] == digest(ROOT / 'results/test_physics_floor.json'),
            'Summary physics seal mismatch')
    for key, name in [('checkpoint_seal_sha256', 'results/checkpoint_seal.json'),
                      ('data_seal_sha256', 'data/seal.json'),
                      ('test_data_seal_sha256', 'results/test_data_seal.json'),
                      ('source_sha256', 'models_gate.py')]:
        require(ledger['context'][key] == digest(ROOT / name), f'Ledger context {key}')
    require(train_floor['protocol_sha256'] == digest(ROOT / 'REVISED_PROTOCOL.md'), 'Physics protocol changed')
    require(train_floor['source_sha256'] == digest(ROOT / 'physics_floor.py'), 'Physics source changed')

    records = {}
    pair_audits = []
    all_checkpoints = []
    for seed in SEEDS:
        models = {}
        logs = {}
        for kind in KINDS:
            path = ROOT / 'checkpoints' / f'{kind}_seed_{seed}.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            training = read(ROOT / 'results/training_logs' / f'{kind}_seed_{seed}.json')
            result = read(ROOT / 'results/test_metrics' / f'{kind}_seed_{seed}.json')
            require(checkpoint['kind'] == kind and checkpoint['seed'] == seed, 'Checkpoint identity')
            require(checkpoint['manifest'] == checkpoint_seal['manifest'], 'Checkpoint manifest mismatch')
            require(checkpoint['protocol'] == protocol, 'Checkpoint protocol mismatch')
            require(checkpoint['normalizer'] == normalizer, 'Checkpoint normalizer mismatch')
            require(checkpoint['log'] == training['log'], 'Checkpoint/JSON training log mismatch')
            updates = checkpoint['completed_updates']
            require(updates % 10000 == 0 and 10000 <= updates <= 100000, 'Training budget invalid')
            require([row['update'] for row in checkpoint['log']] == list(range(500, updates+1, 500)),
                    'Validation checks missing or duplicated')
            selected = min(checkpoint['log'], key=lambda row: row['validation_nmse'])
            require(checkpoint['selected_update'] == selected['update'], 'Checkpoint not validation global minimum')
            same(checkpoint['selected_validation_nmse'], selected['validation_nmse'], 'Selected validation NMSE')
            for row in checkpoint['log']:
                same(row['validation_nmse'], row['validation_horizontal_velocity_delta_nmse'],
                     'Validation selection metric changed')
                prior_best = min(x['validation_nmse'] for x in checkpoint['log'] if x['update'] <= row['update'])
                same(row['best_validation_nmse'], prior_best, 'Running validation best')
                expected_lr = .0001 + (.001-.0001) * .5 * (1 + math.cos(math.pi * min(row['update']-1, 10000)/10000))
                same(row['learning_rate'], expected_lr, 'Learning-rate schedule')
            require(result['checkpoint_sha256'] == digest(path), 'Scored checkpoint mismatch')
            require(result['checkpoint_selected_update'] == selected['update'], 'Scored checkpoint selection mismatch')
            require(result['training_budget_updates'] == updates, 'Scored budget mismatch')
            require(result['test_cubes'] == 30 and result['test_episodes'] == 300, 'Incomplete test dimensions')
            models[kind] = checkpoint
            logs[kind] = checkpoint['log']
            records[(kind, seed)] = result
            all_checkpoints.append(path)
        require(models['plain']['completed_updates'] == models['oracle']['completed_updates'], 'Pair budget differs')
        require(models['plain']['minibatch_sha256'] == models['oracle']['minibatch_sha256'], 'Pair batch stream differs')
        updates = models['plain']['completed_updates']
        recomputed = []
        for endpoint in range(10000, updates+1, 10000):
            families = {}
            for kind in KINDS:
                before = min(r['validation_nmse'] for r in logs[kind] if r['update'] <= endpoint-2500)
                recent = min(r['validation_nmse'] for r in logs[kind] if endpoint-2500 < r['update'] <= endpoint)
                families[kind] = dict(previous_best=before, final2500_best=recent,
                                      relative_improvement=(before-recent)/before if before > 0 else 0.,
                                      still_improving=bool(before > 0 and recent <= .999*before))
            decision = dict(endpoint=endpoint, extend=any(v['still_improving'] for v in families.values()), families=families)
            recomputed.append(decision)
            if endpoint < updates:
                require(decision['extend'], 'Training extended despite predeclared plateau')
        require(recomputed == models['plain']['extension_decisions'], 'Plain extension arithmetic mismatch')
        require(recomputed == models['oracle']['extension_decisions'], 'Oracle extension arithmetic mismatch')
        if updates < 100000:
            require(not recomputed[-1]['extend'], 'Training stopped while extension rule still held')
        cap = updates == 100000 and recomputed[-1]['extend']
        require(models['plain']['convergence_not_established_at_cap'] == cap, 'Cap warning mismatch')
        require(models['oracle']['convergence_not_established_at_cap'] == cap, 'Cap warning mismatch')
        pair_record = read(ROOT / 'results/training_logs' / f'pair_seed_{seed}.json')
        require(pair_record['extension_decisions'] == recomputed, 'Pair ledger extension mismatch')
        require(pair_record['completed_updates'] == updates, 'Pair ledger budget mismatch')
        require(pair_record['minibatch_sha256'] == models['plain']['minibatch_sha256'], 'Pair ledger minibatches mismatch')
        pair_audits.append(dict(seed=seed, completed_updates=updates,
                               selected_updates={k:models[k]['selected_update'] for k in KINDS},
                               extension_decisions_verified=len(recomputed),
                               convergence_not_established_at_cap=cap))

    nonfinite = 0
    checked_cells = 0
    for group in GROUPS:
        for metric in METRICS:
            arrays = {}
            for kind in KINDS:
                values = []
                for seed in SEEDS:
                    cell = records[(kind, seed)]['groups'][group][metric]
                    require(cell['cubes'] == 30 and cell['episodes'] == 300, 'Missing test group coverage')
                    require([c['cube_id'] for c in cell['per_cube']] == FRESH, 'Wrong per-cube test IDs')
                    require(all(c['episodes'] == 10 for c in cell['per_cube']), 'Missing per-cube episodes')
                    per_cube = [c['mean'] for c in cell['per_cube']]
                    expected = None if any(x is None for x in per_cube) else float(np.mean(per_cube))
                    same(cell['mean'], expected, 'Per-cube aggregate mean')
                    require(not cell['missing_cube_ids'], 'Unexpected missing cube IDs')
                    nonfinite += cell['nonfinite_starts']
                    if metric == PRIMARY and cell['mean'] is not None:
                        same(cell['raw_mse_m2_s2'], cell['mean']*scale, 'Raw horizontal MSE')
                        same(cell['rmse_m_s'], math.sqrt(cell['mean']*scale), 'Horizontal RMSE')
                    values.append(cell['mean'])
                    checked_cells += 1
                reported = summary['comparisons'][group][metric][kind]
                require(values == reported['seed_values'], 'Reported seed values mismatch')
                defined = all(v is not None for v in values)
                arrays[kind] = np.asarray(values, dtype=float) if defined else None
                same(reported['mean'], float(np.mean(values)) if defined else None, 'Seed mean')
                same(reported['sample_sd'], float(np.std(values, ddof=1)) if defined else None, 'Sample seed SD')
            cell = summary['comparisons'][group][metric]
            p, o = arrays['plain'], arrays['oracle']
            if p is not None and o is not None:
                gap = p-o
                same(cell['mean_gap'], float(gap.mean()), 'Mean Plain–Oracle gap')
                same(cell['relative_mean_improvement'], float((p.mean()-o.mean())/p.mean()), 'Relative improvement')
                same(cell['paired_gap_sample_sd'], float(gap.std(ddof=1)), 'Paired gap SD')
                for got, expected in zip(cell['paired_seed_improvements'], (p-o)/p):
                    same(got, expected, 'Paired relative improvement')
                separated = bool(o.mean()+o.std(ddof=1) < p.mean()-p.std(ddof=1))
                require(cell['gap_exceeds_sum_of_seed_sds'] == separated, 'Seed-spread rule mismatch')
                require(cell['gap_exceeds_paired_gap_sd'] == bool(gap.mean() > gap.std(ddof=1)), 'Paired SD diagnostic mismatch')
            else:
                require(not cell['defined'] and not cell['gap_exceeds_sum_of_seed_sds'], 'Undefined result treated as pass')
    primary = summary['comparisons']['moving_coast'][PRIMARY]
    at_least_twenty = bool(primary['defined'] and primary['oracle']['mean'] <= .8*primary['plain']['mean'])
    spread = primary['gap_exceeds_sum_of_seed_sds']
    passed = at_least_twenty and spread
    require(summary['at_least_twenty_percent'] == at_least_twenty, '20% gate criterion mismatch')
    require(summary['beyond_seed_spread'] == spread, 'Gate spread criterion mismatch')
    require(summary['primary_gate_passed'] == passed and ledger['primary_gate_passed'] == passed, 'Gate decision mismatch')
    require(summary['all30testcubes_covered'], 'Gate coverage flag false')
    for name in ['requested_unclipped', 'stop_clipped_diagnostic']:
        physics = test_floor['metrics'][name]
        require(physics['cubes'] == 30 and physics['episodes'] == 300, 'Physics test coverage')
        same(physics['nmse'], physics['raw_mse_m2_s2']/scale, 'Physics NMSE denominator')
        same(physics['rmse_m_s'], math.sqrt(physics['raw_mse_m2_s2']), 'Physics RMSE')
        comparison = summary['physics_comparison'][name]
        same(comparison['physics_nmse'], physics['nmse'], 'Physics comparison baseline')
        oracle_mean = primary['oracle']['mean']
        same(comparison['oracle_mean_nmse'], oracle_mean, 'Oracle physics comparison mean')
        same(comparison['oracle_excess_nmse'], oracle_mean-physics['nmse'], 'Oracle excess error')
        same(comparison['oracle_to_physics_error_ratio'], oracle_mean/physics['nmse'], 'Oracle to physics ratio')
        same(comparison['oracle_rmse_from_mean_mse_m_s'], math.sqrt(oracle_mean*scale), 'Oracle RMSE from mean MSE')

    chronology = {name:(ROOT/name).stat().st_mtime for name in [
        'REVISED_PROTOCOL.md', 'results/physics_floor.json', 'PHYSICS_FLOOR_REPORT.md',
        'results/pretraining_review.json', 'results/normalizer.json', 'results/checkpoint_seal.json']}
    require(chronology['REVISED_PROTOCOL.md'] <= chronology['results/physics_floor.json'] <=
            chronology['PHYSICS_FLOOR_REPORT.md'] <= chronology['results/pretraining_review.json'] <=
            chronology['results/normalizer.json'], 'Pretraining protocol/floor chronology mismatch')
    require(chronology['results/normalizer.json'] <= min(p.stat().st_mtime for p in all_checkpoints),
            'Normalizer saved after checkpoint')
    require(max(p.stat().st_mtime for p in all_checkpoints) <= chronology['results/checkpoint_seal.json'] <=
            ledger['started_unix_seconds'] <= ledger['completed_unix_seconds'], 'Freeze/evaluation chronology mismatch')
    chronology['final_evaluation_started'] = ledger['started_unix_seconds']
    chronology['final_evaluation_completed'] = ledger['completed_unix_seconds']
    output = dict(status='passed', audited_unix_seconds=time.time(), audit_source_sha256=digest(__file__),
                  scope='Saved artifacts and arithmetic only; no simulation, prediction, model fitting, or repeated test evaluation.',
                  checked_hash_counts=hash_counts, seed_group_metric_cells_checked=checked_cells,
                  test_cubes=FRESH, test_episodes=300, nonfinite_scored_start_metric_entries=nonfinite,
                  normalization_scalar=scale, paired_training=pair_audits,
                  gate=dict(passed=passed, relative_improvement=primary['relative_mean_improvement'],
                            mean_gap=primary['mean_gap'],
                            sum_seed_sds=primary['plain']['sample_sd']+primary['oracle']['sample_sd'],
                            at_least_twenty_percent=at_least_twenty, beyond_seed_spread=spread),
                  chronology=chronology,
                  chronology_limit='Filesystem modification times corroborate ordering; pretraining manifests also bind unchanged protocol and physics report into each checkpoint.',
                  evidence_sha256={name:digest(ROOT/name) for name in [
                      'results/gate_summary.json','results/final_evaluation.json','results/checkpoint_seal.json',
                      'results/model_protocol.json','results/normalizer.json','results/physics_floor.json',
                      'results/test_physics_floor.json','results/retired_experiment_seal.json']})
    with destination.open('x') as f:
        json.dump(output, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(status='passed', gate=output['gate'], path=str(destination))), flush=True)


if __name__ == '__main__':
    main()
