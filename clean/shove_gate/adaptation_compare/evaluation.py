"""Single frozen Step 3 evaluation; never selects settings from test scores."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import adaptation as ad
from ..push_latent import models as old

ROOT = Path(__file__).resolve().parent
SEEDS = [101, 211, 307, 401, 503]
TEST_IDS = list(range(290, 320))
RATES = [1e-5, 1e-4]
COMPONENTS = ['push_velocity_nmse', 'coast_velocity_nmse', 'full_state_nmse']
SCORES = COMPONENTS + ['composite']
SNAPSHOTS = [1, 5, 6, 10, 15, 16, 20, 25, 30]
RETURNS = [6, 16, 30]


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition, message):
    if not condition: raise RuntimeError(message)
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
def replace(path, value):
    path = Path(path); temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)
def rate_name(lr): return format(lr, '.0e')
def vector_sha(z): return hashlib.sha256(np.asarray(z, dtype=np.float32).tobytes()).hexdigest()


def moments(values):
    a = np.asarray(values, dtype=np.float64)
    require(a.shape == (5,) and np.isfinite(a).all(), 'Need five finite paired seed values')
    return dict(seed_values=a.tolist(), mean=float(a.mean()), sample_sd=float(a.std(ddof=1)))


def aggregate_models(records):
    return {key: moments([r['scores'][key] for r in records]) for key in SCORES}


def aggregate_cubes(records):
    require([r['cube_id'] for r in records] == TEST_IDS, 'Prefix cube coverage/order differs')
    names = list(records[0]['query']['scores'])
    result = {key: float(np.mean([r['query']['scores'][key] for r in records])) for key in names}
    require(np.isfinite(list(result.values())).all(), 'Nonfinite cube aggregate')
    return result


def cube_scores(score, cube_id):
    result = {}
    for key in score['scores']:
        rows = score['aggregates'][key]['per_cube']
        found = [r for r in rows if int(r['cube_id']) == cube_id]
        require(len(found) == 1 and found[0]['episodes'] == 5, 'Missing five query shoves')
        result[key] = float(found[0]['mean'])
    return dict(scores=result, cube_id=cube_id, query_shoves=list(range(5, 10)),
                derived_from_saved_per_cube_aggregate=True)


def p1_decision(a_series, b_series, metric='composite'):
    initial_a, initial_b = a_series['1']['first_cube'][metric], b_series['1']['first_cube'][metric]
    rows = {}
    for n in RETURNS:
        a = a_series[str(n)]['first_cube'][metric]; b = b_series[str(n)]['first_cube'][metric]
        av, ai = np.asarray(a['seed_values']), np.asarray(initial_a['seed_values'])
        bv, bi = np.asarray(b['seed_values']), np.asarray(initial_b['seed_values'])
        no_rise = bool(np.all(av <= ai + 1e-12 + 1e-10*np.abs(ai)))
        b_rise = bool(b['mean']-b['sample_sd'] > initial_b['mean']+initial_b['sample_sd'])
        rows[str(n-1)] = dict(total_visits=n, a_nonrise=no_rise, b_rise_beyond_seed_spread=b_rise,
            a_initial=initial_a, a_return=a, b_initial=initial_b, b_return=b,
            a_paired_change=moments(av-ai), b_paired_change=moments(bv-bi),
            passed=bool(no_rise and b_rise))
    return dict(passed=all(r['passed'] for r in rows.values()), by_other_cubes=rows)


def p2_decision(a_curve, b_curve, metric='composite'):
    target = b_curve['5'][metric]; earliest = None; comparisons = {}
    for k in range(1, 6):
        a = a_curve[str(k)][metric]
        reaches = bool(a['mean'] <= target['mean'])
        if reaches and earliest is None: earliest = k
        comparisons[str(k)] = dict(a=a, b_five=target, reaches=reaches,
            paired_difference=moments(np.asarray(a['seed_values'])-np.asarray(target['seed_values'])))
    return dict(passed=earliest is not None, earliest_prefix=earliest,
                interpretation='fewer shoves' if earliest is not None and earliest < 5 else
                               'matches or beats at five shoves' if earliest == 5 else 'fails to reach',
                comparisons=comparisons)


def build_summary(root, selected):
    root = Path(root); prefix = root/'results/prefix'; seqdir = root/'results/sequence'
    curves = {'A': {}, 'B': {rate_name(lr): {} for lr in RATES}}
    for k in range(1, 6):
        curves['A'][str(k)] = aggregate_models([read(prefix/f'A_seed{s}_k{k}.json')['query'] for s in SEEDS])
        for lr in RATES:
            key = rate_name(lr)
            curves['B'][key][str(k)] = aggregate_models([read(prefix/f'B_lr{key}_seed{s}_k{k}.json') for s in SEEDS])
    baselines = {}
    for name in ['plain', 'history', 'oracle']:
        recs = [read(root/f'results/baselines/{name}_seed{s}.json') for s in SEEDS]
        baselines[name] = dict(query=aggregate_models([r['query'] for r in recs]),
                              first_cube=aggregate_models([r['first_cube'] for r in recs]),
                              training_validation=aggregate_models([r['training_validation'] for r in recs]))
    sequence = {'A': {}, 'B': {rate_name(lr): {} for lr in RATES}}
    for label, key, names in [('A', None, [f'A_seed{s}.json' for s in SEEDS])] + [
            ('B', rate_name(lr), [f'B_lr{rate_name(lr)}_seed{s}.json' for s in SEEDS]) for lr in RATES]:
        recs = [read(seqdir/n) for n in names]
        dest = sequence[label] if key is None else sequence[label][key]
        dest['0'] = {'training_validation': aggregate_models([r['initial_training_validation'] for r in recs])}
        for n in SNAPSHOTS:
            rows = [[x for x in r['snapshots'] if x['visits'] == n][0] for r in recs]
            dest[str(n)] = {pop: aggregate_models([r[pop] for r in rows]) for pop in ['first_cube', 'training_validation']}
    primary_lr = rate_name(selected['B']['learning_rate'])
    all_criteria = {}
    for lr in RATES:
        key = rate_name(lr)
        all_criteria[key] = {m: dict(p1=p1_decision(sequence['A'], sequence['B'][key], m),
                                    p2=p2_decision(curves['A'], curves['B'][key], m)) for m in SCORES}
    primary = all_criteria[primary_lr]['composite']
    return dict(step=3, seeds=SEEDS, test_cubes=TEST_IDS, query_shoves=list(range(5, 10)),
        selected_config=selected, curves=curves, baselines=baselines, sequence=sequence,
        criteria=dict(primary_lr=primary_lr, p1=primary['p1'], p2=primary['p2'],
                      components={m: all_criteria[primary_lr][m] for m in COMPONENTS}, by_lr=all_criteria),
        all_criteria_passed=bool(primary['p1']['passed'] and primary['p2']['passed']),
        retention_interpretation='A retention follows frozen shared weights and independent stored z; not learned resistance to interference.')


def property_table(root):
    return {int(c['cube_id']): np.array([c['mass_kg'], c['combined_friction']], dtype=np.float32)
            for c in read(Path(root)/'data/cubes.json')['cubes']}


def one_seed(root, seed, selected, scales, device):
    prefix = root/'results/prefix'; seqdir = root/'results/sequence'
    support = ad.load_episodes(TEST_IDS, range(5), root=root, allow_test=True)
    query = ad.load_episodes(TEST_IDS, range(5, 10), root=root, allow_test=True)
    trainval = ad.load_episodes(range(180), [8, 9], root=root)
    supports = {c: [e for e in support if e['cube_id'] == c] for c in TEST_IDS}
    queries = {c: [e for e in query if e['cube_id'] == c] for c in TEST_IDS}
    selected_a = selected['A']; z_variant = f"z{selected_a['z_dim']}"
    a_model, a_cp, a_path = ad.load_frozen(z_variant, seed, device)
    norm = a_cp['normalizer']; a_base_sha = old.model_state_digest(a_model)
    trained_z = {c: a_model.embedding.weight[c].detach().cpu().tolist() for c in range(180)}
    plain_base, plain_cp, plain_path = ad.load_frozen('plain', seed, device)
    require(norm == plain_cp['normalizer'], 'Frozen normalizers differ')
    plain_sha = old.model_state_digest(plain_base)
    for name in ['plain', 'history', 'oracle']:
        model, cp, path = ad.load_frozen(name, seed, device)
        prop = property_table(root) if name == 'oracle' else None
        q = ad.score(model, query, cp['normalizer'], scales, device=device, properties=prop)
        tv = ad.score(model, trainval, cp['normalizer'], scales, device=device, properties=prop)
        write(root/f'results/baselines/{name}_seed{seed}.json', dict(seed=seed, model=name,
            checkpoint_sha256=sha(path), query=q, first_cube=cube_scores(q, 290), training_validation=tv))
        del model
    fits_a = {}
    for k in range(1, 6):
        fit = ad.fit_latents(a_model, [e for e in support if e['episode_id'] < k], norm,
                            lr=.01, updates=50, penalty=selected_a['penalty'], device=device)
        require(fit['cube_ids'] == TEST_IDS, 'Latent cube coverage/order differs')
        zs = {int(c): z for c, z in zip(fit['cube_ids'], fit['z_values'])}
        result = ad.score(a_model, query, norm, scales, device=device, z_by_cube=zs)
        write(prefix/f'A_seed{seed}_k{k}.json', dict(seed=seed, support_shoves=k,
            checkpoint_sha256=sha(a_path), fit={**fit, 'z_values': np.asarray(fit['z_values']).tolist()}, query=result))
        fits_a[k] = (fit, zs, result)
        print(json.dumps(dict(stage='prefix_A', seed=seed, support_shoves=k, scores=result['scores'])), flush=True)
    require(old.model_state_digest(a_model) == a_base_sha, 'A shared weights changed')
    for lr in RATES:
        key = rate_name(lr)
        for k in range(1, 6):
            records = []
            for c in TEST_IDS:
                model = copy.deepcopy(plain_base)
                require(old.model_state_digest(model) == plain_sha, 'Independent B failed to reset')
                fit = ad.finetune(model, [e for e in supports[c] if e['episode_id'] < k], norm,
                                  lr=lr, updates=50, device=device)
                result = ad.score(model, queries[c], norm, scales, device=device)
                records.append(dict(cube_id=c, fit=fit, query=result))
                del model
            record = dict(seed=seed, learning_rate=lr, support_shoves=k,
                          checkpoint_sha256=sha(plain_path), per_cube=records, scores=aggregate_cubes(records))
            write(prefix/f'B_lr{key}_seed{seed}_k{k}.json', record)
            print(json.dumps(dict(stage='prefix_B', seed=seed, learning_rate=lr, support_shoves=k,
                                  scores=record['scores'])), flush=True)
    a_train = ad.score(a_model, trainval, norm, scales, device=device, z_by_cube=trained_z)
    fit5, zs5, a_query = fits_a[5]
    first_a = cube_scores(a_query, 290); first_z_hash = vector_sha(zs5[290])
    a_snaps = []; a_visits = []; store = {}
    for visit, c in enumerate(TEST_IDS, start=1):
        store[c] = np.asarray(zs5[c], dtype=np.float32).copy()
        a_visits.append(dict(visit=visit, cube_id=c, query=cube_scores(a_query, c),
                             z_sha256=vector_sha(store[c]), cached_from_independent_five_shove_fit=True))
        if visit in SNAPSHOTS:
            require(old.model_state_digest(a_model) == a_base_sha, 'A weights changed during sequence')
            require(vector_sha(store[290]) == first_z_hash, 'Stored first-cube z changed')
            a_snaps.append(dict(visits=visit, other_cubes_seen=visit-1, first_cube=first_a,
                training_validation=a_train, weights_sha256=a_base_sha, z_first_sha256=first_z_hash,
                cached_predictions_verified_by_unchanged_hashes=True, stored_cube_ids=sorted(store)))
    write(seqdir/f'A_seed{seed}.json', dict(seed=seed, initial_training_validation=a_train,
        visits=a_visits, snapshots=a_snaps, checkpoint_sha256=sha(a_path),
        stored_z={str(c): store[c].tolist() for c in TEST_IDS}))
    for lr in RATES:
        key = rate_name(lr); model = copy.deepcopy(plain_base)
        initial = ad.score(model, trainval, norm, scales, device=device)
        snapshots = []; visits = []
        for visit, c in enumerate(TEST_IDS, start=1):
            before = old.model_state_digest(model)
            fit = ad.finetune(model, supports[c], norm, lr=lr, updates=50, device=device)
            current = ad.score(model, queries[c], norm, scales, device=device)
            after = old.model_state_digest(model)
            visits.append(dict(visit=visit, cube_id=c, fit=fit, query=current,
                               before_weights_sha256=before, after_weights_sha256=after))
            if visit in SNAPSHOTS:
                first = current if visit == 1 else ad.score(model, queries[290], norm, scales, device=device)
                tv = ad.score(model, trainval, norm, scales, device=device)
                checkpoint_path = root/f'results/sequence_weights/B_lr{key}_seed{seed}_visit{visit:02d}.pt'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                require(not checkpoint_path.exists(), 'Sequence snapshot already exists')
                torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, checkpoint_path)
                snapshots.append(dict(visits=visit, other_cubes_seen=visit-1, first_cube=first,
                    training_validation=tv, weights_sha256=after,
                    checkpoint_file=str(checkpoint_path.relative_to(root)), checkpoint_file_sha256=sha(checkpoint_path)))
                print(json.dumps(dict(stage='sequence_B', seed=seed, learning_rate=lr, visits=visit,
                    first_cube=first['scores'], training_validation=tv['scores'])), flush=True)
        write(seqdir/f'B_lr{key}_seed{seed}.json', dict(seed=seed, learning_rate=lr,
            checkpoint_sha256=sha(plain_path), initial_training_validation=initial, visits=visits,
            snapshots=snapshots, optimizer_policy='Fresh Adam each cube, weights retained'))
        del model
    require(old.model_state_digest(plain_base) == plain_sha, 'Original Plain copy changed')


def verify_files(root, manifest):
    for name, expected in manifest.items():
        p = Path(name); p = p if p.is_absolute() else Path(root)/p
        require(sha(p) == expected, f'Changed frozen file: {p}')


def run(root=ROOT, device_name='cuda'):
    from . import collection
    root = Path(root); results = root/'results'; ledger_path = results/'final_evaluation.json'
    require(not ledger_path.exists(), 'Evaluation already started; no automatic rerun')
    require(not (results/'STOP.json').exists(), 'STOP exists; no automatic rerun')
    preflight = read(results/'pretest_review.json')
    require(preflight['status'] == 'passed', 'Pretest review missing')
    verify_files(root, preflight['files'])
    selected = read(results/'selected_config.json'); scales = read(results/'metric_scales.json')
    ledger = dict(status='begun', started_unix_seconds=time.time(),
                  selected_config_sha256=sha(results/'selected_config.json'),
                  pretest_review_sha256=sha(results/'pretest_review.json'),
                  source_sha256=sha(Path(__file__)), protocol_sha256=sha(root/'PROTOCOL.md'))
    write(ledger_path, ledger)
    try:
        device = old.configure_torch(device_name)
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            one_seed(root, seed, selected, scales, device)
        verify_files(root, preflight['files'])
        summary = build_summary(root, selected); write(results/'summary.json', summary)
        ledger.update(status='complete', completed_unix_seconds=time.time(),
            summary_sha256=sha(results/'summary.json'),
            p1_passed=summary['criteria']['p1']['passed'], p2_passed=summary['criteria']['p2']['passed'],
            all_criteria_passed=summary['all_criteria_passed'])
        replace(ledger_path, ledger)
        print(json.dumps(dict(stage='complete', p1=ledger['p1_passed'], p2=ledger['p2_passed'])), flush=True)
    except BaseException as exc:
        ledger.update(status='stopped', stopped_unix_seconds=time.time(), error=repr(exc))
        replace(ledger_path, ledger)
        if not (results/'STOP.json').exists(): write(results/'STOP.json', ledger)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); run(device_name=args.device)
