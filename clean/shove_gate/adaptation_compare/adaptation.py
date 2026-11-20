"""Step 3 validation and adaptation using immutable Step 2 checkpoints.

No original dynamics checkpoint is trained or overwritten. Test evaluation is
owned by evaluation.py; this module's CLI reads training/development cubes only.
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
from torch import nn

from clean.shove_gate.push_latent import models as base
from clean.shove_gate.push_latent import evaluation as base_eval

ROOT = Path(__file__).resolve().parent
STEP2 = ROOT.parent / 'push_latent'
SEEDS = [101, 211, 307, 401, 503]
Z_DIMS = [4, 8, 16]
PENALTIES = [.0001, .001, .01, .1, 1.]
FINETUNE_LRS = [.00001, .0001]
LATENT_LR = .01
UPDATES = 50
VALIDATION_IDS = list(range(180, 200))
COMPONENTS = ['push_velocity_nmse', 'coast_velocity_nmse', 'full_state_nmse']
METRICS = COMPONENTS + ['composite']
PROTOCOL_SHA256 = 'b7c290b7e65cad349cb22cacd278e8f0c4b81d7316b65195f0237d1cb3053675'

read = base.read
digest = base.digest
write_new = base.write_new
write_atomic = base.write_atomic
configure_torch = base.configure_torch
model_state_digest = base.model_state_digest


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_frozen(variant, seed, device, base_root=STEP2):
    """Load a new in-memory copy after checking its immutable source hash."""
    require(variant in ['plain', 'oracle', 'history', 'z4', 'z8', 'z16'], 'Unknown Step3 variant')
    require(seed in SEEDS, 'Unknown paired seed')
    base_root = Path(base_root)
    path = base_root / 'checkpoints' / f'{variant}_seed_{seed}.pt'
    expected = read(base_root / 'results/checkpoint_seal.json')['checkpoints'][path.name]
    require(digest(path) == expected, f'Frozen checkpoint changed: {path}')
    model, checkpoint = base.load_checkpoint(path, device)
    require(checkpoint['variant'] == variant and checkpoint['seed'] == seed, 'Checkpoint identity mismatch')
    require(checkpoint['normalizer'] == read(base_root / 'results/normalizer.json'), 'Checkpoint scaler changed')
    return model, checkpoint, path


def load_episodes(cube_ids, episode_ids, root=ROOT, allow_test=False):
    """Read Step3's prepared copies/links with a strict retired-test boundary."""
    root = Path(root)
    cube_ids, episode_ids = list(cube_ids), list(episode_ids)
    require(len(set(cube_ids)) == len(cube_ids) and len(set(episode_ids)) == len(episode_ids), 'Duplicate requested identities')
    require(all(c in list(range(200)) + list(range(290, 320)) for c in cube_ids), 'Unknown or retired requested cube')
    require(all(e in range(10) for e in episode_ids), 'Unknown requested episode')
    if any(c >= 290 for c in cube_ids):
        require(allow_test, 'Fresh test motion cannot enter validation')
        ledger = root / 'results/final_evaluation.json'
        require(ledger.exists() and read(ledger).get('status') in ['begun', 'complete'], 'Fresh test access requires final evaluation ledger')
    episodes = []
    for c in cube_ids:
        for e in episode_ids:
            path = root / 'data/episodes' / base.filename(c, e)
            with np.load(path, allow_pickle=False) as arr:
                require(np.all(arr['cube_id'] == c) and np.all(arr['episode_id'] == e), 'Loaded episode identity mismatch')
                episodes.append(dict(cube_id=c, episode_id=e,
                                     state=np.asarray(arr['state'], dtype=np.float32),
                                     action=np.asarray(arr['action'], dtype=np.float32),
                                     next_state=np.asarray(arr['next_state'], dtype=np.float32),
                                     phase=np.asarray(arr['phase']).copy()))
    validate_episodes(episodes)
    return episodes


def validate_episodes(episodes, support=False):
    require(bool(episodes), 'Empty episode population')
    seen = set()
    for ep in episodes:
        c, e = int(ep['cube_id']), int(ep['episode_id'])
        require(c in list(range(200)) + list(range(290, 320)), 'Unknown or retired cube identity')
        require(e in range(10), 'Unknown episode identity')
        require((c, e) not in seen, 'Duplicate episode identity')
        seen.add((c, e))
        if support:
            require(e in range(5), 'Query episode cannot enter adaptation')
        x, a, nxt = ep['state'], ep['action'], ep['next_state']
        require(len(x) > 0 and x.shape == nxt.shape and x.shape[1:] == (15,) and a.shape == (len(x), 3), 'Invalid episode shape')
        require(all(np.isfinite(v).all() for v in [x, a, nxt]), 'Nonfinite episode observations')
        require(np.isin(ep['phase'], ['push', 'coast']).all(), 'Unknown phase label')


def derive_scales(episodes, normalizer):
    """Scoring-only energies; no input/output normalization is refitted."""
    validate_episodes(episodes)
    identities = {(int(e['cube_id']), int(e['episode_id'])) for e in episodes}
    require(identities == {(c, e) for c in range(180) for e in range(8)}, 'Metric scales require exactly the original fitting population')
    x = np.concatenate([e['state'] for e in episodes]).astype(np.float64)
    nxt = np.concatenate([e['next_state'] for e in episodes]).astype(np.float64)
    phase = np.concatenate([e['phase'] for e in episodes])
    row_episode = np.repeat(np.arange(len(episodes)), [len(e['state']) for e in episodes])
    cube_ids = np.array([e['cube_id'] for e in episodes])
    moving = (np.linalg.norm(x[:, 9:12], axis=1) > .001) | (np.linalg.norm(x[:, 12:15], axis=1) > .05)
    std = np.asarray(normalizer['state_std'], dtype=np.float32).astype(np.float64)
    horizontal = ((nxt[:, 9:11] - x[:, 9:11]) ** 2).mean(1)
    full = (((nxt - x) / std) ** 2).mean(1)
    coverage = {
        'push': base_eval.aggregate(horizontal, row_episode, cube_ids, phase == 'push'),
        'moving_coast': base_eval.aggregate(horizontal, row_episode, cube_ids, (phase == 'coast') & moving),
        'full_state': base_eval.aggregate(full, row_episode, cube_ids, np.ones(len(x), dtype=bool)),
    }
    require(all(r['cubes'] == 180 and r['episodes'] == 1440 for r in coverage.values()), 'Incomplete metric-scale coverage')
    values = [coverage[k]['mean'] for k in ['push', 'moving_coast', 'full_state']]
    require(all(math.isfinite(v) and v > 0 for v in values), 'Invalid metric energy')
    require(values[0] == normalizer['horizontal_velocity_delta_mse_scale'], 'Push energy differs from frozen Step2 energy')
    return dict(push_velocity_energy=values[0], coast_velocity_energy=values[1], full_state_energy=values[2],
                fit_cube_ids=list(range(180)), fit_episode_ids=list(range(8)), coverage=coverage,
                state_std_precision='The frozen network float32 state_std, promoted to float64 for scoring.',
                zero_change_fitting_scores={key: 1. for key in METRICS})


def _support_pack(episodes, normalizer, device):
    validate_episodes(episodes, support=True)
    pack = base.pack_episodes(episodes, normalizer, None, device, include_history=False)
    cube_ids = sorted({int(e['cube_id']) for e in episodes})
    lookup = {c: i for i, c in enumerate(cube_ids)}
    routing = torch.tensor([lookup[int(c)] for c in pack['row_cube']], device=device)
    counts = {c: sum(int(e['cube_id']) == c for e in episodes) for c in cube_ids}
    # Every cube's data loss is a mean over its episodes, then their steps.
    weights = 1. / (pack['lengths'][pack['row_episode']] * np.array([counts[int(c)] for c in pack['row_cube']]))
    weight = torch.tensor(weights, dtype=torch.float64, device=device)
    require(np.allclose(np.bincount(np.asarray(routing.cpu()), weights=weights), 1.), 'Support weighting is not equal by cube and episode')
    return pack, cube_ids, routing, weight


def _finite_gradients(parameters):
    for p in parameters:
        require(p.grad is not None and bool(torch.isfinite(p.grad).all()), 'Missing or nonfinite adaptation gradient')


def fit_latents(model, support_episodes, normalizer, device, penalty, lr=LATENT_LR, updates=UPDATES):
    """Fit independent new object vectors, with all shared parameters frozen."""
    require(model.z_dim > 0 and penalty > 0 and lr > 0 and updates > 0, 'Invalid latent fitting configuration')
    pack, cube_ids, routing, weights = _support_pack(support_episodes, normalizer, device)
    before = model_state_digest(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    mean = model.embedding.weight.detach().mean(0)
    z = nn.Parameter(mean.repeat(len(cube_ids), 1).clone())
    optimizer = torch.optim.Adam([z], lr=float(lr))
    records = []
    started = time.perf_counter()
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        predicted = model(pack['x'], pack['action'], z=z[routing])
        require(bool(torch.isfinite(predicted).all()), 'Nonfinite latent support output')
        per_row = (predicted - pack['target']).square().mean(1)
        data_loss = (per_row.double() * weights).sum()
        prior = float(penalty) * (z - mean).square().mean(1).sum()
        loss = data_loss + prior
        require(bool(torch.isfinite(loss)), 'Nonfinite latent adaptation loss')
        loss.backward()
        _finite_gradients([z])
        optimizer.step()
        require(bool(torch.isfinite(z).all()), 'Nonfinite fitted z')
        records.append(dict(update=update, mean_cube_data_loss=float(data_loss.detach()) / len(cube_ids),
                            mean_cube_prior=float(prior.detach()) / len(cube_ids),
                            mean_cube_total_loss=float(loss.detach()) / len(cube_ids)))
    require(model_state_digest(model) == before, 'Shared weights or embedding rows changed during z fitting')
    require(all(p.grad is None for p in model.parameters()), 'Shared network accumulated latent-fitting gradients')
    return dict(cube_ids=cube_ids, z_values=z.detach().cpu().numpy().copy(), loss_log=records,
                initial_mean_z=mean.cpu().numpy().tolist(), model_state_sha256=before, weights_unchanged=True,
                learning_rate=float(lr), penalty=float(penalty), updates=updates,
                support_episode_ids=sorted({int(e['episode_id']) for e in support_episodes}),
                elapsed_seconds=time.perf_counter() - started)


def finetune(model, support_episodes, normalizer, device, lr, updates=UPDATES):
    """Update a provided Plain copy; fresh Adam moments at every invocation."""
    require(model.variant == 'plain' and lr > 0 and updates > 0, 'Fine-tuning requires a Plain copy and positive settings')
    pack, cube_ids, _, weights = _support_pack(support_episodes, normalizer, device)
    require(len(cube_ids) == 1, 'Fine-tuning event must use exactly one cube')
    before = model_state_digest(model)
    model.train()
    parameters = list(model.parameters())
    for p in parameters:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=float(lr))
    records = []
    started = time.perf_counter()
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(pack['x'], pack['action'])
        require(bool(torch.isfinite(prediction).all()), 'Nonfinite fine-tuning support output')
        loss = ((prediction - pack['target']).square().mean(1).double() * weights).sum()
        require(bool(torch.isfinite(loss)), 'Nonfinite fine-tuning loss')
        loss.backward()
        _finite_gradients(parameters)
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in parameters), 'Nonfinite fine-tuned weights')
        records.append(dict(update=update, full_state_support_loss=float(loss.detach())))
    model.eval()
    return dict(cube_id=cube_ids[0], learning_rate=float(lr), updates=updates, optimizer_reset=True,
                before_state_sha256=before, after_state_sha256=model_state_digest(model),
                loss_log=records, all_parameter_tensors_receive_gradients=True,
                support_episode_ids=sorted({int(e['episode_id']) for e in support_episodes}),
                elapsed_seconds=time.perf_counter() - started)


def _aggregate(values, pack, mask):
    result = base_eval.aggregate(values, pack['row_episode'], pack['cube_ids'], mask)
    for row in result['per_episode']:
        row['episode_id'] = int(pack['episode_ids'][row['episode_row']])
    require(result['cubes'] == len(set(pack['cube_ids'].tolist())) and result['episodes'] == len(pack['cube_ids']), 'Incomplete scored episode/group coverage')
    return result


@torch.no_grad()
def score(model, episodes, normalizer, scales, device, z_by_cube=None, properties=None, batch_size=4096):
    """One-step scores only, with physical labels restricted to Oracle."""
    validate_episodes(episodes)
    if model.variant == 'oracle':
        require(properties is not None, 'Oracle requires true context')
    else:
        require(properties is None, 'Physical labels may not enter a non-Oracle scorer')
    pack = base.pack_episodes(episodes, normalizer, properties, device, include_history=model.variant == 'history')
    z = None
    if model.z_dim:
        require(z_by_cube is not None, 'Latent scoring requires explicit cube vectors')
        require(set(map(int, pack['cube_ids'])).issubset(z_by_cube), 'Missing cube latent')
        z = torch.tensor(np.stack([z_by_cube[int(c)] for c in pack['row_cube']]), dtype=torch.float32, device=device)
        require(z.shape == (len(pack['x']), model.z_dim), 'Unexpected latent scoring shape')
    else:
        require(z_by_cube is None, 'A non-latent model cannot receive z')
    std = np.asarray(normalizer['state_std'], dtype=np.float32).astype(np.float64)
    norm_std = torch.tensor(std, device=device, dtype=torch.float64)
    full_errors, horizontal_errors = [], []
    model.eval()
    for start in range(0, len(pack['x']), batch_size):
        sl = slice(start, start + batch_size)
        output = model(pack['x'][sl], pack['action'][sl],
                       context=pack['context'][sl] if model.variant == 'oracle' else None,
                       history=pack['history'][sl] if model.variant == 'history' else None,
                       z=None if z is None else z[sl])
        require(bool(torch.isfinite(output).all()), 'Nonfinite one-step prediction')
        true_delta = torch.tensor(pack['next_state'][sl].astype(np.float64) - pack['state'][sl].astype(np.float64), device=device)
        residual = output.double() * norm_std - true_delta
        horizontal_errors.append(residual[:, 9:11].square().mean(1).cpu().numpy())
        full_errors.append((residual / norm_std).square().mean(1).cpu().numpy())
    horizontal, full = np.concatenate(horizontal_errors), np.concatenate(full_errors)
    require(np.isfinite(horizontal).all() and np.isfinite(full).all(), 'Nonfinite one-step metric')
    masks = dict(pack['masks'], all_coast=pack['phase'] == 'coast')
    raw = {
        'raw_push_velocity_mse': _aggregate(horizontal, pack, masks['push']),
        'raw_coast_velocity_mse': _aggregate(horizontal, pack, masks['moving_coast']),
        'raw_all_coast_velocity_mse': _aggregate(horizontal, pack, masks['all_coast']),
        'legacy_full_state_mse': _aggregate(full, pack, masks['all_steps']),
    }
    aggregates = dict(raw)
    for name, values, mask, energy in [
        ('push_velocity_nmse', horizontal, masks['push'], scales['push_velocity_energy']),
        ('coast_velocity_nmse', horizontal, masks['moving_coast'], scales['coast_velocity_energy']),
        ('all_coast_velocity_nmse', horizontal, masks['all_coast'], scales['coast_velocity_energy']),
        ('full_state_nmse', full, masks['all_steps'], scales['full_state_energy']),
    ]:
        require(math.isfinite(energy) and energy > 0, 'Invalid scoring energy')
        aggregates[name] = _aggregate(values / energy, pack, mask)
    ep_indices = [r['episode_row'] for r in aggregates[COMPONENTS[0]]['per_episode']]
    require(all([r['episode_row'] for r in aggregates[c]['per_episode']] == ep_indices for c in COMPONENTS), 'Composite component coverage mismatch')
    composite_ep = []
    for i, row in enumerate(aggregates[COMPONENTS[0]]['per_episode']):
        composite_ep.append(dict(episode_row=row['episode_row'], cube_id=row['cube_id'], episode_id=row['episode_id'],
                                 mean=float(np.mean([aggregates[c]['per_episode'][i]['mean'] for c in COMPONENTS]))))
    by_cube = {}
    for row in composite_ep:
        by_cube.setdefault(row['cube_id'], []).append(row['mean'])
    composite_cube = [dict(cube_id=c, mean=float(np.mean(v)), episodes=len(v)) for c, v in sorted(by_cube.items())]
    aggregates['composite'] = dict(mean=float(np.mean([r['mean'] for r in composite_cube])), cubes=len(composite_cube),
                                   episodes=len(composite_ep), per_cube=composite_cube, per_episode=composite_ep,
                                   missing_cube_ids=[], nonfinite_starts=0)
    require(np.isclose(aggregates['composite']['mean'], np.mean([aggregates[c]['mean'] for c in COMPONENTS]), rtol=1e-12, atol=1e-15), 'Composite arithmetic mismatch')
    return dict(scores={k: v['mean'] for k, v in aggregates.items()}, aggregates=aggregates,
                cube_ids=sorted(set(map(int, pack['cube_ids']))), episode_ids=sorted(set(map(int, pack['episode_ids']))),
                model_variant=model.variant, coast_population='moving coast', properties_used=model.variant == 'oracle')


def combine_cube_scores(records):
    """Combine independent one-cube B query scores with equal cube weighting."""
    require(bool(records), 'No cube scores to combine')
    cube_ids = [r['cube_ids'][0] for r in records]
    require(all(len(r['cube_ids']) == 1 for r in records) and len(set(cube_ids)) == len(cube_ids), 'Invalid independent cube scores')
    keys = list(records[0]['scores'])
    require(all(list(r['scores']) == keys for r in records), 'Inconsistent score components')
    aggregates = {}
    for k in keys:
        local = [r['aggregates'][k] for r in records]
        aggregates[k] = dict(mean=float(np.mean([v['mean'] for v in local])), cubes=len(local),
                             episodes=sum(v['episodes'] for v in local), per_cube=[v['per_cube'][0] for v in local],
                             per_episode=[dict(e) for v in local for e in v['per_episode']], missing_cube_ids=[], nonfinite_starts=0)
    return dict(scores={k: v['mean'] for k, v in aggregates.items()}, aggregates=aggregates,
                cube_ids=sorted(cube_ids), episode_ids=records[0]['episode_ids'], model_variant=records[0]['model_variant'],
                coast_population='moving coast', properties_used=False)


def select_configuration(a_records, b_records):
    a_expected = {(d, p, s) for d in Z_DIMS for p in PENALTIES for s in SEEDS}
    b_expected = {(lr, s) for lr in FINETUNE_LRS for s in SEEDS}
    require(len(a_records) == len(a_expected) and {(r['z_dim'], r['penalty'], r['seed']) for r in a_records} == a_expected, 'Incomplete or duplicated latent validation grid')
    require(len(b_records) == len(b_expected) and {(r['learning_rate'], r['seed']) for r in b_records} == b_expected, 'Incomplete or duplicated fine-tuning validation grid')
    for r in a_records + b_records:
        query = r['query']
        require(query['cube_ids'] == VALIDATION_IDS and query['episode_ids'] == list(range(5, 10)), 'Selection received non-validation/query identities')
        require(all(query['aggregates'][m]['cubes'] == 20 and query['aggregates'][m]['episodes'] == 100 for m in METRICS), 'Validation coverage is incomplete')
        require(all(math.isfinite(query['scores'][m]) for m in METRICS), 'Nonfinite validation score')
    def summary(rows):
        rows = sorted(rows, key=lambda r: r['seed'])
        return {m: base_eval.moments([r['query']['scores'][m] for r in rows]) for m in METRICS}
    a_candidates = [dict(z_dim=d, variant=f'z{d}', penalty=p, learning_rate=LATENT_LR, updates=UPDATES,
                         validation=summary([r for r in a_records if (r['z_dim'], r['penalty']) == (d, p)]))
                    for d in Z_DIMS for p in PENALTIES]
    b_candidates = [dict(learning_rate=lr, updates=UPDATES,
                         validation=summary([r for r in b_records if r['learning_rate'] == lr])) for lr in FINETUNE_LRS]
    chosen_a = min(a_candidates, key=lambda c: (c['validation']['composite']['mean'], c['z_dim'], -c['penalty']))
    chosen_b = min(b_candidates, key=lambda c: (c['validation']['composite']['mean'], c['learning_rate']))
    return dict(A=chosen_a, B=chosen_b, b_rates=b_candidates, a_candidates=a_candidates,
                selection_population=VALIDATION_IDS, selection_metric='composite', seeds=SEEDS,
                support_episodes=list(range(5)), query_episodes=list(range(5, 10)))


def _verify_ready(root):
    root = Path(root)
    require(digest(root / 'PROTOCOL.md') == PROTOCOL_SHA256, 'Written Step3 protocol changed')
    require(not (root / 'results/STOP.json').exists(), 'Prior STOP forbids another scientific run')
    require(not (root / 'results/final_evaluation.json').exists(), 'Validation after test evaluation is forbidden')
    review = read(root / 'results/prevalidation_review.json')
    require(review['status'] == 'passed', 'Prevalidation source/test review did not pass')
    for name, expected in review['source_sha256'].items():
        require(digest(root / name) == expected, f'Reviewed scientific source changed: {name}')
    from .collection import verify_prepared
    require(root.resolve() == ROOT.resolve(), 'Collection preparation belongs to the canonical Step3 root')
    verify_prepared()


def tune(root=ROOT, device_name='cpu'):
    root = Path(root)
    _verify_ready(root)
    require(not (root / 'results/selection_seal.json').exists(), 'Validation selection already frozen; no rerun')
    device = configure_torch(device_name)
    normalizer = read(STEP2 / 'results/normalizer.json')
    fitting = load_episodes(range(180), range(8), root=root)
    scales = derive_scales(fitting, normalizer)
    write_new(root / 'results/metric_scales.json', scales)
    del fitting
    support = load_episodes(VALIDATION_IDS, range(5), root=root)
    query = load_episodes(VALIDATION_IDS, range(5, 10), root=root)
    a_records, b_records = [], []
    started = time.perf_counter()
    for seed in SEEDS:
        for dim in Z_DIMS:
            model, checkpoint, path = load_frozen(f'z{dim}', seed, device)
            for penalty in PENALTIES:
                fitted = fit_latents(model, support, normalizer, device, penalty)
                vectors = {c: z for c, z in zip(fitted['cube_ids'], fitted['z_values'])}
                scored = score(model, query, normalizer, scales, device, z_by_cube=vectors)
                record = dict(seed=seed, z_dim=dim, penalty=penalty, learning_rate=LATENT_LR, updates=UPDATES, query=scored,
                              checkpoint_sha256=digest(path), model_state_sha256=fitted['model_state_sha256'],
                              z_values=fitted['z_values'].tolist(), cube_ids=fitted['cube_ids'], loss_log=fitted['loss_log'],
                              weights_unchanged=fitted['weights_unchanged'], elapsed_seconds=fitted['elapsed_seconds'])
                write_new(root / 'results/validation_runs' / f'z{dim}_seed{seed}_penalty{penalty:g}.json', record)
                a_records.append(record)
                print(json.dumps(dict(stage='validation_a', seed=seed, z_dim=dim, penalty=penalty, scores=scored['scores'])), flush=True)
            del model
        for lr in FINETUNE_LRS:
            cube_results, fit_logs = [], []
            for cid in VALIDATION_IDS:
                model, checkpoint, path = load_frozen('plain', seed, device)
                log = finetune(model, [e for e in support if e['cube_id'] == cid], normalizer, device, lr)
                result = score(model, [e for e in query if e['cube_id'] == cid], normalizer, scales, device)
                cube_results.append(result)
                fit_logs.append(log)
                del model
            record = dict(seed=seed, learning_rate=lr, updates=UPDATES, query=combine_cube_scores(cube_results),
                          fit_logs=fit_logs, checkpoint_sha256=digest(path))
            write_new(root / 'results/validation_runs' / f'plain_seed{seed}_lr{lr:g}.json', record)
            b_records.append(record)
            print(json.dumps(dict(stage='validation_b', seed=seed, learning_rate=lr, scores=record['query']['scores'])), flush=True)
    chosen = select_configuration(a_records, b_records)
    write_new(root / 'results/validation_grid.json', dict(a_records=a_records, b_records=b_records, selection=chosen,
              elapsed_seconds=time.perf_counter() - started))
    write_new(root / 'results/selected_config.json', chosen)
    _verify_ready(root)
    files = {name: digest(root / name) for name in ['PROTOCOL.md', 'adaptation.py', 'results/metric_scales.json',
             'results/validation_grid.json', 'results/selected_config.json', 'results/prevalidation_review.json']}
    for path in sorted((root / 'results/validation_runs').glob('*.json')):
        files[str(path.relative_to(root))] = digest(path)
    checkpoints = read(STEP2 / 'results/checkpoint_seal.json')['checkpoints']
    for name, expected in checkpoints.items():
        require(digest(STEP2 / 'checkpoints' / name) == expected, 'Base checkpoint changed during validation')
    write_new(root / 'results/selection_seal.json', dict(files=files, frozen_step2_checkpoints=checkpoints,
              completed_unix_seconds=time.time(), selected_config_sha256=digest(root / 'results/selected_config.json')))
    print(json.dumps(dict(stage='selection_frozen', A=chosen['A'], B=chosen['B'], b_rates=chosen['b_rates'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    try:
        tune(args.root, args.device)
    except Exception as exc:
        args.root.joinpath('results').mkdir(parents=True, exist_ok=True)
        write_atomic(args.root / 'results/STOP.json', dict(stage='validation', reason=f'{type(exc).__name__}: {exc}', time=time.time()))
        raise


if __name__ == '__main__':
    main()
