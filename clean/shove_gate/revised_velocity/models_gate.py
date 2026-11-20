"""Revised horizontal-velocity Plain/Oracle gate on fresh test cubes.

All fitted statistics and checkpoint selection use training cubes only.
Test motion is opened only after all ten checkpoints have been sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from .physics_floor import compute_metrics as compute_physics_metrics

ROOT = Path(__file__).resolve().parent
SEEDS = [101, 211, 307, 401, 503]
KINDS = ['plain', 'oracle']
GROUPS = ['push', 'moving_coast', 'all_steps']
TEST_CUBES = list(range(230, 260))
PRIMARY_METRIC = 'horizontal_velocity_delta_nmse'
MODEL_PROTOCOL = {
    'version': 6, 'raw_coast_seconds': 1.5, 'state_dim': 15, 'action_dim': 3,
    'hidden_layers': [256, 256, 256], 'activation': 'ReLU',
    'seeds': SEEDS, 'families': KINDS,
    'updates': 10000, 'batch_size': 512, 'optimizer': 'Adam',
    'maximum_updates': 100000, 'extension_block_updates': 10000,
    'extension_rule': 'At every10000updateendpoint compare bestvalidation through endpoint−2500 with bestoverfinal2500. If eitherfamily improves at least0.1%, extendboth10000updates, up to100000. Report unconverged if cap reached while still improving.',
    'learning_rate': .001, 'final_learning_rate': .0001,
    'learning_rate_schedule': 'Cosine across first10000 updates, then constant0.0001 for any extension.',
    'determinism': 'Fixed NumPy and torch seeds; deterministic torch algorithms and CUDA workspace, cudnn benchmarking disabled. Reproduction assumes the same software and hardware.',
    'train_cubes': list(range(200)), 'test_cubes': TEST_CUBES,
    'retired_test_cubes': list(range(200, 230)),
    'train_episodes': list(range(8)), 'validation_episodes': [8, 9],
    'validation_interval': 500,
    'checkpoint_selection': 'Lowest validation moving-coast horizontal-velocity delta NMSE; equal cube and episode weight.',
    'normalization': 'State/action means and populationSD of retained training transitions only; eachSD floored at.001 nativeunits. Oracle context mean/SD from200training-cube labels, never test. Target(next_state-state)/stateSD.',
    'training_sampling': 'Uniform training episode, then uniform retained transition. Paired families restart the same seed-specific NumPy minibatch stream.',
    'inputs': {'plain': 'normalized state15 + normalized applied force3',
               'oracle': 'same + normalized true mass + normalized combined friction'},
    'forbidden_inputs': ['cube_id', 'episode_id', 'step', 'phase', 'trimmed_length', 'padding_mask', 'stop_time'],
    'score_groups': GROUPS,
    'moving_coast': 'phasecoast AND(norm(observed linearvelocity)>.001m/s OR norm(observed angularvelocity)>.05rad/s)',
    'horizontal_velocity_delta_nmse': 'Mean squared error of predicted minus observed horizontal velocity change over X/Y, divided by the equal-cube/episode mean observed horizontal-delta squared energy from training cubes episodes0..7 moving-coast only. A single fixed scalar is used for both coordinates and all groups. No target centering. Zero-delta prediction scores1 on that training group.',
    'one_step_nmse': 'Mean squared normalized next-state error over15statecoordinates; firstmeanwithineligible starts/episode, thenepisodes/cube, thenequalweightcubes.',
    'rollout10_nmse': 'Mean squared normalized stateerror acrossall10 recursivepredictions and15coordinates; no intermediate observedstate fed. Pushwindows entirelypush; movingcoaststarts observedmoving withall10stepscoast; all_steps windows cancrossphase. Rawfixedhorizon supplies targets beyondtrimmedfile.',
    'rollout10_final_nmse': 'Same normalization; final tenth predicted state only, diagnostic.',
    'stopping_position_error_m': 'Fromeveryretainedeligible start, recursivelypredict to preplannedrelease+1.5s. Compare terminalpredictedXY with observed confirmed-restXY. Horizonfromcommandduration only, nevertrimmedlength/truestoptime. Equalepisode/cubeaggregation.',
    'gate': 'Moving-coast horizontal_velocity_delta_nmse meanOracle<=.8*meanPlain AND meanOracle+sampleSDOracle<meanPlain-sampleSDPlain acrossfivepairedseeds. No substitutionaftertest.',
    'divergence': 'No clipping, projection, oracle correction or ground-truth resets. Nonfinite rollout rows make the corresponding aggregate metric undefined; report their count.',
    'test_policy': 'Hash held-out raw and episode bytes against collection records before training, without fitting/scoring on them. Freeze all ten checkpoints before decoding test motion for model scoring. One final evaluation ledger; resume only with identical sealed checkpoints and data. Training forbidden after ledger begins.',
}


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


def seal_protocol(root=ROOT):
    path = Path(root) / 'results/model_protocol.json'
    if path.exists():
        if read(path) != MODEL_PROTOCOL:
            raise RuntimeError('Frozen model protocol differs from implementation')
    else:
        write_new(path, MODEL_PROTOCOL)
    return path


def filename(cube_id, episode_id):
    return f'cube_{cube_id:03d}_episode_{episode_id:02d}.npz'


def verify_data_seal(root):
    for name, expected in read(root / 'data/seal.json').items():
        if digest(root / 'data' / name) != expected:
            raise RuntimeError(f'Changed sealed data definition: {name}')


def require_collection_ready(root):
    verify_data_seal(root)
    audit = read(root / 'results/full_audit.json')
    if audit.get('status') != 'passed':
        raise RuntimeError('Full collected-data audit has not passed')
    expected = {filename(c, e) for c in list(range(200)) + TEST_CUBES for e in range(10)}
    actual = {p.name for p in (root / 'data/episodes').glob('*.npz')}
    if actual != expected:
        raise RuntimeError('Full230cube×10episode dataset is incomplete or contains extra files')


def load_episodes(root, cube_ids, episode_ids, allow_test=False):
    if any(200 <= c < 230 for c in cube_ids):
        raise RuntimeError('Retired test cubes cannot enter the revised experiment')
    if any(c >= 200 for c in cube_ids) and not allow_test:
        raise RuntimeError('Test motion cannot be opened during fitting/validation')
    episodes = []
    for c in cube_ids:
        for e in episode_ids:
            path = root / 'data/episodes' / filename(c, e)
            with np.load(path, allow_pickle=False) as a:
                x, action, nxt = [np.asarray(a[k], dtype=np.float32) for k in ['state', 'action', 'next_state']]
                phase = np.asarray(a['phase']).copy()
                if x.shape != nxt.shape or x.shape[1] != 15 or action.shape != (len(x), 3):
                    raise ValueError(f'Unexpected direct-force schema: {path}')
                if not (np.isfinite(x).all() and np.isfinite(nxt).all() and np.isfinite(action).all()):
                    raise ValueError(f'Nonfinite observation: {path}')
                if not (np.all(a['cube_id'] == c) and np.all(a['episode_id'] == e)):
                    raise ValueError('Filename/metadata identity mismatch')
            episodes.append(dict(cube_id=c, episode_id=e, state=x, action=action, next_state=nxt, phase=phase))
    return episodes


def property_lookup(root):
    cubes = read(root / 'data/cubes.json')['cubes']
    return {int(c['cube_id']): np.array([c['mass_kg'], c['combined_friction']], dtype=np.float32) for c in cubes}


def fit_normalizer(episodes, properties):
    if any(e['cube_id'] >= 200 or e['episode_id'] >= 8 for e in episodes):
        raise ValueError('Normalizer received nontraining data')
    x = np.concatenate([e['state'] for e in episodes]).astype(np.float64)
    action = np.concatenate([e['action'] for e in episodes]).astype(np.float64)
    context = np.stack([properties[c] for c in range(200)]).astype(np.float64)
    result = {}
    for name, values in [('state', x), ('action', action), ('context', context)]:
        result[name + '_mean'] = values.mean(axis=0).tolist()
        result[name + '_std'] = np.maximum(values.std(axis=0), .001).tolist()
    result['fit_cubes'] = list(range(200)); result['fit_episodes'] = list(range(8))
    result['state_rows'] = len(x)
    lengths = np.array([len(e['state']) for e in episodes])
    nxt = np.concatenate([e['next_state'] for e in episodes]).astype(np.float64)
    phase = np.concatenate([e['phase'] for e in episodes])
    moving = (np.linalg.norm(x[:,9:12], axis=1) > .001) | (np.linalg.norm(x[:,12:15], axis=1) > .05)
    scale = equal_episode_cube_mean(((nxt[:,9:11]-x[:,9:11])**2).mean(axis=1),
        np.repeat(np.arange(len(episodes)), lengths), np.array([e['cube_id'] for e in episodes]),
        (phase == 'coast') & moving)
    if scale['mean'] is None or scale['mean'] <= 0:
        raise ValueError('Undefined training horizontal-delta normalization energy')
    result['horizontal_velocity_delta_mse_scale'] = scale['mean']
    result['horizontal_velocity_delta_scale_coverage'] = scale
    return result


class DynamicsMLP(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in KINDS: raise ValueError(kind)
        self.kind = kind
        dims = [18 + (2 if kind == 'oracle' else 0), 256, 256, 256, 15]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2: layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, state, action, context=None):
        values = [state, action]
        if self.kind == 'oracle':
            if context is None or context.shape[-1] != 2:
                raise ValueError('Oracle requires exactly mass and combined-friction context')
            values.append(context)
        return self.net(torch.cat(values, dim=-1))


def tensor_normalizer(normalizer, device):
    result = {k: torch.tensor(v, dtype=torch.float32, device=device)
              for k, v in normalizer.items() if k.endswith('_mean') or k.endswith('_std')}
    result['horizontal_velocity_delta_mse_scale'] = float(normalizer['horizontal_velocity_delta_mse_scale'])
    return result


def pack_training(episodes, normalizer, properties, device):
    lengths = np.array([len(e['state']) for e in episodes], dtype=np.int64)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    x = np.concatenate([e['state'] for e in episodes])
    action = np.concatenate([e['action'] for e in episodes])
    nxt = np.concatenate([e['next_state'] for e in episodes])
    context = np.concatenate([np.tile(properties[e['cube_id']], (len(e['state']), 1)) for e in episodes])
    phase = np.concatenate([e['phase'] for e in episodes])
    moving = (np.linalg.norm(x[:,9:12], axis=1) > .001) | (np.linalg.norm(x[:,12:15], axis=1) > .05)
    state_mean = np.array(normalizer['state_mean'], dtype=np.float32)
    state_std = np.array(normalizer['state_std'], dtype=np.float32)
    a_mean = np.array(normalizer['action_mean'], dtype=np.float32)
    a_std = np.array(normalizer['action_std'], dtype=np.float32)
    c_mean = np.array(normalizer['context_mean'], dtype=np.float32)
    c_std = np.array(normalizer['context_std'], dtype=np.float32)
    return dict(
        x=torch.tensor((x-state_mean)/state_std, device=device),
        action=torch.tensor((action-a_mean)/a_std, device=device),
        target=torch.tensor((nxt-x)/state_std, device=device),
        context=torch.tensor((context-c_mean)/c_std, device=device),
        starts=starts, lengths=lengths,
        row_episode=np.repeat(np.arange(len(episodes)), lengths),
        cube_ids=np.array([e['cube_id'] for e in episodes]),
        moving_coast=(phase == 'coast') & moving,
        horizontal_state_std=torch.tensor(state_std[9:11], device=device),
        horizontal_true_delta=torch.tensor(nxt[:,9:11].astype(np.float64)-x[:,9:11].astype(np.float64), device=device),
        horizontal_velocity_delta_mse_scale=normalizer['horizontal_velocity_delta_mse_scale'],
    )


def equal_episode_cube_mean(values, row_episode, cube_ids, mask=None):
    values = np.asarray(values, dtype=np.float64)
    mask = np.ones(len(values), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    episode_results = []
    for ep, cube in enumerate(cube_ids):
        chosen = values[mask & (row_episode == ep)]
        if not len(chosen): continue
        episode_results.append((int(cube), None if not np.isfinite(chosen).all() else float(chosen.mean()), len(chosen)))
    cube_results = []
    for cube in sorted(set(int(c) for c in cube_ids)):
        entries = [row[1] for row in episode_results if row[0] == cube]
        if entries:
            cube_results.append(dict(cube_id=cube, mean=None if any(v is None for v in entries) else float(np.mean(entries)), episodes=len(entries)))
    undefined = any(c['mean'] is None for c in cube_results)
    mean = None if undefined or not cube_results else float(np.mean([c['mean'] for c in cube_results]))
    return dict(mean=mean, eligible_starts=int(mask.sum()), nonfinite_starts=int(np.sum(mask & ~np.isfinite(values))),
                episodes=len(episode_results), cubes=len(cube_results), per_cube=cube_results,
                missing_cube_ids=sorted(set(map(int, cube_ids)) - {c['cube_id'] for c in cube_results}))


@torch.no_grad()
def validation_score(model, pack, batch_size=8192):
    model.eval(); errors=[]; secondary=[]
    for start in range(0, len(pack['x']), batch_size):
        s=slice(start, start+batch_size)
        prediction=model(pack['x'][s], pack['action'][s], pack['context'][s] if model.kind=='oracle' else None)
        residual=prediction-pack['target'][s]
        errors.append(((prediction[:,9:11].double()*pack['horizontal_state_std'].double()-pack['horizontal_true_delta'][s]).square().mean(-1)
                       / pack['horizontal_velocity_delta_mse_scale']).cpu().numpy())
        secondary.append(residual.square().mean(-1).cpu().numpy())
    values=np.concatenate(errors)
    return dict(primary=equal_episode_cube_mean(values, pack['row_episode'], pack['cube_ids'], pack['moving_coast'])['mean'],
                secondary_all_step=equal_episode_cube_mean(np.concatenate(secondary), pack['row_episode'], pack['cube_ids'])['mean'])


def train_input_manifest(root):
    result={}
    for c in range(200):
        for e in range(10):
            path=root/'data/episodes'/filename(c,e)
            result[str(path.relative_to(root))]=digest(path)
    for name in ['data/cubes.json','data/plans.json','data/split.json','data/protocol.json','data/seal.json',
                 'results/model_protocol.json','results/test_data_seal.json','results/metrics.json','results/full_audit.json',
                 'results/physics_floor.json','PHYSICS_FLOOR_REPORT.md','physics_floor.py']:
        result[name]=digest(root/name)
    result['models_gate.py']=digest(Path(__file__))
    result['REVISED_PROTOCOL.md']=digest(root/'REVISED_PROTOCOL.md')
    return result


def check_manifest(root, manifest):
    for name, expected in manifest.items():
        if digest(root/name)!=expected:raise RuntimeError(f'Model input/source changed: {name}')


def evaluation_data_manifest(root):
    """Integrity-only hashes: no held-out state arrays are decoded here."""
    metrics_path=root/'results/metrics.json'
    rows=read(metrics_path)
    index={(int(r['cube_id']),int(r['episode_id'])):r for r in rows}
    if len(index)!=len(rows):raise ValueError('Duplicate collection identity in metrics')
    output={'results/metrics.json':digest(metrics_path),'results/full_audit.json':digest(root/'results/full_audit.json')}
    for cube in TEST_CUBES:
        for episode in range(10):
            row=index[(cube,episode)]
            for kind in ['episodes','raw']:
                path=root/'data'/kind/filename(cube,episode)
                observed=digest(path)
                if observed!=row[f'{kind}_sha256']:
                    raise RuntimeError(f'Collection hash mismatch for held-out {kind}: {cube}/{episode}')
                output[str(path.relative_to(root))]=observed
    return output


def freeze_test_data(root):
    expected=evaluation_data_manifest(root)
    path=root/'results/test_data_seal.json'
    if path.exists():
        if read(path)!=expected:raise RuntimeError('Held-out data changed after its integrity seal')
    else:write_new(path,expected)
    return expected


def extension_decision(logs, endpoint):
    decisions = {}
    for kind in KINDS:
        before = min(r['validation_nmse'] for r in logs[kind] if r['update'] <= endpoint-2500)
        recent = min(r['validation_nmse'] for r in logs[kind] if r['update'] > endpoint-2500)
        improvement = (before-recent)/before if before > 0 else 0.
        decisions[kind] = dict(previous_best=before, final2500_best=recent,
                               relative_improvement=improvement, still_improving=bool(recent <= before*.999 and before > 0))
    return dict(endpoint=endpoint, extend=any(v['still_improving'] for v in decisions.values()), families=decisions)


def train_pair(root, seed, train_pack, val_pack, normalizer, manifest, device):
    destinations={kind:root/'checkpoints'/f'{kind}_seed_{seed}.pt' for kind in KINDS}
    present={kind:p.exists() for kind,p in destinations.items()}
    if all(present.values()):
        saved={kind:torch.load(p,map_location='cpu',weights_only=False) for kind,p in destinations.items()}
        for old in saved.values():
            if old['manifest'] != manifest or old['protocol'] != MODEL_PROTOCOL:
                raise RuntimeError('Existing checkpoint source/data mismatch')
        if saved['plain']['minibatch_sha256'] != saved['oracle']['minibatch_sha256']:
            raise RuntimeError('Saved pair minibatches differ')
        print(json.dumps(dict(stage='train',status='pair_already_complete',seed=seed)),flush=True)
        return
    if any(present.values()):
        raise RuntimeError('Partial frozen checkpoint pair exists; do not overwrite or silently retrain')
    models={};optimizers={};logs={kind:[] for kind in KINDS};best={kind:float('inf') for kind in KINDS}
    best_state={};best_update={};training_seconds={kind:0. for kind in KINDS}
    for kind in KINDS:
        torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed)
        models[kind]=DynamicsMLP(kind).to(device)
        optimizers[kind]=torch.optim.Adam(models[kind].parameters(),lr=MODEL_PROTOCOL['learning_rate'])
    rng=np.random.default_rng(seed+1000000);sampling_hash=hashlib.sha256()
    decisions=[];started=time.perf_counter();completed=0
    for update in range(1,MODEL_PROTOCOL['maximum_updates']+1):
        episode_index=rng.integers(len(train_pack['lengths']),size=MODEL_PROTOCOL['batch_size'])
        row=train_pack['starts'][episode_index]+(rng.random(MODEL_PROTOCOL['batch_size'])*train_pack['lengths'][episode_index]).astype(np.int64)
        sampling_hash.update(row.tobytes());idx=torch.as_tensor(row,device=device)
        # Equivalent to the original pre-step cosine scheduler through update10000,
        # followed by a constant floor learning rate during validation-driven extensions.
        previous_step=min(update-1,MODEL_PROTOCOL['updates'])
        lr=MODEL_PROTOCOL['final_learning_rate']+(MODEL_PROTOCOL['learning_rate']-MODEL_PROTOCOL['final_learning_rate'])*.5*(1+math.cos(math.pi*previous_step/MODEL_PROTOCOL['updates']))
        for kind in KINDS:
            step_started=time.perf_counter();model=models[kind];optimizer=optimizers[kind]
            for group in optimizer.param_groups:group['lr']=lr
            model.train();optimizer.zero_grad(set_to_none=True)
            output=model(train_pack['x'][idx],train_pack['action'][idx],train_pack['context'][idx] if kind=='oracle' else None)
            loss=(output-train_pack['target'][idx]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss {kind}/{seed}/{update}')
            loss.backward();optimizer.step()
            if update%MODEL_PROTOCOL['validation_interval']==0:
                scores=validation_score(model,val_pack);score=scores['primary']
                if score is None or not math.isfinite(score):raise FloatingPointError('Nonfinite validation error')
                if score<best[kind]:
                    best[kind]=score;best_update[kind]=update
                    best_state[kind]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                record=dict(update=update,training_batch_nmse=float(loss.detach()),validation_nmse=score,
                            validation_horizontal_velocity_delta_nmse=score,
                            validation_all_state_all_step_nmse=scores['secondary_all_step'],
                            best_validation_nmse=best[kind],learning_rate=lr,elapsed_pair_seconds=time.perf_counter()-started)
                logs[kind].append(record)
                print(json.dumps({'stage':'train','kind':kind,'seed':seed,**record}),flush=True)
            training_seconds[kind]+=time.perf_counter()-step_started
        completed=update
        if update%MODEL_PROTOCOL['extension_block_updates']==0:
            decision=extension_decision(logs,update);decisions.append(decision)
            print(json.dumps(dict(stage='extension_decision',seed=seed,**decision)),flush=True)
            if not decision['extend']:break
    capped=completed==MODEL_PROTOCOL['maximum_updates'] and decisions[-1]['extend']
    pair_record=dict(seed=seed,completed_updates=completed,extension_decisions=decisions,
                     convergence_not_established_at_cap=bool(capped),minibatch_sha256=sampling_hash.hexdigest(),
                     elapsed_seconds=time.perf_counter()-started)
    for kind in KINDS:
        checkpoint=dict(kind=kind,seed=seed,model_state=best_state[kind],selected_update=best_update[kind],
                        selected_validation_nmse=best[kind],normalizer=normalizer,protocol=MODEL_PROTOCOL,manifest=manifest,
                        minibatch_sha256=sampling_hash.hexdigest(),parameters=sum(p.numel() for p in models[kind].parameters()),
                        completed_updates=completed,device=str(device),elapsed_seconds=training_seconds[kind],log=logs[kind],
                        extension_decisions=decisions,convergence_not_established_at_cap=bool(capped))
        destinations[kind].parent.mkdir(exist_ok=True)
        with destinations[kind].open('xb') as f:torch.save(checkpoint,f)
        write_new(root/'results/training_logs'/f'{kind}_seed_{seed}.json',
                  {k:v for k,v in checkpoint.items() if k not in ['model_state','normalizer','manifest']})
    write_new(root/'results/training_logs'/f'pair_seed_{seed}.json',pair_record)

def train_all(root=ROOT, device_name='cpu'):
    root=Path(root);seal_protocol(root);require_collection_ready(root)
    if (root/'results/final_evaluation.json').exists():raise RuntimeError('Training is forbidden after test evaluation begins')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device(device_name)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    freeze_test_data(root)
    properties=property_lookup(root)
    train_eps=load_episodes(root,range(200),range(8))
    val_eps=load_episodes(root,range(200),[8,9])
    normalizer=fit_normalizer(train_eps,properties)
    floor=read(root/'results/physics_floor.json')
    if not np.isclose(normalizer['horizontal_velocity_delta_mse_scale'],
                      floor['normalization']['scalar_horizontal_delta_mse'],rtol=1e-10,atol=1e-15):
        raise RuntimeError('Independent physics/model training delta normalization disagrees')
    normpath=root/'results/normalizer.json'
    if normpath.exists():
        if read(normpath)!=normalizer:raise RuntimeError('Changed training-only normalizer')
    else:write_new(normpath,normalizer)
    manifest=train_input_manifest(root)
    train_pack=pack_training(train_eps,normalizer,properties,device)
    val_pack=pack_training(val_eps,normalizer,properties,device)
    del train_eps,val_eps
    for seed in SEEDS:
        train_pair(root,seed,train_pack,val_pack,normalizer,manifest,device)
    checkpoints={f'{kind}_seed_{seed}.pt':digest(root/'checkpoints'/f'{kind}_seed_{seed}.pt') for seed in SEEDS for kind in KINDS}
    for seed in SEEDS:
        a=torch.load(root/'checkpoints'/f'plain_seed_{seed}.pt',map_location='cpu',weights_only=False)
        b=torch.load(root/'checkpoints'/f'oracle_seed_{seed}.pt',map_location='cpu',weights_only=False)
        if a['minibatch_sha256']!=b['minibatch_sha256']:raise RuntimeError('Paired family minibatches differ')
    check_manifest(root,manifest)
    seal=dict(checkpoints=checkpoints,manifest=manifest,protocol_sha256=digest(root/'results/model_protocol.json'))
    path=root/'results/checkpoint_seal.json'
    if path.exists():
        if read(path)!=seal:raise RuntimeError('Changed sealed checkpoints')
    else:write_new(path,seal)
    print(json.dumps({'stage':'train','status':'all_ten_checkpoints_sealed'}),flush=True)


def make_test_pack(root):
    """Open test motion only inside the logged, frozen evaluation invocation."""
    ledger=read(root/'results/final_evaluation.json')
    if ledger.get('status') not in ('begun','complete'):
        raise RuntimeError('No valid final-evaluation ledger')
    episodes=load_episodes(root,TEST_CUBES,range(10),allow_test=True)
    full=[];release=[];stop_xy=[];force=[];full_lengths=[]
    for episode in episodes:
        path=root/'data/raw'/filename(episode['cube_id'],episode['episode_id'])
        with np.load(path,allow_pickle=False) as raw:
            raw_state=np.asarray(raw['state'],dtype=np.float64)
            raw_action=np.asarray(raw['action'],dtype=np.float64)
            r=int(raw['release_index']);dt=float(raw['physics_dt_s'])
            if dt!=.001 or r%5 or raw_state.shape!=(len(raw_action)+1,15):
                raise ValueError(f'Unexpected raw timing/schema:{path}')
            configured_coast=float(read(root/'data/protocol.json')['coast_s'])
            if not np.isclose((len(raw_action)-r)*dt,configured_coast) or configured_coast!=MODEL_PROTOCOL['raw_coast_seconds']:
                raise ValueError('Stopping rollout horizon disagrees with the frozen preplanned coast duration')
            states=raw_state[::5].astype(np.float32)
            if not np.array_equal(states[:len(episode['state'])],episode['state']):
                raise ValueError('Model and raw states disagree')
            if not np.array_equal(raw_action[::5][:len(episode['action'])].astype(np.float32),episode['action']):
                raise ValueError('Saved force commands disagree with raw appliedforces')
            resting=(np.linalg.norm(raw_state[:,9:12],axis=1)<.001)&(np.linalg.norm(raw_state[:,12:15],axis=1)<.05)
            windows=np.convolve(resting[r+1:].astype(int),np.ones(100,dtype=int),mode='valid')
            candidates=np.flatnonzero(windows==100)
            if not len(candidates):raise ValueError('Testcube didnotreachconfirmedrest')
            stop=r+1+int(candidates[0])
            if stop==r+1 and resting[r]:stop=r
            full.append(states);release.append(r//5);stop_xy.append(raw_state[stop,:2])
            force.append(raw_action[0].astype(np.float32));full_lengths.append(len(states)-1)
    lengths=np.array([len(e['state']) for e in episodes],dtype=np.int64)
    row_episode=np.repeat(np.arange(len(episodes)),lengths)
    row_step=np.concatenate([np.arange(n) for n in lengths])
    state=np.concatenate([e['state'] for e in episodes])
    action=np.concatenate([e['action'] for e in episodes])
    next_state=np.concatenate([e['next_state'] for e in episodes])
    phase=np.concatenate([e['phase'] for e in episodes])
    moving=(np.linalg.norm(state[:,9:12],axis=1)>.001)|(np.linalg.norm(state[:,12:15],axis=1)>.05)
    release=np.asarray(release,dtype=np.int64);full_lengths=np.asarray(full_lengths,dtype=np.int64)
    remaining=full_lengths[row_episode]-row_step
    push_remaining=release[row_episode]-row_step
    mask=dict(push=phase=='push',moving_coast=(phase=='coast')&moving,all_steps=np.ones(len(state),dtype=bool))
    rollout_mask=dict(push=mask['push']&(push_remaining>=10),
                      moving_coast=mask['moving_coast']&(remaining>=10),
                      all_steps=remaining>=10)
    max_steps=max(full_lengths)
    padded=np.empty((len(full),max_steps+1,15),dtype=np.float32)
    for e,states in enumerate(full):
        padded[e,:len(states)]=states;padded[e,len(states):]=states[-1]
    return dict(state=state,action=action,next_state=next_state,row_episode=row_episode,row_step=row_step,
                cube_ids=np.array([e['cube_id'] for e in episodes]),episode_ids=np.array([e['episode_id'] for e in episodes]),
                full_state=padded,full_lengths=full_lengths,release_steps=release,
                stop_xy=np.asarray(stop_xy,dtype=np.float64),force=np.asarray(force),
                remaining=remaining,push_remaining=push_remaining,masks=mask,rollout_masks=rollout_mask,
                episode_count=len(episodes),transition_count=len(state))


@torch.no_grad()
def rollout_batch(model, state, action_force, context, normalizer, push_remaining, remaining,
                  targets10, next_state, target_stop_xy):
    """Vectorized recursion with action-defined terminal times, never true stop times."""
    prediction=state.clone()
    total10=torch.zeros(len(state),dtype=torch.float64,device=state.device)
    final10=torch.zeros_like(total10)
    one_step=torch.zeros_like(total10)
    horizontal_delta=torch.zeros_like(total10)
    max_horizon=int(remaining.max().item())
    for k in range(max_horizon):
        command=torch.where((push_remaining>k)[:,None],action_force,torch.zeros_like(action_force))
        normalized_state=(prediction-normalizer['state_mean'])/normalizer['state_std']
        normalized_action=(command-normalizer['action_mean'])/normalizer['action_std']
        delta=model(normalized_state,normalized_action,context if model.kind=='oracle' else None)
        proposed=prediction+delta*normalizer['state_std']
        active=remaining>k
        prediction=torch.where(active[:,None],proposed,prediction)
        if k==0:
            one_step=((prediction-next_state)/normalizer['state_std']).double().square().mean(-1)
            predicted_delta=delta[:,9:11].double()*normalizer['state_std'][9:11].double()
            true_delta=next_state[:,9:11].double()-state[:,9:11].double()
            horizontal_delta=(predicted_delta-true_delta).square().mean(-1)/normalizer['horizontal_velocity_delta_mse_scale']
        if k<10:
            error=((prediction-targets10[:,k])/normalizer['state_std']).double().square().mean(-1)
            total10+=error/10
            if k==9:final10=error
    stop_error=torch.linalg.vector_norm(prediction[:,:2].double()-target_stop_xy,dim=-1)
    return {PRIMARY_METRIC:horizontal_delta.cpu().numpy(), 'one_step_nmse':one_step.cpu().numpy(),'rollout10_nmse':total10.cpu().numpy(),
            'rollout10_final_nmse':final10.cpu().numpy(),'stopping_position_error_m':stop_error.cpu().numpy()}


@torch.no_grad()
def evaluate_checkpoint(root, checkpoint_path, pack, properties, device, batch_size=4096):
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if checkpoint['protocol']!=MODEL_PROTOCOL:raise RuntimeError('Checkpoint protocol differs')
    model=DynamicsMLP(checkpoint['kind']).to(device)
    model.load_state_dict(checkpoint['model_state']);model.eval()
    norm=tensor_normalizer(checkpoint['normalizer'],device)
    context=np.stack([properties[int(c)] for c in pack['cube_ids']])
    episode_context=torch.tensor(context,device=device)
    episode_context=(episode_context-norm['context_mean'])/norm['context_std']
    results={k:[] for k in [PRIMARY_METRIC,'one_step_nmse','rollout10_nmse','rollout10_final_nmse','stopping_position_error_m']}
    started=time.perf_counter()
    for start in range(0,len(pack['state']),batch_size):
        end=min(len(pack['state']),start+batch_size)
        ep=pack['row_episode'][start:end];step=pack['row_step'][start:end]
        horizons=step[:,None]+np.arange(1,11)[None,:]
        horizons=np.minimum(horizons,pack['full_lengths'][ep,None])
        target=pack['full_state'][ep[:,None],horizons]
        values=rollout_batch(model,
            torch.tensor(pack['state'][start:end],device=device),
            torch.tensor(pack['force'][ep],device=device),episode_context[torch.tensor(ep,device=device)],norm,
            torch.tensor(pack['push_remaining'][start:end],device=device),
            torch.tensor(pack['remaining'][start:end],device=device),
            torch.tensor(target,device=device),torch.tensor(pack['next_state'][start:end],device=device),
            torch.tensor(pack['stop_xy'][ep],dtype=torch.float64,device=device))
        for name,value in values.items():results[name].append(value)
        print(json.dumps({'stage':'evaluate','kind':checkpoint['kind'],'seed':checkpoint['seed'],
                          'completed_starts':end,'total_starts':len(pack['state']),
                          'elapsed_seconds':time.perf_counter()-started}),flush=True)
    results={k:np.concatenate(v) for k,v in results.items()}
    groups={}
    for group in GROUPS:
        groups[group]={}
        for name,values in results.items():
            mask=pack['rollout_masks'][group] if name.startswith('rollout10') else pack['masks'][group]
            groups[group][name]=equal_episode_cube_mean(values,pack['row_episode'],pack['cube_ids'],mask)
            if name==PRIMARY_METRIC and groups[group][name]['mean'] is not None:
                raw_mse=groups[group][name]['mean']*checkpoint['normalizer']['horizontal_velocity_delta_mse_scale']
                groups[group][name].update(raw_mse_m2_s2=raw_mse,rmse_m_s=math.sqrt(raw_mse))
    return dict(kind=checkpoint['kind'],seed=checkpoint['seed'],checkpoint_sha256=digest(checkpoint_path),
                checkpoint_selected_update=checkpoint['selected_update'],training_budget_updates=checkpoint['completed_updates'],
                parameters=checkpoint['parameters'],groups=groups,elapsed_seconds=time.perf_counter()-started,
                test_cubes=30,test_episodes=pack['episode_count'],test_transitions=pack['transition_count'],
                no_observed_intermediate_states=True,no_rollout_clipping=True,
                horizon='Each startrollsto preplannedrelease+1.5s, not observedtrimmedlength/stop',
                scored_state_coordinates=15)


def summarize_seed_results(records):
    if {(r['kind'],r['seed']) for r in records}!={(k,s) for k in KINDS for s in SEEDS}:
        raise ValueError('Expected every family and allfivepairedseeds exactly once')
    if len(records)!=10:raise ValueError('Duplicate finalseedresult')
    ordered={(r['kind'],r['seed']):r for r in records}
    summary={}
    metrics=[PRIMARY_METRIC,'one_step_nmse','rollout10_nmse','rollout10_final_nmse','stopping_position_error_m']
    for group in GROUPS:
        summary[group]={}
        for metric in metrics:
            item={};arrays={}
            for kind in KINDS:
                values=[ordered[(kind,seed)]['groups'][group][metric]['mean'] for seed in SEEDS]
                arrays[kind]=np.asarray(values,dtype=float) if all(v is not None for v in values) else None
                item[kind]=dict(seed_values=values,mean=None if arrays[kind] is None else float(arrays[kind].mean()),
                                sample_sd=None if arrays[kind] is None else float(arrays[kind].std(ddof=1)))
            p,o=arrays['plain'],arrays['oracle']
            if p is None or o is None:
                item.update(mean_gap=None,relative_mean_improvement=None,paired_gap_sample_sd=None,
                            paired_seed_improvements=None,gap_exceeds_sum_of_seed_sds=False,
                            gap_exceeds_paired_gap_sd=False,defined=False)
            else:
                gap=p-o;pmean=p.mean();omean=o.mean()
                item.update(mean_gap=float(gap.mean()),relative_mean_improvement=float((pmean-omean)/pmean) if pmean>0 else None,
                            paired_gap_sample_sd=float(gap.std(ddof=1)),
                            paired_seed_improvements=[float((a-b)/a) if a>0 else None for a,b in zip(p,o)],
                            gap_exceeds_sum_of_seed_sds=bool(omean+o.std(ddof=1)<pmean-p.std(ddof=1)),
                            gap_exceeds_paired_gap_sd=bool(gap.mean()>gap.std(ddof=1)),defined=True)
            summary[group][metric]=item
    primary=summary['moving_coast'][PRIMARY_METRIC]
    coverage=all(r['groups']['moving_coast'][PRIMARY_METRIC]['cubes']==30 for r in records)
    improvement=primary['relative_mean_improvement']
    at_least_twenty=bool(primary['defined'] and primary['plain']['mean']>0 and
                        primary['oracle']['mean']<=.8*primary['plain']['mean'])
    passed=bool(coverage and at_least_twenty and primary['gap_exceeds_sum_of_seed_sds'])
    return dict(primary_gate_passed=passed,primary_group='moving_coast',primary_metric=PRIMARY_METRIC,
                at_least_twenty_percent=at_least_twenty,
                beyond_seed_spread=primary['gap_exceeds_sum_of_seed_sds'],all30testcubes_covered=coverage,
                comparisons=summary,seeds=SEEDS,test_policy='One frozen finalcomparison; stop ifgatefails; no beliefmodel',
                undefined_metrics_reported_without_dropping_seeds=True)


def evaluate_all(root=ROOT,device_name='cpu',batch_size=4096):
    root=Path(root);seal_protocol(root);require_collection_ready(root)
    torch.set_num_threads(2);device=torch.device(device_name)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    seal=read(root/'results/checkpoint_seal.json');check_manifest(root,seal['manifest'])
    freeze_test_data(root)
    for name,expected in seal['checkpoints'].items():
        if digest(root/'checkpoints'/name)!=expected:raise RuntimeError('Frozen checkpoint changed')
    ledgerpath=root/'results/final_evaluation.json'
    context=dict(checkpoint_seal_sha256=digest(root/'results/checkpoint_seal.json'),
                 data_seal_sha256=digest(root/'data/seal.json'),
                 test_data_seal_sha256=digest(root/'results/test_data_seal.json'),source_sha256=digest(Path(__file__)))
    if ledgerpath.exists():
        ledger=read(ledgerpath)
        if ledger['context']!=context:raise RuntimeError('Cannot reopen test with changed checkpoint/data/source')
        if ledger['status']=='complete':
            print(json.dumps({'stage':'evaluate','status':'already_complete','summary':str(root/'results/gate_summary.json')}),flush=True)
            return read(root/'results/gate_summary.json')
    else:
        write_new(ledgerpath,dict(status='begun',context=context,started_unix_seconds=time.time(),
                                 note='Alltencheckpoints sealed before any testmotionread. Resume onlycompletes identical invocation.'))
    pack=make_test_pack(root);properties=property_lookup(root);records=[]
    floorpath=root/'results/test_physics_floor.json'
    if floorpath.exists():
        physics=read(floorpath)
    else:
        truth={int(c['cube_id']):c['combined_friction'] for c in read(root/'data/cubes.json')['cubes']}
        episodes=load_episodes(root,TEST_CUBES,range(10),allow_test=True)
        for ep in episodes:
            ep['state']=ep['state'].astype(np.float64);ep['next_state']=ep['next_state'].astype(np.float64)
            ep['combined_friction']=truth[ep['cube_id']]
        scale=read(root/'results/normalizer.json')['horizontal_velocity_delta_mse_scale']
        physics=compute_physics_metrics(episodes,scale)
        physics.update(test_cubes=TEST_CUBES,normalization_scale=scale,
                       source_sha256=digest(root/'physics_floor.py'),test_data_seal_sha256=context['test_data_seal_sha256'])
        write_new(floorpath,physics)
    for seed in SEEDS:
        for kind in KINDS:
            output=root/'results/test_metrics'/f'{kind}_seed_{seed}.json'
            checkpoint=root/'checkpoints'/f'{kind}_seed_{seed}.pt'
            if output.exists():
                result=read(output)
                if result['checkpoint_sha256']!=digest(checkpoint):raise RuntimeError('Saved seedscore has wrongcheckpoint')
            else:
                result=evaluate_checkpoint(root,checkpoint,pack,properties,device,batch_size)
                write_new(output,result)
            records.append(result)
    summary=summarize_seed_results(records)
    oracle=summary['comparisons']['moving_coast'][PRIMARY_METRIC]['oracle']
    floor_comparisons={}
    for name in ['requested_unclipped','stop_clipped_diagnostic']:
        value=physics['metrics'][name]['nmse']
        oracle_mean=oracle['mean']
        floor_comparisons[name]=dict(physics_nmse=value,oracle_mean_nmse=oracle_mean,
            oracle_excess_nmse=None if oracle_mean is None else oracle_mean-value,
            oracle_to_physics_error_ratio=None if oracle_mean is None or value==0 else oracle_mean/value,
            physics_rmse_m_s=physics['metrics'][name]['rmse_m_s'],
            oracle_rmse_from_mean_mse_m_s=None if oracle_mean is None else math.sqrt(oracle_mean*physics['normalization_scale']))
    summary['physics_comparison']=floor_comparisons
    summary['test_physics_floor_sha256']=digest(floorpath)
    summary.update(checkpoint_seal_sha256=context['checkpoint_seal_sha256'],model_protocol_sha256=digest(root/'results/model_protocol.json'),
                   final_evaluation_device=str(device),test_cubes=30,test_episodes=pack['episode_count'],test_transitions=pack['transition_count'])
    summarypath=root/'results/gate_summary.json'
    if summarypath.exists():
        if read(summarypath)!=summary:raise RuntimeError('Existing finalsummary differs')
    else:write_new(summarypath,summary)
    ledger=read(ledgerpath);ledger.update(status='complete',completed_unix_seconds=time.time(),summary_sha256=digest(summarypath),
                                       primary_gate_passed=summary['primary_gate_passed'])
    write_atomic(ledgerpath,ledger)
    print(json.dumps({'stage':'evaluate','status':'complete','primary_gate_passed':summary['primary_gate_passed'],
                      'summary':str(summarypath)}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['seal','train','evaluate','run'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--evaluation-batch-size',type=int,default=4096)
    args=parser.parse_args();root=args.root.resolve()
    if args.stage=='seal':
        print(seal_protocol(root));return
    if args.stage in ('train','run'):train_all(root,args.device)
    if args.stage in ('evaluate','run'):evaluate_all(root,args.device,args.evaluation_batch_size)


if __name__=='__main__':main()
