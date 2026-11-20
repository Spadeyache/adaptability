"""Frozen Plain/Oracle gate for direct-force shoves.

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

ROOT = Path(__file__).resolve().parent
SEEDS = [101, 211, 307, 401, 503]
KINDS = ['plain', 'oracle']
GROUPS = ['push', 'moving_coast', 'all_steps']
MODEL_PROTOCOL = {
    'version': 3, 'raw_coast_seconds': 1.5, 'state_dim': 15, 'action_dim': 3,
    'hidden_layers': [256, 256, 256], 'activation': 'ReLU',
    'seeds': SEEDS, 'families': KINDS,
    'updates': 10000, 'batch_size': 512, 'optimizer': 'Adam',
    'learning_rate': .001, 'final_learning_rate': .0001,
    'learning_rate_schedule': 'CosineAnnealingLR across all10000 updates',
    'determinism': 'Fixed NumPy and torch seeds; deterministic torch algorithms and CUDA workspace, cudnn benchmarking disabled. Reproduction assumes the same software and hardware.',
    'train_cubes': list(range(200)), 'test_cubes': list(range(200, 230)),
    'train_episodes': list(range(8)), 'validation_episodes': [8, 9],
    'validation_interval': 500,
    'checkpoint_selection': 'Lowest validation all-step one-step NMSE; equal cube and episode weight. Full10000-update budget always run.',
    'normalization': 'State/action means and populationSD of retained training transitions only; eachSD floored at.001 nativeunits. Oracle context mean/SD from200training-cube labels, never test. Target(next_state-state)/stateSD.',
    'training_sampling': 'Uniform training episode, then uniform retained transition. Paired families restart the same seed-specific NumPy minibatch stream.',
    'inputs': {'plain': 'normalized state15 + normalized applied force3',
               'oracle': 'same + normalized true mass + normalized combined friction'},
    'forbidden_inputs': ['cube_id', 'episode_id', 'step', 'phase', 'trimmed_length', 'padding_mask', 'stop_time'],
    'score_groups': GROUPS,
    'moving_coast': 'phasecoast AND(norm(observed linearvelocity)>.001m/s OR norm(observed angularvelocity)>.05rad/s)',
    'one_step_nmse': 'Mean squared normalized next-state error over15statecoordinates; firstmeanwithineligible starts/episode, thenepisodes/cube, thenequalweightcubes.',
    'rollout10_nmse': 'Mean squared normalized stateerror acrossall10 recursivepredictions and15coordinates; no intermediate observedstate fed. Pushwindows entirelypush; movingcoaststarts observedmoving withall10stepscoast; all_steps windows cancrossphase. Rawfixedhorizon supplies targets beyondtrimmedfile.',
    'rollout10_final_nmse': 'Same normalization; final tenth predicted state only, diagnostic.',
    'stopping_position_error_m': 'Fromeveryretainedeligible start, recursivelypredict to preplannedrelease+1.5s. Compare terminalpredictedXY with observed confirmed-restXY. Horizonfromcommandduration only, nevertrimmedlength/truestoptime. Equalepisode/cubeaggregation.',
    'gate': 'Moving-coastone-stepNMSE meanOracle<=.8*meanPlain AND meanOracle+sampleSDOracle<meanPlain-sampleSDPlain acrossfivepairedseeds. No substitutionaftertest.',
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
    if not read(root / 'results/pilot_gate.json').get('passed'):
        raise RuntimeError('Conditional direct-force pilot gate has not passed')
    audit = read(root / 'results/full_audit.json')
    if audit.get('status') != 'passed':
        raise RuntimeError('Full collected-data audit has not passed')
    expected = {filename(c, e) for c in range(230) for e in range(10)}
    actual = {p.name for p in (root / 'data/episodes').glob('*.npz')}
    if actual != expected:
        raise RuntimeError('Full230cube×10episode dataset is incomplete or contains extra files')


def load_episodes(root, cube_ids, episode_ids, allow_test=False):
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
    return {k: torch.tensor(v, dtype=torch.float32, device=device)
            for k, v in normalizer.items() if k.endswith('_mean') or k.endswith('_std')}


def pack_training(episodes, normalizer, properties, device):
    lengths = np.array([len(e['state']) for e in episodes], dtype=np.int64)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    x = np.concatenate([e['state'] for e in episodes])
    action = np.concatenate([e['action'] for e in episodes])
    nxt = np.concatenate([e['next_state'] for e in episodes])
    context = np.concatenate([np.tile(properties[e['cube_id']], (len(e['state']), 1)) for e in episodes])
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
    model.eval(); errors=[]
    for start in range(0, len(pack['x']), batch_size):
        s=slice(start, start+batch_size)
        prediction=model(pack['x'][s], pack['action'][s], pack['context'][s] if model.kind=='oracle' else None)
        errors.append((prediction-pack['target'][s]).square().mean(-1).cpu().numpy())
    values=np.concatenate(errors)
    return equal_episode_cube_mean(values, pack['row_episode'], pack['cube_ids'])['mean']


def train_input_manifest(root):
    result={}
    for c in range(200):
        for e in range(10):
            path=root/'data/episodes'/filename(c,e)
            result[str(path.relative_to(root))]=digest(path)
    for name in ['data/cubes.json','data/plans.json','data/split.json','data/protocol.json','data/seal.json',
                 'results/model_protocol.json','results/test_data_seal.json','results/metrics.json','results/full_audit.json']:
        result[name]=digest(root/name)
    result['models_gate.py']=digest(Path(__file__))
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
    for cube in range(200,230):
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


def train_one(root, kind, seed, train_pack, val_pack, normalizer, manifest, device):
    destination=root/'checkpoints'/f'{kind}_seed_{seed}.pt'
    if destination.exists():
        old=torch.load(destination, map_location='cpu', weights_only=False)
        if old['manifest']!=manifest or old['protocol']!=MODEL_PROTOCOL:raise RuntimeError('Existing checkpoint source/data mismatch')
        print(json.dumps({'stage':'train','status':'already_complete','kind':kind,'seed':seed}),flush=True)
        return
    torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    model=DynamicsMLP(kind).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=MODEL_PROTOCOL['learning_rate'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=MODEL_PROTOCOL['updates'],eta_min=MODEL_PROTOCOL['final_learning_rate'])
    rng=np.random.default_rng(seed+1000000)
    sampling_hash=hashlib.sha256()
    log=[];best=float('inf');best_state=None;best_update=None
    started=time.perf_counter()
    for update in range(1,MODEL_PROTOCOL['updates']+1):
        episode_index=rng.integers(len(train_pack['lengths']),size=MODEL_PROTOCOL['batch_size'])
        row=train_pack['starts'][episode_index]+(rng.random(MODEL_PROTOCOL['batch_size'])*train_pack['lengths'][episode_index]).astype(np.int64)
        sampling_hash.update(row.tobytes())
        idx=torch.as_tensor(row,device=device)
        model.train();optimizer.zero_grad(set_to_none=True)
        output=model(train_pack['x'][idx],train_pack['action'][idx],train_pack['context'][idx] if kind=='oracle' else None)
        loss=(output-train_pack['target'][idx]).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss {kind}/{seed}/{update}')
        loss.backward();optimizer.step();scheduler.step()
        if update%MODEL_PROTOCOL['validation_interval']==0:
            score=validation_score(model,val_pack)
            if score is None or not math.isfinite(score):raise FloatingPointError('Nonfinite validation error')
            if score<best:
                best=score;best_update=update
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            record=dict(update=update,training_batch_nmse=float(loss.detach()),validation_nmse=score,
                        best_validation_nmse=best,learning_rate=scheduler.get_last_lr()[0],elapsed_seconds=time.perf_counter()-started)
            log.append(record)
            print(json.dumps({'stage':'train','kind':kind,'seed':seed,**record}),flush=True)
    checkpoint=dict(kind=kind,seed=seed,model_state=best_state,selected_update=best_update,
                    selected_validation_nmse=best,normalizer=normalizer,protocol=MODEL_PROTOCOL,manifest=manifest,
                    minibatch_sha256=sampling_hash.hexdigest(),parameters=sum(p.numel() for p in model.parameters()),
                    completed_updates=MODEL_PROTOCOL['updates'],device=str(device),elapsed_seconds=time.perf_counter()-started,log=log)
    destination.parent.mkdir(exist_ok=True)
    with destination.open('xb') as f:torch.save(checkpoint,f)
    write_new(root/'results/training_logs'/f'{kind}_seed_{seed}.json',{k:v for k,v in checkpoint.items() if k not in ['model_state','normalizer','manifest']})


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
    normpath=root/'results/normalizer.json'
    if normpath.exists():
        if read(normpath)!=normalizer:raise RuntimeError('Changed training-only normalizer')
    else:write_new(normpath,normalizer)
    manifest=train_input_manifest(root)
    train_pack=pack_training(train_eps,normalizer,properties,device)
    val_pack=pack_training(val_eps,normalizer,properties,device)
    del train_eps,val_eps
    for seed in SEEDS:
        for kind in KINDS:train_one(root,kind,seed,train_pack,val_pack,normalizer,manifest,device)
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
    episodes=load_episodes(root,range(200,230),range(10),allow_test=True)
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
        if k<10:
            error=((prediction-targets10[:,k])/normalizer['state_std']).double().square().mean(-1)
            total10+=error/10
            if k==9:final10=error
    stop_error=torch.linalg.vector_norm(prediction[:,:2].double()-target_stop_xy,dim=-1)
    return {'one_step_nmse':one_step.cpu().numpy(),'rollout10_nmse':total10.cpu().numpy(),
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
    results={k:[] for k in ['one_step_nmse','rollout10_nmse','rollout10_final_nmse','stopping_position_error_m']}
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
    metrics=['one_step_nmse','rollout10_nmse','rollout10_final_nmse','stopping_position_error_m']
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
    primary=summary['moving_coast']['one_step_nmse']
    coverage=all(r['groups']['moving_coast']['one_step_nmse']['cubes']==30 for r in records)
    improvement=primary['relative_mean_improvement']
    at_least_twenty=bool(primary['defined'] and primary['plain']['mean']>0 and
                        primary['oracle']['mean']<=.8*primary['plain']['mean'])
    passed=bool(coverage and at_least_twenty and primary['gap_exceeds_sum_of_seed_sds'])
    return dict(primary_gate_passed=passed,primary_group='moving_coast',primary_metric='one_step_nmse',
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
