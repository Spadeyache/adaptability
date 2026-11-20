"""Frozen Step 2 models, training, and validation-only latent adaptation.

This module never performs predictive test evaluation. Test motion is readable
only after the separately implemented final evaluation ledger has begun.
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
Z_DIMS = [2, 4, 8, 16]
VARIANTS = ['plain', 'oracle', 'history'] + [f'z{d}' for d in Z_DIMS]
TRAIN_CUBES = list(range(180))
LATENT_VALIDATION_CUBES = list(range(180, 200))
TEST_CUBES = list(range(260, 290))
GROUPS = ['push', 'moving_coast', 'all_steps']
PRIMARY_METRIC = 'horizontal_velocity_delta_nmse'
WRITTEN_PROTOCOL_SHA256 = '20fe5c2527ecbd061917d5f16fbf341397f1182b9cd2ec2369f2812794014ef1'
MODEL_PROTOCOL = {
    'step': 2, 'written_protocol_sha256': WRITTEN_PROTOCOL_SHA256,
    'seeds': SEEDS, 'variants': VARIANTS, 'state_dim': 15, 'action_dim': 3,
    'hidden_layers': [256, 256, 256], 'activation': 'ReLU',
    'history_steps': 10, 'history_token_dim': 33, 'gru_hidden_dim': 64,
    'history': 'Past completed (state15,action3,delta15) only. Fixed ten-token window, initial-state/zero-action/zero-delta padding, no mask; reset each episode.',
    'latent_dimensions': Z_DIMS, 'embedding_rows': 180, 'embedding_initialization': 'zero',
    'embedding_l2': 1e-4, 'adaptation_prior_l2': 1e-4,
    'train_cubes': TRAIN_CUBES, 'train_episodes': list(range(8)),
    'checkpoint_validation_episodes': [8, 9],
    'latent_validation_cubes': LATENT_VALIDATION_CUBES,
    'support_episodes': list(range(5)), 'query_episodes': list(range(5, 10)),
    'test_cubes': TEST_CUBES, 'retired_test_cubes': list(range(200, 260)),
    'batch_size': 512, 'optimizer': 'Adam', 'minimum_updates': 10000,
    'maximum_updates': 100000, 'extension_block_updates': 10000,
    'validation_interval': 500, 'learning_rate': .001, 'final_learning_rate': .0001,
    'schedule': 'Cosine first10000 updates, then constant0.0001.',
    'extension': 'At each10000 endpoint any of seven variants improving previous best by >=0.1% in final2500 extends every variant another10000; max100000, report cap.',
    'training_objective': 'Full15 normalized state-delta MSE, plus1e-4 mean(all embedding entries squared) for two-part variants.',
    'sampling': 'Uniform fitting episode then uniform retained step; identical row stream for seven variants in each seed.',
    'checkpoint_selection': 'Minimum push horizontal-delta NMSE on weight-fitting cubes episodes8,9.',
    'normalization': 'Only IDs0..179 episodes0..7; state/action populationSD floored0.001, Oracle context from180 fitting labels. Push delta energy equal cube/episode/step; zero delta scores1.',
    'adaptation_learning_rates': [.01, .03, .1], 'adaptation_update_counts': [50, 100, 200, 400],
    'adaptation_objective': 'Equal support-episode push horizontal-delta NMSE per cube plus1e-4 mean((z-training_mean_z)^2); sum independent cube objectives, deterministic full support Adam.',
    'selection': 'One global dim/lr/updates minimizes five-seed mean20validationcube querypushNMSE after5supportshoves. Exact ties smallerdim,fewerupdates,smallerlr.',
    'mass_readout': 'Affine OLS pseudoinverse fitted to180 training cubes support-refitted z; no latent-validation labels, no decoder selection.',
    'forbidden_features': ['IDs','phase','time','step','length','padding_mask','stop_time','termination','future observations'],
    'primary_group': 'push', 'primary_metric': PRIMARY_METRIC,
    'raw_coast_seconds': 1.5, 'model_dt_seconds': .005,
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


def filename(cube_id, episode_id):
    return f'cube_{cube_id:03d}_episode_{episode_id:02d}.npz'


def seal_protocol(root=ROOT):
    root = Path(root)
    if digest(root/'PROTOCOL.md') != WRITTEN_PROTOCOL_SHA256:
        raise RuntimeError('Written protocol changed before model execution')
    path = root/'results/model_protocol.json'
    if path.exists():
        if read(path) != MODEL_PROTOCOL: raise RuntimeError('Frozen model protocol differs')
    else: write_new(path, MODEL_PROTOCOL)
    return path


def check_manifest(root, manifest):
    root = Path(root)
    for name, expected in manifest.items():
        if digest(root/name) != expected: raise RuntimeError(f'Changed sealed input: {name}')


def require_collection_ready(root):
    root = Path(root)
    if (root/'results/STOP.json').exists():
        raise RuntimeError('STOP artifact exists; no automatic rerun is permitted')
    check_manifest(root/'data', read(root/'data/seal.json'))
    if read(root/'results/full_audit.json')['status'] != 'passed':
        raise RuntimeError('Collection audit did not pass')
    expected = {filename(c,e) for c in list(range(200))+TEST_CUBES for e in range(10)}
    for kind in ['episodes','raw']:
        if {p.name for p in (root/'data'/kind).glob('*.npz')} != expected:
            raise RuntimeError('Unexpected Step2 dataset identities')


def load_episodes(root, cube_ids, episode_ids, allow_test=False):
    root = Path(root); cube_ids = list(cube_ids); episode_ids = list(episode_ids)
    if any(200 <= c < 260 for c in cube_ids):
        raise RuntimeError('Retired test cubes cannot enter Step2')
    if any(c >= 260 for c in cube_ids):
        if not allow_test: raise RuntimeError('Test motion cannot enter fitting or tuning')
        ledger_path = root/'results/final_evaluation.json'
        if not ledger_path.exists() or read(ledger_path).get('status') not in ['begun','complete']:
            raise RuntimeError('Fresh test motion requires the frozen final evaluation ledger')
    if any(c not in list(range(200))+TEST_CUBES for c in cube_ids):
        raise RuntimeError('Unknown cube identity')
    episodes = []
    for c in cube_ids:
        for e in episode_ids:
            path = root/'data/episodes'/filename(c,e)
            with np.load(path, allow_pickle=False) as a:
                state, action, nxt = [np.asarray(a[k],dtype=np.float32) for k in ['state','action','next_state']]
                phase = np.asarray(a['phase']).copy()
                if state.shape != nxt.shape or state.shape[1] != 15 or action.shape != (len(state),3):
                    raise RuntimeError(f'Unexpected episode shape: {path}')
                if not all(np.isfinite(x).all() for x in [state,action,nxt]):
                    raise FloatingPointError('Nonfinite observations')
                if not (np.all(a['cube_id']==c) and np.all(a['episode_id']==e)):
                    raise RuntimeError('Episode identity mismatch')
            episodes.append(dict(cube_id=c,episode_id=e,state=state,action=action,next_state=nxt,phase=phase))
    return episodes


def property_lookup(root):
    return {int(c['cube_id']):np.array([c['mass_kg'],c['combined_friction']],dtype=np.float32)
            for c in read(Path(root)/'data/cubes.json')['cubes']}


def equal_episode_cube_mean(values, row_episode, cube_ids, mask=None):
    values = np.asarray(values,dtype=np.float64)
    mask = np.ones(len(values),dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    row_episode = np.asarray(row_episode)
    by_cube = {}; starts = 0; episodes = 0
    for ep,cube in enumerate(cube_ids):
        chosen = values[mask & (row_episode==ep)]
        if not len(chosen): continue
        v = None if not np.isfinite(chosen).all() else float(chosen.mean())
        by_cube.setdefault(int(cube),[]).append(v); starts += len(chosen); episodes += 1
    per_cube = [dict(cube_id=c,mean=None if any(v is None for v in vals) else float(np.mean(vals)),episodes=len(vals))
                for c,vals in sorted(by_cube.items())]
    mean = None if not per_cube or any(c['mean'] is None for c in per_cube) else float(np.mean([c['mean'] for c in per_cube]))
    return dict(mean=mean,eligible_starts=int(starts),nonfinite_starts=int(np.sum(mask & ~np.isfinite(values))),
                episodes=episodes,cubes=len(per_cube),per_cube=per_cube,
                missing_cube_ids=sorted(set(map(int,cube_ids))-set(by_cube)))


def fit_normalizer(episodes, properties):
    if any(e['cube_id'] not in TRAIN_CUBES or e['episode_id'] not in range(8) for e in episodes):
        raise RuntimeError('Normalizer received non-fitting data')
    x = np.concatenate([e['state'] for e in episodes]).astype(np.float64)
    action = np.concatenate([e['action'] for e in episodes]).astype(np.float64)
    nxt = np.concatenate([e['next_state'] for e in episodes]).astype(np.float64)
    context = np.stack([properties[c] for c in TRAIN_CUBES]).astype(np.float64)
    result = {}
    for name,values in [('state',x),('action',action),('context',context)]:
        result[name+'_mean'] = values.mean(0).tolist()
        result[name+'_std'] = np.maximum(values.std(0),.001).tolist()
    lengths = [len(e['state']) for e in episodes]
    row_episode = np.repeat(np.arange(len(episodes)),lengths)
    cube_ids = np.array([e['cube_id'] for e in episodes])
    push = np.concatenate([e['phase']=='push' for e in episodes])
    scale = equal_episode_cube_mean(np.mean((nxt[:,9:11]-x[:,9:11])**2,axis=1),row_episode,cube_ids,push)
    if scale['mean'] is None or scale['mean'] <= 0: raise RuntimeError('Invalid fitting push energy')
    result.update(fit_cubes=TRAIN_CUBES,fit_episodes=list(range(8)),state_rows=len(x),
                  horizontal_velocity_delta_mse_scale=scale['mean'],push_scale_coverage=scale)
    return result


def tensor_normalizer(normalizer, device):
    output = {k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in normalizer.items()
              if k.endswith('_mean') or k.endswith('_std')}
    output['horizontal_velocity_delta_mse_scale'] = float(normalizer['horizontal_velocity_delta_mse_scale'])
    return output


def build_history(episodes, normalizer):
    sm = np.asarray(normalizer['state_mean'],dtype=np.float32); ss = np.asarray(normalizer['state_std'],dtype=np.float32)
    am = np.asarray(normalizer['action_mean'],dtype=np.float32); ast = np.asarray(normalizer['action_std'],dtype=np.float32)
    windows = []
    for ep in episodes:
        x = np.asarray(ep['state'],dtype=np.float32); a = np.asarray(ep['action'],dtype=np.float32)
        nxt = np.asarray(ep['next_state'],dtype=np.float32)
        tokens = np.concatenate([(x-sm)/ss,(a-am)/ast,(nxt-x)/ss],axis=1)
        neutral = np.concatenate([(x[0]-sm)/ss,-am/ast,np.zeros(15,dtype=np.float32)])
        padded = np.concatenate([np.tile(neutral,(10,1)),tokens],axis=0)
        # Window at row t ends at token t-1; the current target is never included.
        windows.append(np.stack([padded[t:t+10] for t in range(len(x))]))
    return np.concatenate(windows).astype(np.float32)


def pack_episodes(episodes, normalizer, properties, device, include_history=True):
    lengths = np.array([len(e['state']) for e in episodes],dtype=np.int64)
    starts = np.r_[0,np.cumsum(lengths)[:-1]]
    x = np.concatenate([e['state'] for e in episodes]).astype(np.float32)
    action = np.concatenate([e['action'] for e in episodes]).astype(np.float32)
    nxt = np.concatenate([e['next_state'] for e in episodes]).astype(np.float32)
    phase = np.concatenate([e['phase'] for e in episodes])
    cube_ids = np.array([e['cube_id'] for e in episodes],dtype=np.int64)
    row_episode = np.repeat(np.arange(len(episodes)),lengths)
    row_cube = cube_ids[row_episode]
    if properties is None:
        context = np.zeros((len(x),2),dtype=np.float32)
    else:
        context = np.stack([properties[int(c)] for c in row_cube])
    sm = np.asarray(normalizer['state_mean'],dtype=np.float32); ss = np.asarray(normalizer['state_std'],dtype=np.float32)
    am = np.asarray(normalizer['action_mean'],dtype=np.float32); ast = np.asarray(normalizer['action_std'],dtype=np.float32)
    cm = np.asarray(normalizer['context_mean'],dtype=np.float32); cs = np.asarray(normalizer['context_std'],dtype=np.float32)
    moving = (np.linalg.norm(x[:,9:12],axis=1)>.001) | (np.linalg.norm(x[:,12:15],axis=1)>.05)
    result = dict(x=torch.tensor((x-sm)/ss,device=device),action=torch.tensor((action-am)/ast,device=device),
        target=torch.tensor((nxt-x)/ss,device=device),context=torch.tensor((context-cm)/cs,device=device),
        horizontal_true_delta=torch.tensor(nxt[:,9:11].astype(np.float64)-x[:,9:11].astype(np.float64),device=device),
        horizontal_state_std=torch.tensor(ss[9:11],device=device),
        horizontal_velocity_delta_mse_scale=float(normalizer['horizontal_velocity_delta_mse_scale']),
        cube_index=torch.tensor(row_cube,device=device),row_cube=row_cube,
        state=x,next_state=nxt,raw_action=action,phase=phase,lengths=lengths,starts=starts,
        row_episode=row_episode,row_step=np.concatenate([np.arange(n) for n in lengths]),
        cube_ids=cube_ids,episode_ids=np.array([e['episode_id'] for e in episodes]),
        masks=dict(push=phase=='push',moving_coast=(phase=='coast')&moving,all_steps=np.ones(len(x),dtype=bool)))
    if include_history: result['history'] = torch.tensor(build_history(episodes,normalizer),device=device)
    return result


class DynamicsModel(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in VARIANTS: raise ValueError(variant)
        self.variant = variant
        self.z_dim = int(variant[1:]) if variant.startswith('z') else 0
        self.embedding = nn.Embedding(180,self.z_dim) if self.z_dim else None
        if self.embedding is not None: nn.init.zeros_(self.embedding.weight)
        self.gru = nn.GRU(33,64,batch_first=True) if variant=='history' else None
        extra = 2 if variant=='oracle' else 64 if variant=='history' else self.z_dim
        dims = [18+extra,256,256,256,15]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i],dims[i+1]))
            if i < len(dims)-2: layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, state, action, context=None, history=None, z=None):
        values = [state,action]
        if self.variant=='oracle':
            if context is None or context.shape != (len(state),2): raise ValueError('Oracle requires two context columns')
            values.append(context)
        elif self.variant=='history':
            if history is None or history.shape != (len(state),10,33): raise ValueError('History requires ten past tokens')
            values.append(self.gru(history)[1][-1])
        elif self.z_dim:
            if z is None or z.shape != (len(state),self.z_dim): raise ValueError('Two-part requires an explicit latent vector')
            values.append(z)
        return self.net(torch.cat(values,dim=-1))


def predict_pack(model, pack, rows, external_z=None):
    z = None
    if model.z_dim:
        if external_z is None:
            indices = pack['cube_index'][rows]
            if torch.any((indices<0)|(indices>=180)): raise RuntimeError('Held-out cube cannot index training embedding')
            z = model.embedding(indices)
        else: z = external_z
    return model(pack['x'][rows],pack['action'][rows],
                 context=pack['context'][rows] if model.variant=='oracle' else None,
                 history=pack['history'][rows] if model.variant=='history' else None,z=z)


@torch.no_grad()
def validation_score(model, pack, batch_size=4096):
    model.eval(); primary=[]; secondary=[]
    for start in range(0,len(pack['x']),batch_size):
        sl = slice(start,start+batch_size)
        output = predict_pack(model,pack,sl)
        if not torch.isfinite(output).all(): raise FloatingPointError('Nonfinite validation output')
        raw_delta = output[:,9:11].double()*pack['horizontal_state_std'].double()
        primary.append(((raw_delta-pack['horizontal_true_delta'][sl]).square().mean(-1)/pack['horizontal_velocity_delta_mse_scale']).cpu().numpy())
        secondary.append((output-pack['target'][sl]).square().mean(-1).cpu().numpy())
    values = np.concatenate(primary)
    return dict(primary=equal_episode_cube_mean(values,pack['row_episode'],pack['cube_ids'],pack['masks']['push'])['mean'],
                secondary_all_step=equal_episode_cube_mean(np.concatenate(secondary),pack['row_episode'],pack['cube_ids'])['mean'])


def extension_decision(logs, endpoint):
    decisions = {}
    for variant in VARIANTS:
        before = min(r['validation_nmse'] for r in logs[variant] if r['update']<=endpoint-2500)
        recent = min(r['validation_nmse'] for r in logs[variant] if endpoint-2500<r['update']<=endpoint)
        improvement = (before-recent)/before if before>0 else 0.
        decisions[variant] = dict(previous_best=before,final2500_best=recent,relative_improvement=improvement,
                                  still_improving=bool(before>0 and recent<=before*.999))
    return dict(endpoint=endpoint,extend=any(v['still_improving'] for v in decisions.values()),variants=decisions)


def evaluation_data_manifest(root):
    root = Path(root); rows=read(root/'results/metrics.json')
    index = {(int(r['cube_id']),int(r['episode_id'])):r for r in rows}
    if len(index)!=len(rows): raise RuntimeError('Duplicate collection records')
    output = {name:digest(root/name) for name in ['results/metrics.json','results/full_audit.json']}
    for c in TEST_CUBES:
        for e in range(10):
            row = index[(c,e)]
            for kind in ['episodes','raw']:
                path=root/'data'/kind/filename(c,e); observed=digest(path)
                if observed!=row[kind+'_sha256']: raise RuntimeError('Fresh trace hash mismatch')
                output[str(path.relative_to(root))]=observed
    return output


def freeze_test_data(root):
    root=Path(root); value=evaluation_data_manifest(root); path=root/'results/test_data_seal.json'
    if path.exists():
        if read(path)!=value: raise RuntimeError('Fresh test bytes changed')
    else: write_new(path,value)
    return value


def training_manifest(root):
    root=Path(root); result={}
    for c in range(200):
        for e in range(10):
            path=root/'data/episodes'/filename(c,e); result[str(path.relative_to(root))]=digest(path)
    for name in ['models.py','evaluation.py','collection.py','PROTOCOL.md',
                 'data/cubes.json','data/plans.json','data/split.json','data/protocol.json','data/seal.json',
                 'results/model_protocol.json','results/normalizer.json','results/test_data_seal.json',
                 'results/full_audit.json','results/metrics.json','results/source_seal.json']:
        result[name]=digest(root/name)
    return result


def configure_torch(device_name):
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device=torch.device(device_name)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    return device


def train_group(root,seed,train_pack,val_pack,normalizer,manifest,device):
    destinations={v:root/'checkpoints'/f'{v}_seed_{seed}.pt' for v in VARIANTS}
    present={v:p.exists() for v,p in destinations.items()}
    if all(present.values()):
        saved=[torch.load(p,map_location='cpu',weights_only=False) for p in destinations.values()]
        if any(c['manifest']!=manifest or c['protocol']!=MODEL_PROTOCOL for c in saved): raise RuntimeError('Existing checkpoint mismatch')
        if len({c['minibatch_sha256'] for c in saved})!=1: raise RuntimeError('Existing batch streams differ')
        print(json.dumps(dict(stage='train',seed=seed,status='group_already_complete')),flush=True); return
    if any(present.values()): raise RuntimeError('Partial completed group exists; stop instead of retraining')
    models={}; optimizers={}; logs={v:[] for v in VARIANTS}; best={v:float('inf') for v in VARIANTS}
    best_state={}; best_update={}; timings={v:0. for v in VARIANTS}
    for v in VARIANTS:
        torch.manual_seed(seed)
        if device.type=='cuda': torch.cuda.manual_seed_all(seed)
        models[v]=DynamicsModel(v).to(device)
        optimizers[v]=torch.optim.Adam(models[v].parameters(),lr=.001)
    rng=np.random.default_rng(seed+1000000); sampling_hash=hashlib.sha256(); started=time.perf_counter()
    decisions=[]; completed=0
    for update in range(1,MODEL_PROTOCOL['maximum_updates']+1):
        episode_index=rng.integers(len(train_pack['lengths']),size=MODEL_PROTOCOL['batch_size'])
        row=train_pack['starts'][episode_index]+(rng.random(MODEL_PROTOCOL['batch_size'])*train_pack['lengths'][episode_index]).astype(np.int64)
        sampling_hash.update(row.tobytes()); idx=torch.as_tensor(row,device=device)
        lr=.0001+(.001-.0001)*.5*(1+math.cos(math.pi*min(update-1,10000)/10000))
        for v in VARIANTS:
            began=time.perf_counter(); model=models[v]; optimizer=optimizers[v]
            for param_group in optimizer.param_groups: param_group['lr']=lr
            model.train(); optimizer.zero_grad(set_to_none=True)
            output=predict_pack(model,train_pack,idx)
            mse=(output-train_pack['target'][idx]).square().mean()
            penalty=1e-4*model.embedding.weight.square().mean() if model.z_dim else torch.zeros((),device=device)
            loss=mse+penalty
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite training loss {v}/{seed}/{update}')
            loss.backward(); optimizer.step()
            if update%500==0:
                scores=validation_score(model,val_pack); score=scores['primary']
                if score is None or not math.isfinite(score): raise FloatingPointError('Nonfinite validation metric')
                if score<best[v]:
                    best[v]=score; best_update[v]=update
                    best_state[v]={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
                record=dict(update=update,training_batch_nmse=float(mse.detach()),embedding_penalty=float(penalty.detach()),
                    validation_nmse=score,validation_push_horizontal_velocity_delta_nmse=score,
                    validation_all_state_all_step_nmse=scores['secondary_all_step'],best_validation_nmse=best[v],
                    learning_rate=lr,elapsed_group_seconds=time.perf_counter()-started)
                logs[v].append(record)
                print(json.dumps(dict(stage='train',variant=v,seed=seed,**record)),flush=True)
            timings[v]+=time.perf_counter()-began
        completed=update
        if update%10000==0:
            decision=extension_decision(logs,update); decisions.append(decision)
            print(json.dumps(dict(stage='extension_decision',seed=seed,**decision)),flush=True)
            if not decision['extend']: break
    capped=bool(completed==100000 and decisions[-1]['extend'])
    group=dict(seed=seed,completed_updates=completed,extension_decisions=decisions,
               convergence_not_established_at_cap=capped,minibatch_sha256=sampling_hash.hexdigest(),
               elapsed_seconds=time.perf_counter()-started)
    for v in VARIANTS:
        checkpoint=dict(variant=v,seed=seed,z_dim=models[v].z_dim,model_state=best_state[v],selected_update=best_update[v],
            selected_validation_nmse=best[v],normalizer=normalizer,protocol=MODEL_PROTOCOL,manifest=manifest,
            minibatch_sha256=sampling_hash.hexdigest(),parameters=sum(p.numel() for p in models[v].parameters()),
            completed_updates=completed,device=str(device),elapsed_seconds=timings[v],log=logs[v],
            extension_decisions=decisions,convergence_not_established_at_cap=capped)
        destinations[v].parent.mkdir(exist_ok=True)
        with destinations[v].open('xb') as f: torch.save(checkpoint,f)
        write_new(root/'results/training_logs'/f'{v}_seed_{seed}.json',
                  {k:x for k,x in checkpoint.items() if k not in ['model_state','normalizer','manifest']})
    write_new(root/'results/training_logs'/f'group_seed_{seed}.json',group)


def train_all(root=ROOT,device_name='cpu'):
    root=Path(root); seal_protocol(root); require_collection_ready(root)
    if (root/'results/final_evaluation.json').exists(): raise RuntimeError('No training after final evaluation begins')
    device=configure_torch(device_name); freeze_test_data(root)
    properties=property_lookup(root)
    train_eps=load_episodes(root,TRAIN_CUBES,range(8)); val_eps=load_episodes(root,TRAIN_CUBES,[8,9])
    normalizer=fit_normalizer(train_eps,properties); path=root/'results/normalizer.json'
    if path.exists():
        if read(path)!=normalizer: raise RuntimeError('Fitting-only normalizer changed')
    else: write_new(path,normalizer)
    manifest=training_manifest(root)
    train_pack=pack_episodes(train_eps,normalizer,properties,device)
    val_pack=pack_episodes(val_eps,normalizer,properties,device)
    del train_eps,val_eps
    for seed in SEEDS: train_group(root,seed,train_pack,val_pack,normalizer,manifest,device)
    checkpoints={f'{v}_seed_{s}.pt':digest(root/'checkpoints'/f'{v}_seed_{s}.pt') for s in SEEDS for v in VARIANTS}
    check_manifest(root,manifest)
    seal=dict(checkpoints=checkpoints,manifest=manifest,protocol_sha256=digest(root/'results/model_protocol.json'))
    path=root/'results/checkpoint_seal.json'
    if path.exists():
        if read(path)!=seal: raise RuntimeError('Frozen checkpoint seal changed')
    else: write_new(path,seal)
    print(json.dumps(dict(stage='train',status='all35_checkpoints_sealed')),flush=True)


def load_checkpoint(path,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint['protocol']!=MODEL_PROTOCOL: raise RuntimeError('Checkpoint protocol mismatch')
    model=DynamicsModel(checkpoint['variant']).to(device)
    model.load_state_dict(checkpoint['model_state']); model.eval()
    return model,checkpoint


def model_state_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fit_zs(model,support_episodes,normalizer,initial_mean,lr,updates,device,save_steps=None):
    if not model.z_dim: raise ValueError('Latent adaptation requires a two-part model')
    if any(e['episode_id'] not in range(5) for e in support_episodes):
        raise RuntimeError('Query episode cannot enter latent fitting')
    cube_ids=sorted({int(e['cube_id']) for e in support_episodes})
    if not cube_ids or updates<=0 or lr<=0: raise ValueError('Empty or invalid latent fitting request')
    save_steps=sorted(set([updates] if save_steps is None else list(save_steps)+[updates]))
    if save_steps[0]<=0 or save_steps[-1]>updates: raise ValueError('Invalid latent snapshot steps')
    pack=pack_episodes(support_episodes,normalizer,None,device,include_history=False)
    chosen=np.flatnonzero(pack['masks']['push'])
    if not len(chosen): raise RuntimeError('No push support transitions')
    index={c:i for i,c in enumerate(cube_ids)}
    row_cube=np.array([index[int(c)] for c in pack['row_cube'][chosen]],dtype=np.int64)
    per_cube_eps={c:sum(e['cube_id']==c for e in support_episodes) for c in cube_ids}
    per_episode_push=np.bincount(pack['row_episode'][chosen],minlength=len(support_episodes))
    if np.any(per_episode_push==0): raise RuntimeError('Empty support shove')
    selected_episode=pack['row_episode'][chosen]
    weights=1./(per_episode_push[selected_episode]*np.array([per_cube_eps[int(c)] for c in pack['row_cube'][chosen]]))
    idx=torch.tensor(chosen,device=device); routing=torch.tensor(row_cube,device=device)
    weight=torch.tensor(weights,dtype=torch.float64,device=device)
    x=pack['x'][idx]; action=pack['action'][idx]; target=pack['horizontal_true_delta'][idx]
    state_std=pack['horizontal_state_std'].double(); scale=pack['horizontal_velocity_delta_mse_scale']
    del pack
    mean=torch.as_tensor(initial_mean,dtype=torch.float32,device=device).reshape(-1)
    if mean.shape!=(model.z_dim,): raise ValueError('Wrong mean latent shape')
    before=model_state_digest(model); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    z=nn.Parameter(mean.repeat(len(cube_ids),1).clone()); optimizer=torch.optim.Adam([z],lr=float(lr))
    snapshots={}; log=[]; started=time.perf_counter()
    for update in range(1,updates+1):
        optimizer.zero_grad(set_to_none=True)
        output=model(x,action,z=z[routing])
        if not torch.isfinite(output).all(): raise FloatingPointError('Nonfinite latent fitting prediction')
        residual=output[:,9:11].double()*state_std-target
        per_row=residual.square().mean(-1)/scale
        data_loss=(per_row*weight).sum()
        prior=1e-4*(z-mean).square().mean(1).sum()
        loss=data_loss+prior
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite latent fitting loss')
        loss.backward(); optimizer.step()
        if not torch.isfinite(z).all(): raise FloatingPointError('Nonfinite fitted latent')
        if update in save_steps:
            snapshots[update]=z.detach().cpu().numpy().copy()
            log.append(dict(update=update,mean_cube_support_nmse=float(data_loss.detach())/len(cube_ids),
                            mean_cube_prior=float(prior.detach())/len(cube_ids),elapsed_seconds=time.perf_counter()-started))
    if model_state_digest(model)!=before: raise RuntimeError('Network weights changed during latent fitting')
    return dict(cube_ids=cube_ids,snapshots=snapshots,log=log,weights_unchanged=True,
                model_state_sha256=before,elapsed_seconds=time.perf_counter()-started)


@torch.no_grad()
def score_latents(model,query_episodes,normalizer,cube_ids,z_values,device):
    pack=pack_episodes(query_episodes,normalizer,None,device,include_history=False)
    lookup={c:i for i,c in enumerate(cube_ids)}
    routing=torch.tensor([lookup[int(c)] for c in pack['row_cube']],device=device)
    z=torch.as_tensor(z_values,dtype=torch.float32,device=device)
    errors=[]
    for start in range(0,len(pack['x']),4096):
        sl=slice(start,start+4096)
        output=model(pack['x'][sl],pack['action'][sl],z=z[routing[sl]])
        if not torch.isfinite(output).all(): raise FloatingPointError('Nonfinite latent query output')
        pred=output[:,9:11].double()*pack['horizontal_state_std'].double()
        errors.append(((pred-pack['horizontal_true_delta'][sl]).square().mean(-1)/pack['horizontal_velocity_delta_mse_scale']).cpu().numpy())
    return equal_episode_cube_mean(np.concatenate(errors),pack['row_episode'],pack['cube_ids'],pack['masks']['push'])


def select_latent_configuration(records):
    expected={(d,lr,u,s) for d in Z_DIMS for lr in [.01,.03,.1] for u in [50,100,200,400] for s in SEEDS}
    found={(r['z_dim'],r['learning_rate'],r['updates'],r['seed']) for r in records}
    if len(records)!=len(expected) or found!=expected: raise RuntimeError('Incomplete or duplicate latent validation grid')
    if any(sorted(c['cube_id'] for c in r['query']['per_cube'])!=LATENT_VALIDATION_CUBES for r in records):
        raise RuntimeError('Latent selection received cubes outside the reserved validation set')
    candidates=[]
    for d in Z_DIMS:
        for lr in [.01,.03,.1]:
            for u in [50,100,200,400]:
                selected=[r for r in records if (r['z_dim'],r['learning_rate'],r['updates'])==(d,lr,u)]
                if any(r['query']['cubes']!=20 or r['query']['episodes']!=100 or r['query']['mean'] is None for r in selected):
                    raise RuntimeError('Incomplete latent-validation coverage')
                vals=[r['query']['mean'] for r in selected]
                candidates.append(dict(z_dim=d,variant=f'z{d}',learning_rate=lr,updates=u,
                    validation_mean_nmse=float(np.mean(vals)),validation_seed_sample_sd=float(np.std(vals,ddof=1)),seed_values=vals))
    best=min(candidates,key=lambda c:(c['validation_mean_nmse'],c['z_dim'],c['updates'],c['learning_rate']))
    return best,candidates


def fit_mass_readout(cube_ids,z_values,mass_lookup):
    if list(cube_ids)!=TRAIN_CUBES:
        raise RuntimeError('Mass readout must fit exactly the180 weight-fitting cubes')
    labels=np.array([mass_lookup[c] for c in cube_ids],dtype=np.float64)
    design=np.column_stack([np.ones(len(z_values)),np.asarray(z_values,dtype=np.float64)])
    coefficients=np.linalg.pinv(design)@labels
    residual=design@coefficients-labels
    return dict(coefficients=coefficients.tolist(),training_mean_mass_kg=float(labels.mean()),
        fit_cube_ids=list(cube_ids),fit_count=len(cube_ids),
        train_r2=float(1-np.sum(residual**2)/np.sum((labels-labels.mean())**2)),
        train_mae_kg=float(np.abs(residual).mean()),train_rmse_kg=float(np.sqrt(np.mean(residual**2))))


def tune_and_fit_readouts(root=ROOT,device_name='cpu'):
    root=Path(root); seal_protocol(root); require_collection_ready(root)
    if (root/'results/final_evaluation.json').exists(): raise RuntimeError('No tuning after final evaluation begins')
    device=configure_torch(device_name); seal=read(root/'results/checkpoint_seal.json')
    check_manifest(root,seal['manifest']); check_manifest(root/'checkpoints',seal['checkpoints']); freeze_test_data(root)
    if (root/'results/adaptation_seal.json').exists():
        old=read(root/'results/adaptation_seal.json'); check_manifest(root,old['files'])
        print(json.dumps(dict(stage='tune',status='already_complete')),flush=True); return
    normalizer=read(root/'results/normalizer.json')
    support=load_episodes(root,LATENT_VALIDATION_CUBES,range(5))
    query=load_episodes(root,LATENT_VALIDATION_CUBES,range(5,10))
    records=[]; started=time.perf_counter()
    for seed in SEEDS:
        for dim in Z_DIMS:
            model,checkpoint=load_checkpoint(root/'checkpoints'/f'z{dim}_seed_{seed}.pt',device)
            mean=model.embedding.weight.detach().mean(0)
            for lr in [.01,.03,.1]:
                path=root/'results/latent_validation_runs'/f'z{dim}_seed_{seed}_lr{lr:g}.json'
                if path.exists():
                    saved=read(path)
                    if saved['checkpoint_sha256']!=digest(root/'checkpoints'/f'z{dim}_seed_{seed}.pt'):
                        raise RuntimeError('Saved latent tuning checkpoint changed')
                    records.extend(saved['records']); continue
                fitted=fit_zs(model,support,normalizer,mean,lr,400,device,save_steps=[50,100,200,400])
                local=[]
                for updates in [50,100,200,400]:
                    score=score_latents(model,query,normalizer,fitted['cube_ids'],fitted['snapshots'][updates],device)
                    local.append(dict(z_dim=dim,learning_rate=lr,updates=updates,seed=seed,query=score))
                write_new(path,dict(checkpoint_sha256=digest(root/'checkpoints'/f'z{dim}_seed_{seed}.pt'),
                    records=local,support_log=fitted['log'],weights_unchanged=fitted['weights_unchanged']))
                records.extend(local)
                print(json.dumps(dict(stage='latent_validation',seed=seed,z_dim=dim,learning_rate=lr,
                    scores={str(r['updates']):r['query']['mean'] for r in local})),flush=True)
            del model
    chosen,candidates=select_latent_configuration(records)
    tuning=dict(records=records,candidates=candidates,selection=chosen,
                validation_cubes=LATENT_VALIDATION_CUBES,support_episodes=list(range(5)),query_episodes=list(range(5,10)),
                elapsed_seconds=time.perf_counter()-started,checkpoint_seal_sha256=digest(root/'results/checkpoint_seal.json'))
    for name,value in [('latent_tuning.json',tuning),('selected_config.json',chosen)]:
        path=root/'results'/name
        if path.exists():
            # A partial interrupted tuning stage can preserve already selected values.
            previous=read(path)
            if name=='selected_config.json' and previous!=value: raise RuntimeError('Selected configuration changed')
            if name=='latent_tuning.json' and previous['records']!=records: raise RuntimeError('Tuning records changed')
        else: write_new(path,value)
    training_support=load_episodes(root,TRAIN_CUBES,range(5))
    mass={int(c['cube_id']):float(c['mass_kg']) for c in read(root/'data/cubes.json')['cubes'] if c['cube_id'] in TRAIN_CUBES}
    readouts={}
    for seed in SEEDS:
        path=root/'results'/f'training_fitted_z_seed_{seed}.npz'
        model,checkpoint=load_checkpoint(root/'checkpoints'/f'{chosen["variant"]}_seed_{seed}.pt',device)
        if path.exists():
            with np.load(path,allow_pickle=False) as a: cube_ids=a['cube_ids'].tolist(); values=a['z'].copy()
            if cube_ids!=TRAIN_CUBES: raise RuntimeError('Saved decoder fitting cube IDs changed')
        else:
            mean=model.embedding.weight.detach().mean(0)
            fitted=fit_zs(model,training_support,normalizer,mean,chosen['learning_rate'],chosen['updates'],device)
            cube_ids=fitted['cube_ids']; values=fitted['snapshots'][chosen['updates']]
            np.savez_compressed(path,cube_ids=np.asarray(cube_ids),z=values)
        readouts[str(seed)]=dict(**fit_mass_readout(cube_ids,values,mass),fit_z_sha256=digest(path),
            checkpoint_sha256=digest(root/'checkpoints'/f'{chosen["variant"]}_seed_{seed}.pt'))
        print(json.dumps(dict(stage='mass_readout_fit',seed=seed,train_r2=readouts[str(seed)]['train_r2'])),flush=True)
    readout_result=dict(by_seed=readouts,method='Affine OLS pseudoinverse on support-refitted training z',
                        selection_sha256=digest(root/'results/selected_config.json'))
    path=root/'results/mass_readouts.json'
    if path.exists():
        if read(path)!=readout_result: raise RuntimeError('Mass readout changed')
    else: write_new(path,readout_result)
    files={name:digest(root/name) for name in ['results/selected_config.json','results/latent_tuning.json',
        'results/mass_readouts.json','results/normalizer.json','results/checkpoint_seal.json','models.py','evaluation.py','PROTOCOL.md']}
    for p in sorted((root/'results/latent_validation_runs').glob('*.json')):
        files[str(p.relative_to(root))]=digest(p)
    for seed in SEEDS:
        name=f'results/training_fitted_z_seed_{seed}.npz'; files[name]=digest(root/name)
    check_manifest(root,seal['manifest']); check_manifest(root/'checkpoints',seal['checkpoints'])
    write_new(root/'results/adaptation_seal.json',dict(files=files,checkpoint_seal_sha256=digest(root/'results/checkpoint_seal.json'),
              completed_unix_seconds=time.time(),selected_config=chosen))
    print(json.dumps(dict(stage='tune',status='adaptation_and_mass_readouts_frozen',selected=chosen)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['seal','train','tune','run'])
    parser.add_argument('--root',type=Path,default=ROOT); parser.add_argument('--device',default='cpu')
    args=parser.parse_args(); root=args.root.resolve()
    try:
        if args.stage=='seal': print(seal_protocol(root)); return
        if args.stage in ['train','run']: train_all(root,args.device)
        if args.stage in ['tune','run']: tune_and_fit_readouts(root,args.device)
    except Exception as error:
        stop=root/'results/STOP.json'
        if not stop.exists():
            write_new(stop,dict(stage=args.stage,error_type=type(error).__name__,message=str(error),
                                recorded_unix_seconds=time.time(),automatic_rerun_forbidden=True))
        raise


if __name__=='__main__': main()
