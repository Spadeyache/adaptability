"""Train-only scaling, explicit episode roles, and guarded held-out reads."""
from dataclasses import dataclass
import json
import os
import time
import numpy as np
import torch
from clean.protocol import ROOT, PROTOCOL, digest, verify, write_new
from clean.audit import moving_mask


def record_access(group, episodes, purpose, count, candidate=None):
    with (ROOT/'results/data_access.jsonl').open('a') as f:
        f.write(json.dumps(dict(group=group, episodes=list(episodes), purpose=purpose,
            cube_count=count, candidate=candidate, timestamp=time.time(), pid=os.getpid()))+'\n')


def episode_arrays(group, episodes, purpose, final_candidate=None):
    episodes=tuple(episodes)
    split=verify()
    if group not in ('train','test','out_of_range'):
        raise ValueError(group)
    if group!='train':
        ledger=ROOT/'results/final_evaluations.jsonl'
        if final_candidate is None or not ledger.exists():
            raise PermissionError('Held-out motion is final-evaluation only')
        events=[json.loads(line) for line in ledger.read_text().splitlines()]
        begin=[e for e in events if e['candidate']==final_candidate and e['event']=='begin']
        end=[e for e in events if e['candidate']==final_candidate and e['event']=='complete']
        if len(begin)!=1 or end:
            raise PermissionError('No active unique final evaluation for this candidate')
        if purpose not in ('final_fit','final_score'):
            raise PermissionError('Held-out data requested for nonfinal purpose')
        allowed=PROTOCOL['fit_episodes'] if purpose=='final_fit' else PROTOCOL['final_score_episodes']
        if any(e not in allowed for e in episodes):
            raise PermissionError('Held-out episode role mismatch')
    elif purpose=='train' and any(e not in PROTOCOL['train_episodes'] for e in episodes):
        raise PermissionError('Validation episodes cannot train shared weights')
    record_access(group, episodes, purpose, len(split[group]), final_candidate)
    for cube in split[group]:
        for episode in episodes:
            path=ROOT/'data'/group/f'cube_{cube:04d}_episode_{episode:02d}.npz'
            receipt=json.loads(path.with_suffix('.json').read_text())
            if digest(path)!=receipt['sha256']:
                raise ValueError(f'Episode changed: {path}')
            with np.load(path,allow_pickle=False) as f:
                yield cube,episode,{key:f[key].copy() for key in ['state','action','next_state','step']}


def fit_normalizer():
    path=ROOT/'data/normalizer.json'
    if path.exists():
        norm=json.loads(path.read_text())
        if norm['split_sha256']!=digest(ROOT/'data/split.json'):
            raise RuntimeError('Normalizer split mismatch')
        return norm
    xs=[]; acts=[]
    for _,_,arr in episode_arrays('train',range(8),'train'):
        xs.append(arr['state']);acts.append(arr['action'])
    x=np.concatenate(xs).astype(np.float64);a=np.concatenate(acts).astype(np.float64)
    cubes=json.loads((ROOT/'data/cubes.json').read_text())['cubes']
    labels=np.asarray([[c['mass_kg'],c['friction']] for c in cubes if c['split']=='train'])
    norm=dict(state_mean=x.mean(0).tolist(),state_std=np.maximum(x.std(0),.001).tolist(),
        action_mean=a.mean(0).tolist(),action_std=np.maximum(a.std(0),.001).tolist(),
        oracle_property_mean=labels.mean(0).tolist(),oracle_property_std=labels.std(0).tolist(),
        split_sha256=digest(ROOT/'data/split.json'),source='200 training cubes, episodes 0-7 only',
        train_transition_count=len(x))
    write_new(path,norm)
    return norm


def properties_for(cube_ids, normalizer):
    """Explicit Oracle-only adapter; call site must be the Oracle branch."""
    cubes=json.loads((ROOT/'data/cubes.json').read_text())['cubes']
    labels=torch.tensor([[c['mass_kg'],c['friction']] for c in cubes],device=cube_ids.device,dtype=torch.float32)
    mean=torch.tensor(normalizer['oracle_property_mean'],device=cube_ids.device)
    std=torch.tensor(normalizer['oracle_property_std'],device=cube_ids.device)
    return (labels[cube_ids]-mean)/std


@dataclass
class Pack:
    x: torch.Tensor
    a: torch.Tensor
    y: torch.Tensor
    cube_ids: torch.Tensor
    episode_ids: torch.Tensor
    step: torch.Tensor
    remaining: torch.Tensor
    moving: torch.Tensor
    starts: torch.Tensor
    start_rows: torch.Tensor
    group: str='train'
    purpose: str='train'

    def __len__(self):
        return len(self.x)

    @property
    def delta(self):
        return self.y-self.x

    def history(self, indices):
        lengths=(indices-self.start_rows[indices]+1).clamp(max=20)
        offsets=torch.arange(20,device=self.x.device)[None,:]
        valid=offsets<lengths[:,None]
        rows=indices[:,None]-lengths[:,None]+1+offsets
        rows=torch.where(valid,rows,indices[:,None])
        windows=torch.cat((self.x[rows],self.a[rows]),dim=-1)
        return windows.masked_fill(~valid[...,None],0),lengths


def load_pack(group,episodes,normalizer,device='cpu',purpose='train',final_candidate=None):
    xs=[];acts=[];ys=[];ids=[];eps=[];steps=[];remaining=[];moving=[];starts=[]
    for cube,episode,arr in episode_arrays(group,episodes,purpose,final_candidate):
        n=len(arr['state'])
        xs.append(arr['state']);acts.append(arr['action']);ys.append(arr['next_state'])
        ids.append(np.full(n,cube));eps.append(np.full(n,episode));steps.append(arr['step'])
        remaining.append(np.arange(n,0,-1));moving.append(moving_mask(arr['state'],arr['next_state']))
        starts.append(np.r_[True,np.zeros(n-1,dtype=bool)])
    def tensor(arrays,dtype):
        return torch.as_tensor(np.concatenate(arrays),dtype=dtype,device=device)
    x=tensor(xs,torch.float32);a=tensor(acts,torch.float32);y=tensor(ys,torch.float32)
    xm=torch.tensor(normalizer['state_mean'],device=device,dtype=torch.float32)
    xsd=torch.tensor(normalizer['state_std'],device=device,dtype=torch.float32)
    am=torch.tensor(normalizer['action_mean'],device=device,dtype=torch.float32)
    asd=torch.tensor(normalizer['action_std'],device=device,dtype=torch.float32)
    start=tensor(starts,torch.bool)
    rows=torch.arange(len(x),device=device)
    start_rows=torch.cummax(torch.where(start,rows,0),dim=0).values
    return Pack((x-xm)/xsd,(a-am)/asd,(y-xm)/xsd,tensor(ids,torch.long),tensor(eps,torch.long),
        tensor(steps,torch.long),tensor(remaining,torch.long),tensor(moving,torch.bool),start,start_rows,group,purpose)
