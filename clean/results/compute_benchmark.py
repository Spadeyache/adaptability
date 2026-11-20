"""Synthetic hardware timing only; no research data or scientific training."""
import json
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from clean.models import DynamicsModel, make_history_windows
from clean.train import _prediction_loss, resolve_config


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def bench(kind,device):
    torch.manual_seed(1234)
    torch.set_num_threads(2)
    if device == 'cuda':
        torch.cuda.manual_seed_all(1234)
    n=10240
    x=torch.randn(n,48,device=device)
    a=torch.randn(n,9,device=device)
    starts=torch.zeros(n,dtype=torch.bool,device=device)
    starts[::128]=True
    rows=torch.arange(n,device=device)
    start_rows=torch.cummax(torch.where(starts,rows,0),0).values
    def history(indices):
        lengths=(indices-start_rows[indices]+1).clamp(max=20)
        offsets=torch.arange(20,device=device)[None,:]
        valid=offsets<lengths[:,None]
        selected=indices[:,None]-lengths[:,None]+1+offsets
        selected=torch.where(valid,selected,indices[:,None])
        windows=torch.cat((x[selected],a[selected]),-1)
        return windows.masked_fill(~valid[...,None],0),lengths
    pack=SimpleNamespace(x=x,a=a,y=x+torch.randn_like(x)*.05,
        cube_ids=torch.zeros(n,dtype=torch.long,device=device),
        moving=torch.rand(n,device=device)>.5,history=history)
    cfg=resolve_config({})
    model=DynamicsModel(kind,48,9,width=128,depth=2).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    rng=torch.Generator(device='cpu').manual_seed(1234)
    def update():
        indices=torch.randint(n,(256,),generator=rng).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss=_prediction_loss(model,pack,indices,cfg)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite synthetic benchmark loss')
        loss.backward()
        optimizer.step()
    for _ in range(10):update()
    sync(device)
    start=time.perf_counter()
    for _ in range(50):update()
    sync(device)
    elapsed=time.perf_counter()-start
    return dict(kind=kind,device=device,updates=50,warmup_updates=10,
        elapsed_seconds=elapsed,seconds_per_update=elapsed/50,
        updates_per_second=50/elapsed,num_parameters=sum(p.numel() for p in model.parameters()),
        batch_size=256,history_K=20,width=128,depth=2,cpu_threads=torch.get_num_threads())

result=dict(purpose='Hardware timing on synthetic tensors only; not research metrics',
    torch_version=torch.__version__,platform=platform.platform(),cpu_count=os.cpu_count(),
    cuda_available=torch.cuda.is_available(),runs=[])
if torch.cuda.is_available():result['cuda_device']=torch.cuda.get_device_name(0)
for device in ['cpu','cuda'] if torch.cuda.is_available() else ['cpu']:
    for kind in ['plain','history']:
        row=bench(kind,device);result['runs'].append(row)
        print(json.dumps(row),flush=True)
path=Path(__file__).with_name('compute_benchmark.json')
with path.open('x') as handle:json.dump(result,handle,indent=2)
print('Saved '+str(path),flush=True)
