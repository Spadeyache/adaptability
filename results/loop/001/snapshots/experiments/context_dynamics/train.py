"""Train the context encoder and world model jointly on multistep state predictions."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from robot_adaptation.estimators.context_dynamics.model import ContextDynamics
from .data import Trajectories


@torch.no_grad()
def validation(model, data, ids, length):
    total, static, count = 0.,0.,0
    for block in ids.split(64):
        states, commands = data.tiled(block,length)
        context = model.encode(data.constants[block]).repeat_interleave(len(states)//len(block),dim=0)
        losses = model.loss_per_window(states,commands,context)
        residual = (states[:,0:1]-states[:,1:])*(model.state_sd/model.delta_sd)
        horizon = torch.arange(1,length+1,device=data.device)
        static += (residual.square().mean(-1)/horizon).mean(-1).sum().item()
        total += losses.sum().item()
        count += len(losses)
    return total/count, static/count


def train(data_path, dest, latent, width, horizon, updates, batch, learning_rate, seed, device,
          extra=None, balance_state=False):
    dest.mkdir(parents=True,exist_ok=False)
    config = dict(dataset=str(data_path.resolve()),latent=latent,width=width,horizon=horizon,
                  updates=updates,batch=batch,learning_rate=learning_rate,seed=seed,device=device,
                  extra_training=[str(path.resolve()) for path in extra or []],
                  balance_state=balance_state)
    (dest/'settings.json').write_text(json.dumps(config,indent=2))
    sources = dest/'source'
    sources.mkdir()
    for path in [Path(__file__),Path(__file__).with_name('data.py'),
                 Path('robot_adaptation/estimators/context_dynamics/model.py')]:
        shutil.copy2(path,sources/path.name)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    data = Trajectories(data_path,device)
    for path in extra or []:
        data.add_training_episodes(path)
    model = ContextDynamics(latent,width).to(device)
    data.fit_scaling(model)
    data.states = model.normalize(data.states)
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,
                                  fused=device.startswith('cuda'),weight_decay=1e-5)
    weights = None
    if balance_state:
        # Equal contribution per observed field, including gripper width.
        sizes = (9,9,3,6,1,3,6,3,3)
        weights = torch.tensor([43/(len(sizes)*k) for k in sizes for _ in range(k)],device=device)
    best = np.inf
    clock = time.monotonic()
    for step in range(1,updates+1):
        ids = data.train_ids[torch.randint(len(data.train_ids),(batch,),device=device)]
        times = torch.randint(data.horizon-horizon+1,(batch,),device=device)
        states, commands = data.windows(ids,times,horizon)
        z = model.encode(data.constants[ids])
        loss = model.loss_per_window(states,commands,z,weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        fraction = step/updates
        optimizer.param_groups[0]['lr'] = learning_rate*(.12+.88*(1+math.cos(math.pi*fraction))/2)
        if step==1 or step%500==0 or step==updates:
            score, static = validation(model,data,data.select_ids,horizon)
            if not np.isfinite(score):
                raise ValueError('Nonfinite validation rollout')
            row = dict(update=step,validation=score,persistence=static,train_loss=loss.item(),
                       elapsed_s=time.monotonic()-clock)
            print(json.dumps(row),flush=True)
            with (dest/'training.jsonl').open('a') as fp:
                fp.write(json.dumps(row)+'\n')
            if score < best:
                best = score
                torch.save(dict(model=model.state_dict(),latent=latent,width=width,
                    horizon=horizon,validation=score,update=step),dest/'model.tmp')
                os.replace(dest/'model.tmp',dest/'model.pt')
    (dest/'complete.json').write_text(json.dumps(dict(best=best,updates=updates)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--latent',type=int,default=8)
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--horizon',type=int,default=5)
    p.add_argument('--updates',type=int,default=8000)
    p.add_argument('--batch',type=int,default=384)
    p.add_argument('--lr',type=float,default=.0005)
    p.add_argument('--seed',type=int,default=9101)
    p.add_argument('--device',default='cuda')
    p.add_argument('--extra-training',type=Path,action='append')
    p.add_argument('--balance-state',action='store_true')
    a=p.parse_args()
    train(a.data,a.output,a.latent,a.width,a.horizon,a.updates,a.batch,a.lr,a.seed,a.device,a.extra_training,a.balance_state)
