"""Paired context ablations, coordinate sweeps and held-out linear readouts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import Trajectories, restore
from robot_adaptation.simulation.context_dynamics import CONSTANTS, TASKS


def as_array(tensor):
    return tensor.detach().cpu().numpy()


def write_csv(path, rows):
    with path.open('w',newline='') as fp:
        writer = csv.DictWriter(fp,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def evaluate(model, data, ids, contexts, length):
    pieces = []
    for block, z in zip(ids.split(128),contexts.split(128)):
        states, commands = data.tiled(block,length)
        windows = len(states)//len(block)
        loss = model.loss_per_window(states,commands,z.repeat_interleave(windows,dim=0))
        pieces.append(as_array(loss.reshape(len(block),windows).mean(-1)))
    return np.concatenate(pieces)


def probe(train_z, test_z, train_physics, test_physics):
    zmean,zsd = train_z.mean(0),train_z.std(0).clip(1e-8)
    design = (train_z-zmean)/zsd
    ymean,ysd = train_physics.mean(0),train_physics.std(0)
    weights = np.linalg.lstsq(design,(train_physics-ymean)/ysd,rcond=None)[0]
    prediction = ((test_z-zmean)/zsd @ weights)*ysd + ymean
    r2 = 1-((prediction-test_physics)**2).sum(0)/((test_physics-test_physics.mean(0))**2).sum(0)
    return r2, dict(z_mean=zmean,z_sd=zsd,weights=weights,y_mean=ymean,y_sd=ysd)


def plot(dest, ablation, offsets, example, population, r2, episode_id):
    fig, axes = plt.subplots(2,2,figsize=(11.5,7.7),layout='constrained')
    ax = axes[0,0]
    keys = list(ablation)
    ax.bar(keys,[ablation[k]['loss'] for k in keys],color=['#16817f','#cb6752','#899ab4','#a2aaba'])
    ax.set(ylabel='Mean normalized rollout loss',title='Ablations on the same held-out motions')
    for area, curves, title in [(axes[0,1], example,f'Coordinate sweeps on Pick episode {episode_id}'),
                                (axes[1,0], population,'Coordinate sweeps across the held-out fold')]:
        for k,curve in enumerate(curves):
            area.plot(offsets, curve/curve[len(offsets)//2],label=f'z{k+1}')
        area.axvline(0,color='0.4',ls=':')
        area.set(xlabel='Offset from encoded context (training SD)',ylabel='Loss / encoded-context loss',title=title)
        area.legend(ncol=4,fontsize=7)
    ax=axes[1,1]
    names = ['Mass','Cube friction','Table friction','Grip cap','Arm damping','Arm stiffness',
             'Action delay','COM x','COM y','COM z']
    ax.barh(names[::-1],r2[::-1],color='#427ca6')
    ax.set(xlabel='Held-out linear probe R²',title='Readout from the training encoder')
    fig.savefig(dest/'checks.png',dpi=175)
    fig.savefig(dest/'checks.pdf')
    plt.close(fig)


@torch.inference_mode()
def check(folder, run, fold, device, external=None):
    if not (run/'complete.json').is_file():
        raise ValueError('Training must finish before collapse checks')
    dest = run/('checks_external' if external else f'checks_{fold}')
    dest.mkdir(exist_ok=False)
    model, saved = restore(run,device)
    data = Trajectories(folder,device)
    extra = json.loads((run/'settings.json').read_text()).get('extra_training')
    for path in ([extra] if isinstance(extra,str) else extra or []):
        data.add_training_episodes(Path(path))
    train_physics = data.constants[data.train_ids]
    training = model.encode(train_physics)
    length = saved['horizon']
    ids = data.check_ids[fold]
    if external:
        if external.resolve() in [folder.resolve(), *(Path(p).resolve() for p in ([extra] if isinstance(extra,str) else extra or []))]:
            raise ValueError('Confirmation data must be independent of all fitted datasets')
        del data
        data = Trajectories(external,device)
        ids = torch.arange(len(data.states),device=device)
    data.states = model.normalize(data.states)
    z = model.encode(data.constants[ids])
    mean, sd = training.mean(0), training.std(0)
    tasks = as_array(data.tasks[ids])
    rng = np.random.default_rng(28241+fold)
    sources = np.arange(len(ids))
    for task in np.unique(tasks):
        same = rng.permutation(np.flatnonzero(tasks==task))
        sources[same] = np.roll(same,1)
    assert np.all(sources!=np.arange(len(ids))) and np.array_equal(tasks,tasks[sources])
    contexts = {'Encoded': z,'Swapped': z[sources], 'Mean': mean.expand_as(z), 'Zero': torch.zeros_like(z)}
    errors = {name:evaluate(model,data,ids,ctx,length) for name,ctx in contexts.items()}
    draws = rng.integers(len(ids),size=(5000,len(ids)))
    ablation = {}
    for name,value in errors.items():
        diff = value-errors['Encoded']
        ci = np.quantile(diff[draws].mean(1),[.025,.975])
        relative = value.mean()/errors['Encoded'].mean()-1
        ablation[name] = dict(loss=float(value.mean()),relative_increase=float(relative),
                              diff_ci=ci.tolist(),passes=bool(relative>=.01 and ci[0]>0))
    persistence = 0.
    for block in ids.split(64):
        states, commands = data.tiled(block,length)
        delta = (states[:,0:1]-states[:,1:])*(model.state_sd/model.delta_sd)
        loss = (delta.square().mean(-1)/torch.arange(1,length+1,device=device)).mean().item()
        persistence += len(block)*loss/len(ids)

    example_idx = int(np.flatnonzero(tasks==0)[0])
    example_id = ids[example_idx].item()
    offsets = np.linspace(-2,2,17)
    curves = np.empty((model.latent,len(offsets),len(ids)),np.float32)
    for dim in range(model.latent):
        for i,off in enumerate(offsets):
            if off == 0:
                curves[dim,i] = errors['Encoded']
            else:
                perturbed = z.clone()
                perturbed[:,dim] += off*sd[dim]
                curves[dim,i] = evaluate(model,data,ids,perturbed,length)
    population = curves.mean(-1)
    example = curves[:,:,example_idx]
    sweep = []
    for dim, curve in enumerate(example):
        best = float(offsets[np.argmin(curve)])
        passed = curve[8] < min(curve[4],curve[12]) and abs(best)<=.5
        sweep.append(dict(dim=dim+1,best_offset_sd=best,encoded_loss=float(curve[8]),
                          min_loss=float(curve.min()),passes=bool(passed),
                          population_best_offset_sd=float(offsets[np.argmin(population[dim])])))
    r2, readout = probe(as_array(training),as_array(z), as_array(train_physics),
                        as_array(data.constants[ids]))
    np.savez(dest/'readout.npz',**readout)
    np.savez_compressed(dest/'sweeps.npz',episode_ids=as_array(ids),offsets=offsets,loss=curves)
    episode_rows = []
    for i, episode in enumerate(as_array(ids)):
        episode_rows.append(dict(episode=int(episode),task=TASKS[tasks[i]],
                                 swapped_episode=int(ids[sources[i]]),
                                 **{name:float(loss[i]) for name,loss in errors.items()}))
    write_csv(dest/'episode_errors.csv',episode_rows)
    gates = dict(forecasts=ablation['Encoded']['loss']<persistence,
                 swap=ablation['Swapped']['passes'], mean=ablation['Mean']['passes'],
                 zero=ablation['Zero']['passes'], sweep=all(x['passes'] for x in sweep),
                 readouts=bool(np.isfinite(r2).all()))
    result = dict(fold=fold,external=str(external) if external else None,episodes=len(ids),horizon=length,latent=model.latent,persistence=persistence,
                  example_id=example_id,ablation=ablation,coordinate_sweep=sweep,
                  linear_probe_r2=dict(zip(CONSTANTS,r2.tolist())),gates=gates,passed=all(gates.values()))
    write_json(dest/'results.json',result)
    plot(dest,ablation,offsets,example,population,r2,example_id)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path)
    p.add_argument('run',type=Path)
    p.add_argument('--fold',type=int,choices=(0,1),default=0)
    p.add_argument('--device',default='cuda')
    p.add_argument('--external',type=Path)
    a=p.parse_args()
    check(a.data,a.run,a.fold,a.device,a.external)
