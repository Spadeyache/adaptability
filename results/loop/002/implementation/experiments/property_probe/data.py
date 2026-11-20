"""Guarded loader for the inventoried main cube datasets.

Labels are separate targets/checks, never concatenated with observations.
The correlation rule is a screening rule, not proof of direct label leakage.
"""
from pathlib import Path
import json
import numpy as np

class InputCorrelationError(ValueError):
    def __init__(self,report):
        self.report=report
        super().__init__('INPUT GUARD FAILED: absolute across-cube correlation exceeds 0.9: '+json.dumps(report['violations']))

def observed_features(z):
    blocks=[];names=[];dropped=[]
    for field in ('state','action','next_state'):
        raw=np.asarray(z[field],dtype=np.float64)
        if field!='action' and raw.shape[1]!=43:
            raise ValueError('Unrecognized raw state schema; explicit schema review required')
        if field=='action' and raw.shape[1] not in (5,8):
            raise ValueError('Unrecognized action schema')
        labels=[f'{field}[{j}]' for j in range(raw.shape[1])]
        if field!='action' and 'relative_position' in z:
            prefix='next_' if field=='next_state' else ''
            extras=[np.asarray(z[prefix+k],dtype=np.float64) for k in ('relative_position','relative_velocity','finger_position_error')]
            raw=np.concatenate([raw,*extras],axis=1)
            labels += [f'{prefix}{k}[{j}]' for k in ('relative_position','relative_velocity','finger_position_error') for j in range(2 if k=='finger_position_error' else 3)]
        # Projected45 is relative Z; projected48 is its backward-difference velocity.
        # Remove both states' columns and descendants before any statistics or normalization.
        omit={42,45,48} if field!='action' else {42,45}
        keep=[j for j in range(raw.shape[1]) if j not in omit]
        dropped += [labels[j] for j in range(raw.shape[1]) if j in omit]
        blocks.append(raw[:,keep]);names += [labels[j] for j in keep]
    x=np.concatenate(blocks,axis=1)
    if not np.isfinite(x).all():raise ValueError('Nonfinite probe input')
    return x,names,dropped

def check_cube_correlations(cube_features,targets,names,*,dataset,split,task):
    """Equal weight per cube; each feature's mean/std/min/max is checked separately."""
    x=np.asarray(cube_features,dtype=np.float64);y=np.asarray(targets,dtype=np.float64)
    if len(x)<3 or y.shape!=(len(x),2):raise ValueError('Need at least three cubes and both labels for guard')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite guard data')
    xc=x-x.mean(0);yc=y-y.mean(0)
    denom=np.sqrt(np.sum(xc*xc,axis=0)[:,None]*np.sum(yc*yc,axis=0)[None,:])
    r=np.divide(xc.T@yc,denom,out=np.zeros((x.shape[1],2)),where=denom>0)
    violations=[dict(feature=names[i],target=('mass','friction')[j],pearson=float(r[i,j])) for i,j in np.argwhere(np.abs(r)>0.9)]
    report=dict(dataset=str(dataset),split=split,task=int(task),cubes=len(x),threshold=0.9,
                method='Absolute Pearson over per-cube mean/std/min/max; tasks and splits checked separately',
                max_absolute_correlation=float(np.max(np.abs(r))),violations=violations,passed=not violations)
    if violations:raise InputCorrelationError(report)
    return report

def load_task(dataset,split,task):
    """No skip-guard switch. Returns episode inputs only if the cube-level guard passes."""
    dataset=Path(dataset);cfg=json.loads((dataset/'dataset.json').read_text())
    if split not in ('train','test'):raise ValueError('Expected existing train/test split')
    start=0 if split=='train' else cfg['train_cubes'];count=cfg[split+'_cubes']
    episodes=[];summary=[];targets=[];names=None;dropped=None
    for cube in range(start,start+count):
        with np.load(dataset/split/f'cube_{cube:04}.npz',allow_pickle=False) as z:
            if not np.all(z['cube_id']==cube):raise ValueError('Cube identity mismatch')
            y=[]
            for key in ('true_mass','true_friction'):
                a=z[key]
                if not np.all(a==a[0]):raise ValueError('Physics label changed within cube')
                y.append(float(a[0]))
            features,fn,removed=observed_features(z)
            if names is not None and names!=fn:raise ValueError('Feature schemas differ')
            names=fn;dropped=removed
            selected=z['task']==task
            x=features[selected]
            if not len(x):raise ValueError('Missing task')
            summary.append(np.concatenate([x.mean(0),x.std(0),x.min(0),x.max(0)]));targets.append(y)
            for episode in np.unique(z['episode_id'][selected]):
                mask=selected & (z['episode_id']==episode)
                episodes.append(dict(cube=cube,episode=int(episode),x=features[mask],target=np.asarray(y)))
    summary_names=[stat+':'+n for stat in ('mean','std','min','max') for n in names]
    report=check_cube_correlations(summary,targets,summary_names,dataset=dataset,split=split,task=task)
    report['dropped_features']=dropped
    return episodes,names,report
