"""Direct-force action-only probes; never fit prediction or belief models.

Only the preplanned rectangular force pulse is an admissible input. Its commanded
on-time is independently randomized and is distinct from observed episode length.
The latter is tested separately as an explicitly forbidden side channel.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT=Path(__file__).resolve().parent
SEEDS=(101,211,307,401,503)
TARGETS=('mass_kg','friction','combined_friction')
FEATURES=('applied_force_x_N','applied_force_y_N','applied_force_z_N','force_magnitude_N','commanded_duration_s')


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pulse(plan):
    """Independent reconstruction from public command settings only."""
    magnitude=float(plan['force_n'])
    angle=float(plan['direction_rad'])
    force=magnitude*np.array([np.cos(angle),np.sin(angle),0.])
    duration=float(plan['duration_s'])
    return force,duration


def feature(plan):
    force,duration=pulse(plan)
    return np.r_[force,np.linalg.norm(force),duration]


def model(family,parameter,seed):
    if family=='polynomial_ridge':
        return make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=parameter))
    if family=='extra_trees':
        return ExtraTreesRegressor(n_estimators=192,min_samples_leaf=int(parameter),max_features=1.,random_state=seed,n_jobs=2)
    raise ValueError(family)


def predict(model_object,x,y,train,heldout):
    center=y[train].mean(axis=0);scale=y[train].std(axis=0)
    if np.any(scale<=0):raise ValueError('Degenerate fitting labels')
    model_object.fit(x[train],((y[train]-center)/scale)[:,[0,2]])
    independent=model_object.predict(x[heldout])*scale[[0,2]]+center[[0,2]]
    output=np.empty((int(heldout.sum()),3));output[:,[0,2]]=independent
    affine=np.linalg.lstsq(np.c_[y[train,2],np.ones(int(train.sum()))],y[train,1],rcond=None)[0]
    output[:,1]=independent[:,1]*affine[0]+affine[1]
    return output


def aggregate(records,baseline):
    result={}
    for family in sorted({r['family'] for r in records}):
        values=np.array([[r['r2'][t] for t in TARGETS] for r in records if r['family']==family])
        result[family]={t:dict(mean_r2=float(values[:,j].mean()),seed_std_r2=float(values[:,j].std(ddof=1)),
                               min_r2=float(values[:,j].min()),max_r2=float(values[:,j].max()),
                               mean_r2_minus_baseline=float(values[:,j].mean()-baseline[j])) for j,t in enumerate(TARGETS)}
    return result


def generator(plans,cubes):
    x=np.array([feature(p) for p in plans]);ids=np.array([p['cube_id'] for p in plans])
    y=np.array([[cubes[int(cid)][t] for t in TARGETS] for cid in ids],dtype=np.float64)
    train=ids<200;test=ids>=200
    if train.sum()!=2000 or test.sum()!=300:raise ValueError('Expected 200/30 cubes with ten plans each')
    baseline=r2_score(y[test],np.broadcast_to(y[train].mean(axis=0),y[test].shape),multioutput='raw_values')
    records=[]
    grids={'polynomial_ridge':(1.,10.,100.,1000.,10000.),'extra_trees':(8,32,128)}
    for seed in SEEDS:
        order=np.random.default_rng(seed).permutation(200)
        fit=np.isin(ids,order[:180]);validation=np.isin(ids,order[180:])
        scale=y[fit].std(axis=0)
        for family,grid in grids.items():
            candidates=[]
            for parameter in grid:
                prediction=predict(model(family,parameter,seed),x,y,fit,validation)
                mse=float(np.mean(np.square((prediction[:,[0,2]]-y[validation][:,[0,2]])/scale[[0,2]])))
                candidates.append((mse,parameter))
            error,parameter=min(candidates)
            prediction=predict(model(family,parameter,seed),x,y,train,test)
            scores=r2_score(y[test],prediction,multioutput='raw_values')
            records.append(dict(family=family,seed=seed,selected_parameter=parameter,
                                validation_standardized_mse=error,r2=dict(zip(TARGETS,map(float,scores)))))
    return dict(scope='200/30 independent cube split using planned force commands only; no test motion.',
                feature_names=FEATURES,constant_train_mean_r2=dict(zip(TARGETS,map(float,baseline))),
                selection='180/20 training-cube validation only, then refit 200; no test tuning.',
                records=records,aggregate=aggregate(records,baseline),
                all_r2_below_0_1=all(r['r2'][t]<.1 for r in records for t in TARGETS))


def pilot(root,plans,cubes):
    table={(p['cube_id'],p['episode_id']):p for p in plans}
    rows=[];ids=[];lengths=[];y=[];hashes={}
    for cid in range(10):
        for episode_id in range(10):
            name=f'cube_{cid:03d}_episode_{episode_id:02d}.npz'
            path=root/'data/episodes'/name;raw_path=root/'data/raw'/name
            hashes[name]=digest(path)
            with np.load(path,allow_pickle=False) as ep,np.load(raw_path,allow_pickle=False) as raw:
                plan=table[(cid,episode_id)];force,duration=pulse(plan)
                applied=raw['action'];dt=float(raw['physics_dt_s'])
                start=int(raw['push_start_index']) if 'push_start_index' in raw.files else 0
                window=applied[start:start+round(.100/dt)]
                if window.shape!=(round(.100/dt),3):raise ValueError('Insufficient fixed100ms raw command window')
                active=np.linalg.norm(window,axis=1)>1e-10
                n=round(duration/dt)
                if not np.array_equal(active,np.arange(len(window))<n):raise ValueError('Applied pulse on-time differs from plan')
                if not np.allclose(window[:n],force,atol=1e-8,rtol=1e-7) or not np.all(window[n:]==0):
                    raise ValueError('Raw force window differs from commanded pulse')
                # Read only force commands and independently commanded duration;
                # raw full window has fixed length regardless of rest or trimming.
                rows.append(np.r_[window[0],np.linalg.norm(window[0]),n*dt])
                lengths.append(len(ep['step']));ids.append(cid)
                y.append([cubes[cid][t] for t in TARGETS])
    x=np.asarray(rows);lengths=np.asarray(lengths,dtype=float)[:,None];ids=np.asarray(ids);y=np.asarray(y)
    baseline=np.empty_like(y)
    for cid in range(10):baseline[ids==cid]=y[ids!=cid].mean(axis=0)
    baseline_r2=r2_score(y,baseline,multioutput='raw_values');baseline_sse=np.square(y-baseline).sum(axis=0)
    groups={}
    for name,matrix in [('commands_only',x),('retained_length_only',lengths),('commands_plus_retained_length',np.c_[x,lengths])]:
        records=[]
        for family,parameter in [('polynomial_ridge',100.),('extra_trees',5)]:
            for seed in SEEDS:
                prediction=np.empty_like(y)
                for cid in range(10):
                    fit=ids!=cid;heldout=ids==cid
                    prediction[heldout]=predict(model(family,parameter,seed),matrix,y,fit,heldout)
                scores=r2_score(y,prediction,multioutput='raw_values')
                skill=1.-np.square(y-prediction).sum(axis=0)/baseline_sse
                records.append(dict(family=family,seed=seed,fixed_parameter=parameter,
                                    r2=dict(zip(TARGETS,map(float,scores))),
                                    baseline_relative_skill=dict(zip(TARGETS,map(float,skill)))))
        groups[name]=dict(records=records,aggregate=aggregate(records,baseline_r2))
    return dict(scope='Actual saved ten-cube direct-force pilot; pooled leave-one-cube-out predictions; fixed probe hyperparameters.',
                n_cubes=10,n_episodes=100,feature_names=FEATURES,episode_sha256=hashes,
                fixed_raw_command_window_s=.100,commanded_duration_is_randomized_not_stop_time=True,
                fold_mean_baseline_r2=dict(zip(TARGETS,map(float,baseline_r2))),feature_sets=groups,
                interpretation='With ten equal cube groups, leave-one-cube-out constant means have R2=-0.234568. Negative pilot R2 is not silently called zero. Length features are forbidden outcome-derived diagnostics, not permitted model inputs.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args();root=args.root.resolve()
    started=time.perf_counter();out=root/'results/action_probe.json'
    source=digest(__file__);inputs={n:digest(root/'data'/n) for n in ['cubes.json','plans.json']}
    if out.exists():
        previous=read(out)
        if previous.get('source_sha256')!=source or previous.get('input_sha256')!=inputs:raise RuntimeError('Refusing to overwrite different probe evidence')
        print(json.dumps({'status':'already_complete','result':str(out)}));return
    cubes={c['cube_id']:c for c in read(root/'data/cubes.json')['cubes']};plans=read(root/'data/plans.json')
    result=dict(generator=generator(plans,cubes),pilot=pilot(root,plans,cubes),
                source_sha256=source,input_sha256=inputs,elapsed_seconds=time.perf_counter()-started,
                no_prediction_or_belief_model_training=True)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'result':str(out),'elapsed_seconds':result['elapsed_seconds'],'generator':result['generator']['aggregate'],
                      'pilot_commands':result['pilot']['feature_sets']['commands_only']['aggregate']},indent=2))


if __name__=='__main__':main()
