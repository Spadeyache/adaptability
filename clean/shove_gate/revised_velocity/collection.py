"""Fresh, sealed test cubes for the preregistered horizontal-velocity gate.

Old training traces are reused byte-for-byte; retired test traces never enter
this data tree. Simulation and command code are imported unchanged from v5.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent
SEED = 2026111517
TRAIN_IDS = list(range(200))
TEST_IDS = list(range(230, 260))
IDS = TRAIN_IDS + TEST_IDS
PROTOCOL_FILE = ROOT / 'REVISED_PROTOCOL.md'


def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def filename(c, e): return f'cube_{c:03d}_episode_{e:02d}.npz'
def require(condition, message):
    if not condition: raise AssertionError(message)
def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def make_plan(cid, ep):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, cid, ep, 200]))
    force = float(rng.uniform(4., 7.)); direction = float(rng.uniform(-np.pi, np.pi))
    rng.uniform(-.006, .006)  # Preserve v5 generator draw order.
    start = rng.uniform(-.10, .10, size=2).tolist(); yaw = float(rng.uniform(-np.pi, np.pi))
    duration = np.random.default_rng(np.random.SeedSequence([SEED, cid, ep, 300]))
    steps = int(duration.integers(12, 21))
    return dict(cube_id=cid, episode_id=ep, force_n=force, direction_rad=direction,
                duration_steps=steps, duration_s=steps*.005, start_xy=start, yaw_rad=yaw)


def source_paths():
    return [Path(__file__), PROTOCOL_FILE, PRIOR/'script.py', PRIOR/'simulation.py',
            PRIOR.parent/'native.py', PRIOR/'data/cubes.json', PRIOR/'data/plans.json',
            PRIOR/'data/seal.json', PRIOR/'results/metrics.json', PRIOR/'results/full_audit.json']


def verify():
    for name, sha in read(ROOT/'data/seal.json').items():
        require(digest(ROOT/'data'/name)==sha, f'Changed data definition: {name}')
    for name, sha in read(ROOT/'results/source_seal.json').items():
        require(digest(Path(name))==sha, f'Changed source: {name}')
    require(not (ROOT/'results/STOP.json').exists(), 'Collection STOP exists')


def initialize():
    require(PROTOCOL_FILE.is_file(), 'Written revision protocol must precede collection')
    if (ROOT/'data/seal.json').exists():
        verify(); return
    require(read(PRIOR/'results/full_audit.json')['status']=='passed', 'Old data audit not passed')
    for name, sha in read(PRIOR/'data/seal.json').items():
        require(digest(PRIOR/'data'/name)==sha, f'Old data definition changed: {name}')
    cubes = [c for c in read(PRIOR/'data/cubes.json')['cubes'] if c['cube_id'] in TRAIN_IDS]
    for cid in TEST_IDS:
        rng = np.random.default_rng(np.random.SeedSequence([SEED, cid, 100]))
        mass = float(rng.uniform(.18, .50)); mu = float(rng.uniform(.4, 1.))
        cubes.append(dict(cube_id=cid, split='test', mass_kg=mass, friction=mu, combined_friction=(mu+.3)/2))
    plans = [p for p in read(PRIOR/'data/plans.json') if p['cube_id'] in TRAIN_IDS]
    plans += [make_plan(c, e) for c in TEST_IDS for e in range(10)]
    data = ROOT/'data'; results = ROOT/'results'
    protocol = dict(revision='horizontal_velocity_delta_v1', source_physics_revision=5,
        fresh_test_seed=SEED, retired_test_ids=list(range(200,230)),
        mass_range_kg=[.18,.50], cube_friction_range=[.4,1.], floor_material_friction=.3,
        force_range_n=[4.,7.], duration_range_s=[.060,.100], duration_grid_s=.005,
        physics_dt_s=.001, model_dt_s=.005, coast_s=1.5, settle_s=.1, trim_after_stop_steps=5,
        revised_protocol_sha256=digest(PROTOCOL_FILE),
        test_randomization='Fresh seed; independent property stream100, command stream200, duration stream300; no rejection or resampling.',
        reused_training='Existing v5 train cubes0..199 and alltenepisodes; symlinks verified against old saved SHA256.',
        old_test_policy='IDs200..229 excluded; old results untouched.',
        state='Cube physical state15', action='Applied forceXYZ only; exactlyzero coast',
        forbidden_features=['cube_id','episode_id','step','phase','length','padding_mask','stop_time'],
        score_groups=['push','moving_coast','all_steps'])
    write_new(data/'cubes.json', {'cubes':cubes})
    write_new(data/'plans.json', plans)
    write_new(data/'split.json', dict(train=TRAIN_IDS,test=TEST_IDS,retired_test=list(range(200,230)),
                                    episodes_per_cube=10,train_seed=2026111509,test_seed=SEED))
    write_new(data/'protocol.json', protocol)
    write_new(data/'schema.json', read(PRIOR/'data/schema.json'))
    old_rows = [r for r in read(PRIOR/'results/metrics.json') if r['cube_id'] in TRAIN_IDS]
    for kind in ['episodes','raw']:
        (data/kind).mkdir(exist_ok=True)
        for r in old_rows:
            name=filename(r['cube_id'],r['episode_id']); source=PRIOR/'data'/kind/name
            require(digest(source)==r[kind+'_sha256'], f'Reused trace changed: {source}')
            (data/kind/name).symlink_to(source)
    write_new(results/'reused_training_records.json', old_rows)
    write_new(results/'source_seal.json', {str(p.resolve()):digest(p) for p in source_paths()})
    write_new(data/'seal.json', {n:digest(data/n) for n in ['cubes.json','split.json','plans.json','protocol.json','schema.json']})
    verify()


def collect():
    started=time.perf_counter(); initialize(); verify()
    from clean.shove_gate.simulation import simulate,measure,trim_episode
    results=ROOT/'results'; data=ROOT/'data'
    cubes={c['cube_id']:c for c in read(data/'cubes.json')['cubes']}
    plans=[p for p in read(data/'plans.json') if p['cube_id'] in TEST_IDS]
    ledger=results/'fresh_collection_records.jsonl'
    rows=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    done={(r['cube_id'],r['episode_id']) for r in rows}
    for p in plans:
        key=p['cube_id'],p['episode_id']
        if key in done: continue
        began=time.perf_counter(); episode,raw=simulate(cubes[key[0]],p)
        row=dict(cube_id=key[0],episode_id=key[1],**measure(raw,p))
        episode,trim=trim_episode(episode,raw,row); row.update(trim)
        for kind,arrays in [('episodes',episode),('raw',raw)]:
            path=data/kind/filename(*key)
            require(not path.exists(), f'Unindexed existing trace: {path}')
            np.savez_compressed(path,**arrays); row[kind+'_sha256']=digest(path)
        row['wall_seconds']=time.perf_counter()-began
        with ledger.open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        rows.append(row)
        failures=[]
        if not row['stopped']: failures.append('No confirmed rest within fixed raw horizon')
        if row['max_coast_force_n']!=0: failures.append('Nonzero coast force')
        if row['max_tilt_deg']>5: failures.append('Tilt over5degrees')
        if not .019<row['min_z_m']<=row['max_z_m']<.021: failures.append('Cube left flat surface')
        if failures:
            write_new(results/'STOP.json',dict(cube_id=key[0],episode_id=key[1],reasons=failures))
            raise RuntimeError(str(failures))
        if key[1]==9: print(json.dumps(dict(cube_id=key[0],fresh_episodes=len(rows))),flush=True)
    require(len(rows)==300, 'Incomplete fresh test collection')
    allrows=read(results/'reused_training_records.json')+rows
    for name,values in [('fresh_metrics.json',rows),('metrics.json',allrows)]:
        path=results/name
        if path.exists(): require(read(path)==values, f'Changed completed metrics: {name}')
        else: write_new(path,values)
    timing=dict(fresh_cubes=30,fresh_episodes=300,reused_training_cubes=200,reused_training_episodes=2000,
                this_invocation_seconds=time.perf_counter()-started,
                fresh_episode_seconds_sum=sum(r['wall_seconds'] for r in rows),
                fresh_transitions=sum(r['retained_steps'] for r in rows),
                fresh_trimmed_steps=sum(r['trimmed_steps'] for r in rows))
    if not (results/'collection_timing.json').exists(): write_new(results/'collection_timing.json',timing)
    verify(); print(json.dumps(timing),flush=True)


def aggregate(rows):
    coast=sum(r['retained_coast_steps'] for r in rows); moving=sum(r['retained_moving_coast_steps'] for r in rows)
    return dict(cubes=len({r['cube_id'] for r in rows}),episodes=len(rows),
        launches=sum(r['launch_speed_m_s']>.02 for r in rows),launch_fraction=sum(r['launch_speed_m_s']>.02 for r in rows)/len(rows),
        moving_coast_fraction=moving/coast,coast_transitions=coast,moving_coast_transitions=moving,
        transitions=sum(r['retained_steps'] for r in rows),trimmed_transitions=sum(r['trimmed_steps'] for r in rows),
        maximum_coast_distance_m=max(r['stopping_distance_m'] for r in rows),
        maximum_time_to_stop_s=max(r['time_to_stop_s'] for r in rows),
        all_stopped=all(r['stopped'] for r in rows),
        coast_distance_at_least_1m_episodes=[dict(cube_id=r['cube_id'],episode_id=r['episode_id'],distance_m=r['stopping_distance_m']) for r in rows if r['stopping_distance_m']>=1.])


def audit():
    verify(); data=ROOT/'data'; results=ROOT/'results'
    rows=read(results/'metrics.json'); cubes=read(data/'cubes.json')['cubes']; plans=read(data/'plans.json')
    identities={(c,e) for c in IDS for e in range(10)}
    require({(r['cube_id'],r['episode_id']) for r in rows}==identities and len(rows)==2300,'Wrong trace identities')
    require({c['cube_id'] for c in cubes}==set(IDS) and len(cubes)==230,'Wrong cube definitions')
    require({(p['cube_id'],p['episode_id']) for p in plans}==identities and len(plans)==2300,'Wrong plans')
    table={(p['cube_id'],p['episode_id']):p for p in plans}
    old_cubes={c['cube_id']:c for c in read(PRIOR/'data/cubes.json')['cubes']}
    old_plans={(p['cube_id'],p['episode_id']):p for p in read(PRIOR/'data/plans.json')}
    for c in cubes:
        cid=c['cube_id']
        if cid in TRAIN_IDS: require(c==old_cubes[cid],'Training label changed')
        else:
            rng=np.random.default_rng(np.random.SeedSequence([SEED,cid,100]))
            require(c['mass_kg']==float(rng.uniform(.18,.50)) and c['friction']==float(rng.uniform(.4,1.)),'Fresh label generator mismatch')
            require(c['combined_friction']==(c['friction']+.3)/2,'Combined coefficient mismatch')
    for key,p in table.items():
        require(p==(old_plans[key] if key[0] in TRAIN_IDS else make_plan(*key)),'Plan changed')
        require(not any(k in p for k in ['mass_kg','friction','combined_friction']),'Plan includes physical label')
    expected_names={filename(*key) for key in identities}
    for kind in ['episodes','raw']:
        require({p.name for p in (data/kind).glob('*.npz')}==expected_names,'Unexpected trace files')
    max_action_error=0.
    for r in rows:
        key=r['cube_id'],r['episode_id']; p=table[key]; name=filename(*key)
        for kind in ['episodes','raw']:
            path=data/kind/name; require(digest(path)==r[kind+'_sha256'],'Trace hash mismatch')
            require(path.is_symlink()==(key[0] in TRAIN_IDS),'Incorrect new/reused trace storage')
        with np.load(data/'episodes'/name,allow_pickle=False) as ep,np.load(data/'raw'/name,allow_pickle=False) as raw:
            require(set(ep.files)=={'cube_id','episode_id','step','phase','state','action','next_state'},'Unexpected model schema')
            require(set(raw.files)=={'state','action','phase','release_index','physics_dt_s'},'Unexpected raw fields')
            s=raw['state']; a=raw['action']; release=int(raw['release_index']); n=len(ep['step']); push=p['duration_steps']
            require(float(raw['physics_dt_s'])==.001 and release==push*5,'Wrong timing')
            require(s.shape==(release+1501,15) and a.shape==(release+1500,3),'Wrong raw horizon')
            require(np.array_equal(raw['phase'],[0]*release+[1]*1500),'Wrong raw phases')
            require(ep['state'].shape==ep['next_state'].shape==(n,15) and ep['action'].shape==(n,3),'Wrong model dimensions')
            require(np.array_equal(ep['step'],np.arange(n)),'Wrong step IDs')
            require(np.all(ep['cube_id']==key[0]) and np.all(ep['episode_id']==key[1]),'Wrong cube/episode IDs')
            require(np.array_equal(ep['phase'],['push']*push+['coast']*(n-push)),'Wrong model phases')
            selected=s[::5].astype(np.float32)
            require(np.array_equal(ep['state'],selected[:n]) and np.array_equal(ep['next_state'],selected[1:n+1]),'State alignment mismatch')
            require(np.array_equal(ep['action'],a[::5][:n]),'Saved action is not applied force')
            expected=np.zeros_like(a); expected[:release]=p['force_n']*np.array([np.cos(p['direction_rad']),np.sin(p['direction_rad']),0.])
            error=float(np.max(np.abs(a-expected))); max_action_error=max(max_action_error,error)
            require(error<1e-6 and np.all(a[release:]==0) and np.all(a[:,2]==0),'Force schedule mismatch')
            require(all(np.isfinite(v).all() for v in [s,a,ep['state'],ep['next_state']]),'Nonfinite trace')
            resting=(np.linalg.norm(s[:,9:12],axis=1)<.001)&(np.linalg.norm(s[:,12:15],axis=1)<.05)
            candidates=np.flatnonzero(np.convolve(resting[release+1:].astype(int),np.ones(100,dtype=int),mode='valid')==100)
            require(len(candidates)>0,'No confirmed rest')
            stop=release+1+int(candidates[0])
            if stop==release+1 and resting[release]: stop=release
            require(r['stop_raw_index']==stop and r['stopped'],'Wrong stop onset')
            require(n==min(len(selected)-1,int(np.ceil(stop/5))+5),'Wrong trimming endpoint')
            moving=(np.linalg.norm(selected[:-1,9:12],axis=1)>.001)|(np.linalg.norm(selected[:-1,12:15],axis=1)>.05)
            require(not np.any(moving[n:]),'Moving steps trimmed')
            expected_fields=dict(retained_steps=n,untrimmed_steps=len(selected)-1,trimmed_steps=len(selected)-1-n,
                                 retained_coast_steps=n-push,retained_moving_coast_steps=int(moving[push:n].sum()))
            require(all(r[k]==v for k,v in expected_fields.items()),'Wrong saved step counts')
            require(np.isclose(r['launch_speed_m_s'],np.linalg.norm(s[release,9:11])),'Wrong launch speed')
            require(np.isclose(r['stopping_distance_m'],np.linalg.norm(s[stop,:2]-s[release,:2])),'Wrong coast distance')
            require(np.isclose(r['time_to_stop_s'],(stop-release)*.001),'Wrong stop time')
            require(np.allclose(s[0,:2],p['start_xy'],atol=1e-6),'Wrong start position')
            yaw=p['yaw_rad']; require(np.allclose(s[0,3:9],[np.cos(yaw),np.sin(yaw),0,-np.sin(yaw),np.cos(yaw),0],atol=2e-5),'Wrong start yaw')
            normals=np.cross(s[:,3:6],s[:,6:9]); tilt=np.degrees(np.arccos(np.clip(normals[:,2],-1,1)))
            require(tilt.max()<5 and .019<s[:,2].min()<=s[:,2].max()<.021,'Cube left sliding regime')
    result=dict(status='passed',**aggregate(rows),fresh_test_summary=aggregate([r for r in rows if r['cube_id'] in TEST_IDS]),
        training_summary=aggregate([r for r in rows if r['cube_id'] in TRAIN_IDS]),
        fresh_test_ids=TEST_IDS,retired_test_ids=list(range(200,230)),fresh_test_seed=SEED,
        maximum_force_plan_error_n=max_action_error,reused_training_bytes_unchanged=True,
        old_test_traces_excluded=True,no_motion_based_test_resampling=True,
        checks=['data/source SHA256 seals','independent fresh RNG streams','train identities and bytes preserved',
                'all2300 raw/model transitions aligned','applied horizontal COM force and exact zero coast',
                'stop confirmation and five-step tail recomputed','no moving steps trimmed','finite flat sliding trajectories'])
    out=results/'full_audit.json'
    if out.exists(): require(read(out)==result,'Audit changed')
    else: write_new(out,result)
    print(json.dumps(result,indent=2),flush=True)


def probe():
    verify(); from clean.shove_gate.probe_v5 import generator
    out=ROOT/'results/action_probe.json'
    if out.exists(): print(json.dumps({'status':'already_complete','result':str(out)})); return
    started=time.perf_counter(); data=ROOT/'data'
    cubes={c['cube_id']:c for c in read(data/'cubes.json')['cubes']}
    result=dict(generator=generator(read(data/'plans.json'),cubes),
        source_sha256=digest(PRIOR/'probe_v5.py'),
        inputs_sha256={n:digest(data/n) for n in ['cubes.json','plans.json']},
        elapsed_seconds=time.perf_counter()-started,
        new_test_motion_used=False,forbidden_length_features_excluded=True,
        interpretation='Fixed planned-command features only. Hyperparameters selected on training cubes; no test-based tuning or resampling.')
    write_new(out,result); print(json.dumps({'elapsed_seconds':result['elapsed_seconds'],'generator':result['generator']['aggregate']},indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['initialize','collect','audit','probe'])
    args=parser.parse_args(); globals()[args.command]()


if __name__=='__main__': main()
