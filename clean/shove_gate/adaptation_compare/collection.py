"""Step 3 byte-identical training reuse and post-selection fresh test collection.

No fresh property, command, or duration draw occurs in prepare_training().
All fresh generation requires a frozen, explicitly announced validation setting.
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
STEP2 = PRIOR / 'push_latent'
SEED = 2026111613
TRAIN_IDS = list(range(180))
VALIDATION_IDS = list(range(180, 200))
REUSED_IDS = list(range(200))
TEST_IDS = list(range(290, 320))
IDS = REUSED_IDS + TEST_IDS
PROTOCOL_FILE = ROOT / 'PROTOCOL.md'
PROTOCOL_SHA256 = 'b7c290b7e65cad349cb22cacd278e8f0c4b81d7316b65195f0237d1cb3053675'


def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def logical_digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def filename(c, e): return f'cube_{c:03d}_episode_{e:02d}.npz'
def require(condition, message):
    if not condition: raise AssertionError(message)
def write_new(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
def replace_derived(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    temporary.replace(path)


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
    return [Path(__file__),PROTOCOL_FILE,PRIOR/'script.py',PRIOR/'simulation.py',
            PRIOR.parent/'native.py',STEP2/'collection.py',STEP2/'data/cubes.json',
            STEP2/'data/plans.json',STEP2/'data/schema.json',STEP2/'results/metrics.json',
            STEP2/'results/final_audit.json',STEP2/'results/checkpoint_seal.json',
            STEP2/'results/normalizer.json']


def retired_manifest():
    inherited=read(STEP2/'results/retired_experiment_seal.json')
    for name,expected in inherited.items(): require(digest(name)==expected,f'Earlier retired artifact changed: {name}')
    result=dict(inherited)
    for path in STEP2.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ['.pyc','.tmp']:
            resolved=str(path.resolve());actual=digest(path)
            require(resolved not in result or result[resolved]==actual,f'Retired source mismatch: {path}')
            result[resolved]=actual
    return dict(sorted(result.items()))


def verify_prepared():
    require(digest(PROTOCOL_FILE)==PROTOCOL_SHA256,'Written protocol changed')
    require(not (ROOT/'results/STOP.json').exists(),'STOP artifact exists')
    for manifest in ['source_seal.json','retired_experiment_seal.json','prepared_training_seal.json']:
        for name,sha in read(ROOT/'results'/manifest).items():
            require(digest(ROOT/name)==sha,f'Changed sealed artifact: {name}')
    preparation=read(ROOT/'results/preparation.json')
    cubes=[c for c in read(ROOT/'data/cubes.json')['cubes'] if c['cube_id'] in REUSED_IDS]
    plans=[p for p in read(ROOT/'data/plans.json') if p['cube_id'] in REUSED_IDS]
    require(len(cubes)==200 and logical_digest(cubes)==preparation['training_cubes_logical_sha256'],'Training properties changed')
    require(len(plans)==2000 and logical_digest(plans)==preparation['training_plans_logical_sha256'],'Training plans changed')
    require(read(ROOT/'data/split.json')['train']==TRAIN_IDS and read(ROOT/'data/split.json')['validation']==VALIDATION_IDS,'Training split changed')
    return preparation


def prepare_training():
    require(PROTOCOL_FILE.is_file() and digest(PROTOCOL_FILE)==PROTOCOL_SHA256,'Written protocol must precede preparation')
    if (ROOT/'results/preparation.json').exists(): return verify_prepared()
    require(read(STEP2/'results/final_audit.json')['status']=='passed','Step 2 final audit did not pass')
    for name,sha in read(STEP2/'data/seal.json').items():require(digest(STEP2/'data'/name)==sha,f'Step 2 data definition changed: {name}')
    cp=read(STEP2/'results/checkpoint_seal.json')
    require(len(cp['checkpoints'])==35,'Expected all35 frozen Step 2 checkpoints')
    for name,sha in cp['checkpoints'].items():require(digest(STEP2/'checkpoints'/name)==sha,f'Step 2 checkpoint changed: {name}')
    cubes=[c for c in read(STEP2/'data/cubes.json')['cubes'] if c['cube_id'] in REUSED_IDS]
    plans=[p for p in read(STEP2/'data/plans.json') if p['cube_id'] in REUSED_IDS]
    old_rows=[r for r in read(STEP2/'results/metrics.json') if r['cube_id'] in REUSED_IDS]
    require(len(cubes)==200 and len(plans)==2000 and len(old_rows)==2000,'Incomplete reused training population')
    data=ROOT/'data';results=ROOT/'results';training_seal={}
    protocol=dict(revision='adaptation_comparison_step3',source_physics_revision=5,fresh_test_seed=SEED,
        retired_test_ids=list(range(200,290)),mass_range_kg=[.18,.50],cube_friction_range=[.4,1.],floor_material_friction=.3,
        force_range_n=[4.,7.],duration_range_s=[.060,.100],duration_grid_s=.005,physics_dt_s=.001,model_dt_s=.005,
        coast_s=1.5,settle_s=.1,trim_after_stop_steps=5,revised_protocol_sha256=PROTOCOL_SHA256,
        test_randomization='Only after frozen settings announced. Independent property stream100, command200, duration300; no rejection.',
        reused_training='All original200 cubes and ten shoves; SHA256-verified trace symlinks.',
        state='Cube physical state15',action='Applied forceXYZ only; exactlyzero coast',
        forbidden_features=['cube_id','episode_id','step','phase','length','padding_mask','stop_time'],
        score_groups=['push','moving_coast','all_steps'])
    write_new(data/'cubes.json',{'cubes':cubes});write_new(data/'plans.json',plans)
    write_new(data/'split.json',dict(train=TRAIN_IDS,validation=VALIDATION_IDS,reused=REUSED_IDS,test=TEST_IDS,
        retired_test=list(range(200,290)),episodes_per_cube=10,train_seed=2026111509,test_seed=SEED))
    write_new(data/'protocol.json',protocol);write_new(data/'schema.json',read(STEP2/'data/schema.json'))
    for kind in ['episodes','raw']:
        (data/kind).mkdir(exist_ok=True)
        for row in old_rows:
            name=filename(row['cube_id'],row['episode_id']);source=STEP2/'data'/kind/name
            require(digest(source)==row[kind+'_sha256'],f'Reused trace changed: {source}')
            target=data/kind/name;target.symlink_to(source.resolve());training_seal[str(target.relative_to(ROOT))]=row[kind+'_sha256']
    for name in ['split.json','protocol.json','schema.json']:training_seal['data/'+name]=digest(data/name)
    write_new(results/'reused_training_records.json',old_rows)
    training_seal['results/reused_training_records.json']=digest(results/'reused_training_records.json')
    retired=retired_manifest();write_new(results/'retired_experiment_seal.json',retired)
    write_new(results/'source_seal.json',{str(p.resolve()):digest(p) for p in source_paths()})
    write_new(results/'prepared_training_seal.json',training_seal)
    preparation=dict(status='passed',prepared_unix_seconds=time.time(),protocol_sha256=PROTOCOL_SHA256,
        reused_training_cubes=200,reused_training_episodes=2000,reused_trace_files=4000,
        training_cubes_logical_sha256=logical_digest(cubes),training_plans_logical_sha256=logical_digest(plans),
        retired_artifacts_preserved=len(retired),base_checkpoints_preserved=35,
        no_fresh_properties_or_plans_generated=True,no_fresh_traces_accessed=True)
    write_new(results/'preparation.json',preparation)
    verify_prepared();print(json.dumps(preparation,indent=2),flush=True);return preparation


def require_announced_selection():
    results=ROOT/'results'
    selected=read(results/'selected_config.json');seal=read(results/'selection_seal.json')
    require('results/selected_config.json' in seal['files'],'Selection seal omits chosen setting')
    for name,sha in seal['files'].items():require(digest(ROOT/name)==sha,f'Changed validation selection artifact: {name}')
    announced=read(results/'settings_announced.json')
    require(announced['selected_config_sha256']==digest(results/'selected_config.json'),'Announcement does not bind selected setting')
    require(announced['announced_unix_seconds']>= (results/'selected_config.json').stat().st_mtime,'Announcement predates setting')
    require(announced['announced_unix_seconds']<=time.time(),'Announcement time is in future')
    return dict(selected_config_sha256=digest(results/'selected_config.json'),selection_seal_sha256=digest(results/'selection_seal.json'),
                settings_announced_sha256=digest(results/'settings_announced.json'),announced_unix_seconds=announced['announced_unix_seconds'])


def initialize_test():
    verify_prepared();selection=require_announced_selection()
    if (ROOT/'data/seal.json').exists():verify();return
    require(not (ROOT/'results/fresh_initialization.json').exists(),'Incomplete fresh initialization; do not regenerate')
    data=ROOT/'data';cubes=read(data/'cubes.json')['cubes'];plans=read(data/'plans.json')
    require([c['cube_id'] for c in cubes]==REUSED_IDS and len(plans)==2000,'Unexpected pretest data state')
    started=time.time()
    for cid in TEST_IDS:
        rng=np.random.default_rng(np.random.SeedSequence([SEED,cid,100]))
        mass=float(rng.uniform(.18,.50));mu=float(rng.uniform(.4,1.))
        cubes.append(dict(cube_id=cid,split='test',mass_kg=mass,friction=mu,combined_friction=(mu+.3)/2))
    plans.extend(make_plan(c,e) for c in TEST_IDS for e in range(10))
    replace_derived(data/'cubes.json',{'cubes':cubes});replace_derived(data/'plans.json',plans)
    write_new(ROOT/'results/fresh_initialization.json',dict(initialized_unix_seconds=started,**selection,
              fresh_seed=SEED,fresh_test_ids=TEST_IDS,no_rejection_or_resampling=True))
    write_new(data/'seal.json',{n:digest(data/n) for n in ['cubes.json','split.json','plans.json','protocol.json','schema.json']})
    verify()


def verify():
    verify_prepared();selection=require_announced_selection()
    initialized=read(ROOT/'results/fresh_initialization.json')
    require(initialized['initialized_unix_seconds']>=selection['announced_unix_seconds'],'Fresh generation preceded announcement')
    for key,value in selection.items():require(initialized[key]==value,'Selection changed after fresh initialization')
    for name,sha in read(ROOT/'data/seal.json').items():require(digest(ROOT/'data'/name)==sha,f'Changed data definition: {name}')


def collect():
    started=time.perf_counter(); initialize_test(); verify()
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
    split=read(data/'split.json')
    require(split['train']==TRAIN_IDS and split['validation']==VALIDATION_IDS and split['test']==TEST_IDS,'Wrong cube-role split')
    for name,sha in read(results/'retired_experiment_seal.json').items():
        require(digest(Path(name))==sha,f'Retired artifact changed: {name}')
    require({(p['cube_id'],p['episode_id']) for p in plans}==identities and len(plans)==2300,'Wrong plans')
    table={(p['cube_id'],p['episode_id']):p for p in plans}
    old_cubes={c['cube_id']:c for c in read(STEP2/'data/cubes.json')['cubes']}
    old_plans={(p['cube_id'],p['episode_id']):p for p in read(STEP2/'data/plans.json')}
    for c in cubes:
        cid=c['cube_id']
        if cid in REUSED_IDS: require(c==old_cubes[cid],'Training label changed')
        else:
            rng=np.random.default_rng(np.random.SeedSequence([SEED,cid,100]))
            require(c['mass_kg']==float(rng.uniform(.18,.50)) and c['friction']==float(rng.uniform(.4,1.)),'Fresh label generator mismatch')
            require(c['combined_friction']==(c['friction']+.3)/2,'Combined coefficient mismatch')
    for key,p in table.items():
        require(p==(old_plans[key] if key[0] in REUSED_IDS else make_plan(*key)),'Plan changed')
        require(not any(k in p for k in ['mass_kg','friction','combined_friction']),'Plan includes physical label')
    expected_names={filename(*key) for key in identities}
    for kind in ['episodes','raw']:
        require({p.name for p in (data/kind).glob('*.npz')}==expected_names,'Unexpected trace files')
    max_action_error=0.
    for r in rows:
        key=r['cube_id'],r['episode_id']; p=table[key]; name=filename(*key)
        for kind in ['episodes','raw']:
            path=data/kind/name; require(digest(path)==r[kind+'_sha256'],'Trace hash mismatch')
            require(path.is_symlink()==(key[0] in REUSED_IDS),'Incorrect new/reused trace storage')
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
        validation_summary=aggregate([r for r in rows if r['cube_id'] in VALIDATION_IDS]),
        reused_summary=aggregate([r for r in rows if r['cube_id'] in REUSED_IDS]),
        weight_training_ids=TRAIN_IDS,latent_validation_ids=VALIDATION_IDS,
        fresh_test_ids=TEST_IDS,retired_test_ids=list(range(200,290)),fresh_test_seed=SEED,
        maximum_force_plan_error_n=max_action_error,reused_training_bytes_unchanged=True,
        old_test_traces_excluded=True,no_motion_based_test_resampling=True,
        retired_artifacts_unchanged=len(read(results/'retired_experiment_seal.json')),
        checks=['data/source SHA256 seals','independent fresh RNG streams','train identities and bytes preserved',
                'all2300 raw/model transitions aligned','applied horizontal COM force and exact zero coast',
                'stop confirmation and five-step tail recomputed','no moving steps trimmed','finite flat sliding trajectories'])
    out=results/'full_audit.json'
    if out.exists(): require(read(out)==result,'Audit changed')
    else: write_new(out,result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','collect','audit'])
    args=parser.parse_args()
    try:
        {'prepare':prepare_training,'collect':collect,'audit':audit}[args.command]()
    except Exception as exc:
        out=ROOT/'results/STOP.json'
        if not out.exists():write_new(out,dict(stage='collection_'+args.command,error=repr(exc),stopped_unix_seconds=time.time()))
        raise


if __name__=='__main__':main()
