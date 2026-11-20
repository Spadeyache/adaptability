"""Independent direct-force data audit. No simulation or model fitting."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise AssertionError(message)
def close(actual,expected,message,atol=1e-9,rtol=1e-7):
    require(np.allclose(actual,expected,atol=atol,rtol=rtol),message)


def audit(root,stage='pilot'):
    root=Path(root);data=root/'data';results=root/'results'
    expected=100 if stage=='pilot' else 2300
    path=results/('pilot_metrics.json' if stage=='pilot' else 'metrics.json')
    rows=read(path)
    require(len(rows)==expected,f'Expected{expected}episodes; found{len(rows)}')
    require(not (results/'STOP.json').exists(),'Collector STOP.json exists')
    for name,value in read(data/'seal.json').items():require(digest(data/name)==value,f'Changed data seal:{name}')
    for name,value in read(results/'source_seal.json').items():require(digest(root.parent/name)==value,f'Changed source/original seal:{name}')
    protocol=read(data/'protocol.json');split=read(data/'split.json')
    require(protocol['pilot_revision']==5,'Expected direct-force revision5')
    require(protocol['planned_evaluation']['score_groups']==['push','moving_coast','all_steps'],'Three reporting groups changed')
    require(''.join(protocol['planned_evaluation']['primary_gate'].split()).startswith('Moving-coastone-stepNMSE:'),'Wrong primary gate')
    require(protocol['force_range_n']==[4.,7.],'Force range changed')
    require(protocol['trim_after_stop_steps']==5,'Trim rule changed')
    require(split['train']==list(range(200)) and split['test']==list(range(200,230)),'Cube split changed')
    require(split['pilot']==list(range(10)),'Pilot identities changed')
    prior=root/'versions/v4/data'
    for name in ['cubes.json','split.json']:
        require(digest(data/name)==digest(prior/name),f'Existing identity metadata changed:{name}')
    cubes=read(data/'cubes.json')['cubes'];plans=read(data/'plans.json')
    require(len(cubes)==230 and len(plans)==2300,'Expected230cube definitions and2300plans')
    table={(p['cube_id'],p['episode_id']):p for p in plans};old={(p['cube_id'],p['episode_id']):p for p in read(prior/'plans.json')}
    require(len(table)==2300,'Duplicate plan identity')
    for cube in cubes:
        cid=cube['cube_id'];rng=np.random.default_rng(np.random.SeedSequence([split['seed'],cid,100]))
        close(cube['mass_kg'],rng.uniform(.18,.50),'Mass RNG changed',atol=0,rtol=0)
        close(cube['friction'],rng.uniform(.4,1.),'Friction RNG changed',atol=0,rtol=0)
        close(cube['combined_friction'],(cube['friction']+.3)/2,'Combined coefficient mismatch',atol=0,rtol=0)
    for key,p in table.items():
        previous=old[key]
        close(p['force_n'],previous['cap_n'],'Force draw changed independently of requested actuator replacement',atol=0,rtol=0)
        for name in ['direction_rad','start_xy','yaw_rad']:require(p[name]==previous[name],f'Nuisance plan changed:{name}')
        rng=np.random.default_rng(np.random.SeedSequence([split['seed'],key[0],key[1],300]))
        duration_steps=int(rng.integers(12,21))
        require(p['duration_steps']==duration_steps,'Command duration RNG mismatch')
        close(p['duration_s'],duration_steps*.005,'Command duration grid mismatch',atol=0,rtol=0)
        require(4<=p['force_n']<=7 and 12<=duration_steps<=20,'Command outside requested range')
        require('contact_offset_m' not in p and 'cap_n' not in p,'Obsolete pusher settings remain in direct-force plans')

    # Audit physical-label routing and actual native actuation call from source.
    policy=ast.parse((root/'script.py').read_text())
    forbidden={'mass','mass_kg','friction','combined_friction','properties','cube_id','episode_id','open','load','read_text'}
    names={n.id for n in ast.walk(policy) if isinstance(n,ast.Name)}
    strings={n.value for n in ast.walk(policy) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    require(not (names&forbidden or strings&forbidden),'Command policy reads hidden labels or identities')
    simulation=ast.parse((root/'simulation.py').read_text())
    properties=[n for n in ast.walk(simulation) if isinstance(n,ast.Name) and n.id=='properties']
    setters=[n for n in ast.walk(simulation) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='set_cube_properties']
    require(len(properties)==1 and len(setters)==1 and properties[0] in setters[0].args,'Properties used outside physical setup setter')
    calls=[n for n in ast.walk(simulation) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    require(not any(c.func.attr in ['add_sphere_collision','add_force_at_point','set_locked_motion_axes'] for c in calls),'Pusher/off-center actuator or locked cube remains')
    force_calls=[c for c in calls if c.func.attr=='add_force_torque']
    require(len(force_calls)==1,'Expected one direct force application path')
    require(ast.literal_eval(force_calls[0].args[1])==[0,0,0],'Direct applied torque must be zero')
    require(isinstance(force_calls[0].func.value,ast.Name) and force_calls[0].func.value.id=='body','Force must be applied to cube body')
    builders=[c for c in calls if c.func.attr=='build']
    require(len(builders)==1,'Expected cube only, with no pusher dynamic actor')

    identities={(r['cube_id'],r['episode_id']) for r in rows}
    expected_ids={(c,e) for c in range(10 if stage=='pilot' else 230) for e in range(10)}
    require(identities==expected_ids and len(rows)==len(identities),'Wrong or duplicate collected identities')
    expected_files={f'cube_{c:03d}_episode_{e:02d}.npz' for c,e in expected_ids}
    for kind in ['episodes','raw']:
        actual={p.name for p in (data/kind).glob('*.npz')}
        require(expected_files<=actual,'Missing saved episode')
        if stage=='full':require(actual==expected_files,'Unexpected saved episode')
        elif len(actual)>100:
            require(read(results/'pilot_gate.json')['passed'],'Extra cubes collected before pilot gatepassed')
    dt=.001;stride=5
    coast_raw_steps=round(float(protocol['coast_s'])/dt)
    retained=removed=coast_count=moving_count=push_count=full_coast_count=full_moving_count=launches=loose_launches=0
    max_distance=max_stop_time=max_force=max_coast_force=max_tilt=0.
    durations=[];episode_lengths=[];force_errors=[]
    all_stopped=True
    for row in rows:
        key=row['cube_id'],row['episode_id'];plan=table[key]
        name=f'cube_{key[0]:03d}_episode_{key[1]:02d}.npz'
        for kind in ['episodes','raw']:require(digest(data/kind/name)==row[f'{kind}_sha256'],f'Changed trace:{kind}/{name}')
        with np.load(data/'episodes'/name,allow_pickle=False) as ep,np.load(data/'raw'/name,allow_pickle=False) as raw:
            require(set(ep.files)=={'cube_id','episode_id','step','phase','state','action','next_state'},'Unexpected model fields')
            require(not (set(raw.files)&forbidden),'Hidden label in raw diagnostic fields')
            state=raw['state'];actions=raw['action'];phase=raw['phase'];release=int(raw['release_index'])
            raw_n=release+coast_raw_steps;n=len(ep['step']);push_n=plan['duration_steps']
            require(float(raw['physics_dt_s'])==dt and release==push_n*stride,'Raw pulse timing mismatch')
            require(state.shape==(raw_n+1,15) and actions.shape==(raw_n,3),'Unexpected raw shapes')
            require(np.array_equal(phase,[0]*release+[1]*coast_raw_steps),'Raw phases mismatch')
            require(ep['state'].shape==ep['next_state'].shape==(n,15) and ep['action'].shape==(n,3),'Model input dimension mismatch')
            require(np.array_equal(ep['cube_id'],np.full(n,key[0])) and np.array_equal(ep['episode_id'],np.full(n,key[1])),'Episode IDs mismatch')
            require(np.array_equal(ep['step'],np.arange(n)),'Noncontiguous step index')
            require(np.array_equal(ep['phase'],['push']*push_n+['coast']*(n-push_n)),'Model phases mismatch')
            require(np.array_equal(ep['state'][1:],ep['next_state'][:-1]),'State transition discontinuity')
            selected=state[::stride].astype(np.float32)
            require(np.array_equal(ep['state'],selected[:n]) and np.array_equal(ep['next_state'],selected[1:n+1]),'Raw/model state alignment mismatch')
            direction=np.array([np.cos(plan['direction_rad']),np.sin(plan['direction_rad']),0.])
            expected_action=np.zeros((raw_n,3));expected_action[:release]=plan['force_n']*direction
            close(actions,expected_action,'Applied force differs from independent plan',atol=1e-7,rtol=1e-7)
            require(np.all(actions[release:]==0),'Nonzero force during coast')
            require(np.all(actions[:,2]==0),'External force is not horizontal')
            require(np.array_equal(ep['action'],actions[::stride][:n].astype(np.float32)),'Saved action is not applied force vector')
            require(all(np.isfinite(a).all() for a in [state,actions,ep['state'],ep['next_state']]),'Non-finite state/action')
            close(state[0,:2],plan['start_xy'],'Initial XY changed',atol=1e-6)
            yaw=plan['yaw_rad']
            close(state[0,3:9],[np.cos(yaw),np.sin(yaw),0.,-np.sin(yaw),np.cos(yaw),0.],'Initial yaw changed',atol=2e-5)
            require(np.linalg.norm(state[0,9:12])<.001 and np.linalg.norm(state[0,12:15])<.05,'Cube not settled at push start')
            resting=(np.linalg.norm(state[:,9:12],axis=1)<.001)&(np.linalg.norm(state[:,12:15],axis=1)<.05)
            windows=np.convolve(resting[release+1:].astype(int),np.ones(100,dtype=int),mode='valid')
            candidates=np.flatnonzero(windows==100)
            require(candidates.size>0,'No confirmed stop within raw horizon')
            stop=release+1+int(candidates[0])
            if stop==release+1 and resting[release]:stop=release
            require(row['stopped'] and row['stop_raw_index']==stop,'Recorded stop onset mismatch')
            speed=float(np.linalg.norm(state[release,9:11]));distance=float(np.linalg.norm(state[stop,:2]-state[release,:2]))
            close(row['launch_speed_m_s'],speed,'Launch speed mismatch')
            close(row['stopping_distance_m'],distance,'Stopping distance mismatch')
            close(row['stopping_xy_m'],state[stop,:2],'Stop XY mismatch')
            close(row['time_to_stop_s'],(stop-release)*dt,'Stop timing mismatch')
            keep=min(len(selected)-1,int(np.ceil(stop/stride))+5)
            require(n==keep,'Trim does not retain five steps after stop onset')
            require(25<=n*stride-stop<30,'Trimmed tail does not span25–29ms after rest')
            moving=(np.linalg.norm(ep['state'][:,9:12],axis=1)>.001)|(np.linalg.norm(ep['state'][:,12:15],axis=1)>.05)
            fullmoving=(np.linalg.norm(selected[:-1,9:12],axis=1)>.001)|(np.linalg.norm(selected[:-1,12:15],axis=1)>.05)
            require(not np.any(fullmoving[n:]),'Trim removed moving steps')
            full_n=len(selected)-1
            for field,value in [('retained_steps',n),('untrimmed_steps',full_n),('trimmed_steps',full_n-n),
                                ('retained_coast_steps',n-push_n),('retained_moving_coast_steps',int(moving[push_n:].sum()))]:
                require(row[field]==value,f'Trim metric mismatch:{field}')
            normals=np.cross(state[:,3:6],state[:,6:9]);tilt=np.degrees(np.arccos(np.clip(normals[:,2],-1,1)))
            require(tilt.max()<5 and .019<state[:,2].min()<=state[:,2].max()<.021,'Cube left flat sliding regime')
            retained+=n;removed+=full_n-n;push_count+=push_n;coast_count+=n-push_n
            moving_count+=int(moving[push_n:].sum());full_coast_count+=full_n-push_n;full_moving_count+=int(fullmoving[push_n:].sum())
            launches+=int(speed>.02);loose_launches+=int(speed>.001)
            max_distance=max(max_distance,distance);max_stop_time=max(max_stop_time,(stop-release)*dt)
            max_force=max(max_force,float(np.linalg.norm(actions,axis=1).max()))
            max_coast_force=max(max_coast_force,float(np.linalg.norm(actions[release:],axis=1).max()))
            max_tilt=max(max_tilt,float(tilt.max()));durations.append(push_n);episode_lengths.append(n)
            force_errors.append(float(np.abs(actions-expected_action).max()))
    passed=launches/expected>=.9 and max_distance<1.
    if stage=='pilot':
        gate=read(results/'pilot_gate.json')
        require(gate['passed']==passed,'Collector pilot gate disagrees with independent audit')
    else:
        require(read(results/'pilot_gate.json')['passed'],'Full collection without passing pilot')
        require(read(results/'pilot_audit.json')['status']=='passed','Full collection without independently audited pilot')
    authorization_gate=bool(read(results/'pilot_gate.json')['passed'])
    long_coasts=[dict(cube_id=r['cube_id'],episode_id=r['episode_id'],
                     coast_distance_m=r['stopping_distance_m'],raw_sha256=r['raw_sha256'])
                 for r in rows if r['stopping_distance_m']>=1.]
    maximum_row=max(rows,key=lambda r:r['stopping_distance_m'])
    return dict(status='passed',stage=stage,cubes=len({r['cube_id'] for r in rows}),episodes=expected,
                transitions=retained,untrimmed_transitions=retained+removed,trimmed_transitions=removed,
                push_transitions=push_count,coast_transitions=coast_count,moving_coast_transitions=moving_count,
                moving_coast_fraction=moving_count/coast_count,untrimmed_coast_transitions=full_coast_count,
                untrimmed_moving_coast_transitions=full_moving_count,untrimmed_moving_coast_fraction=full_moving_count/full_coast_count,
                launches=launches,launch_fraction=launches/expected,launches_over_1mm_s=loose_launches,
                launch_threshold_m_s=.02,launch_target_passed=launches/expected>=.9,
                maximum_coast_distance_m=max_distance,maximum_time_to_stop_s=max_stop_time,
                maximum_applied_force_n=max_force,maximum_coast_force_n=max_coast_force,
                maximum_force_plan_error_n=max(force_errors),maximum_tilt_deg=max_tilt,
                all_stopped=all_stopped,episode_length_min=min(episode_lengths),episode_length_max=max(episode_lengths),
                push_duration_steps_histogram={str(i):durations.count(i) for i in range(12,21)},
                conditional_collection_gate_passed=authorization_gate,
                pilot_authorization_gate_passed=authorization_gate,
                stage_meets_pilot_thresholds=passed,
                full_dataset_meets_pilot_thresholds=passed if stage=='full' else None,
                continuation_scope='The user condition is evaluated on the ten-cube pilot. Full-data outliers are reported and retained; they do not trigger retrospective resampling or undo the recorded pilot authorization.',
                coast_distance_at_least_1m_count=len(long_coasts),
                coast_distance_at_least_1m_train_count=sum(r['cube_id']<200 for r in long_coasts),
                coast_distance_at_least_1m_test_count=sum(r['cube_id']>=200 for r in long_coasts),
                coast_distance_at_least_1m_episodes=long_coasts,
                maximum_coast_distance_episode=dict(cube_id=maximum_row['cube_id'],episode_id=maximum_row['episode_id'],raw_sha256=maximum_row['raw_sha256']),
                checks=['original and active source/data seals','same cube identities and nuisance positions/yaws/directions',
                        'independent discrete-uniform commanded duration stream','cube-only15D state and applied-force-only3D action',
                        'no pusher, direct native cube force withzero applied torque atCOM','hidden labels only physical setup setter',
                        'applied force schedule matches plan, exactlyzero during coast','raw/model state/action alignment',
                        'stop onset independently recomputed','five-step stationary tail and no moving step removed',
                        'launch and distance conditional gate independently recomputed'],
                limitation='Direct external force is an idealized actuator. Length, IDs, true stop timing and masks are forbidden model features. This audit does not evaluate prediction accuracy.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--stage',choices=['pilot','full'],default='pilot');args=parser.parse_args();root=args.root.resolve()
    out=root/'results'/('pilot_audit.json' if args.stage=='pilot' else 'full_audit.json')
    try:result=audit(root,args.stage)
    except (AssertionError,KeyError,FileNotFoundError) as e:
        out.write_text(json.dumps(dict(status='failed',stage=args.stage,reason=str(e)),indent=2)+'\n');raise
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
