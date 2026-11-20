"""Independent saved-pilot checks. Never collect motion or train a model.

The PhysX contact API exports normal impulse only. Full transmitted force is
therefore checked against the free pusher's measured 3D momentum change.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, context, atol=1e-10, rtol=1e-9):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol), context)


def audit(root, expected=100):
    data, results = root / 'data', root / 'results'
    rows = read(results / 'pilot_metrics.json')
    require(len(rows) == expected, f'Expected {expected} saved episodes, found {len(rows)}')
    require(not (results / 'STOP.json').exists(), 'Collector STOP.json exists; pilot is not valid')
    for name, value in read(data / 'seal.json').items():
        require(digest(data / name) == value, f'Probe seal changed: {name}')
    for name, value in read(results / 'source_seal.json').items():
        require(digest(root.parent / name) == value, f'Original/source seal changed: {name}')
    protocol=read(data/'protocol.json')
    require(protocol['pilot_revision']==4,'Expected the requested stronger-push revision')
    require(protocol['cap_range_n']==[4.,7.],'Configured cap range differs from user request')
    require(protocol['trim_after_stop_steps']==5,'Expected five retained steps after confirmed rest')
    evaluation=protocol['planned_evaluation']
    require(evaluation['score_groups']==['push','moving_coast','all_steps'],
            'Later comparisons must retain all three requested groups')
    require(evaluation['primary_gate'].startswith('Moving-coast one-step NMSE:'),
            'Gate must be moving-coast rather than all-coast')

    # This is a code-path check, not a correlation-based assumption of no leak.
    policy = ast.parse((root / 'script.py').read_text())
    forbidden = {'mass', 'mass_kg', 'friction', 'combined_friction', 'properties',
                 'cube_id', 'episode_id', 'open', 'load', 'loadtxt', 'read_text'}
    names = {n.id for n in ast.walk(policy) if isinstance(n, ast.Name)}
    strings = {n.value for n in ast.walk(policy) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    require(not (names & forbidden or strings & forbidden), 'Policy reads properties or identity')
    imports = [alias.name for n in ast.walk(policy) if isinstance(n, ast.Import) for alias in n.names]
    require(imports == ['numpy'], f'Unexpected policy imports: {imports}')
    require(not any(isinstance(n, ast.ImportFrom) for n in ast.walk(policy)), 'Unexpected policy import-from')
    simulation = ast.parse((root / 'simulation.py').read_text())
    property_reads = [n for n in ast.walk(simulation) if isinstance(n, ast.Name) and n.id == 'properties']
    calls = [n for n in ast.walk(simulation) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'set_cube_properties']
    require(len(property_reads) == 1 and len(calls) == 1 and property_reads[0] in calls[0].args,
            'Simulator properties have a path other than the setup setter')

    # Use parsed constants for this independent check, not collector helpers.
    lock_calls = [n for n in ast.walk(simulation) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == 'set_locked_motion_axes']
    require(len(lock_calls) == 1, 'Expected one pusher motion-lock setup')
    lock_axes = ast.literal_eval(lock_calls[0].args[0])
    require(lock_axes == [False, False, False, True, True, True],
            'Full 3D impulse recovery requires all pusher translation axes free')
    gravity_calls = [n for n in ast.walk(simulation) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'set_disable_gravity']
    require(len(gravity_calls) == 1 and ast.literal_eval(gravity_calls[0].args[0]) is True,
            'Pusher gravity must be disabled for the stated momentum balance')

    split = read(data / 'split.json')
    require(split['train'] == list(range(200)) and split['test'] == list(range(200, 230)), 'Cube split mismatch')
    require(split['pilot'] == list(range(10)), 'Pilot cube identities changed')
    cubes = read(data / 'cubes.json')['cubes']
    require(len(cubes) == 230, 'Expected 230 newly sampled cube definitions')
    prior = root / 'versions/v3/data'
    require(prior.exists(), 'Previous accepted pilot must remain archived')
    require(digest(data/'cubes.json') == digest(prior/'cubes.json'), 'Pilot cube properties changed')
    require(digest(data/'split.json') == digest(prior/'split.json'), 'Pilot split changed')
    old_plans = {(p['cube_id'],p['episode_id']):p for p in read(prior/'plans.json')}
    plans = read(data / 'plans.json')
    require(len(plans) == 2300, 'Expected 2300 independent shove plans')
    plan_index = {(p['cube_id'], p['episode_id']): p for p in plans}
    require(len(plan_index) == 2300, 'Duplicate shove plans')
    for cube in cubes:
        cid = cube['cube_id']
        rng = np.random.default_rng(np.random.SeedSequence([split['seed'], cid, 100]))
        close(cube['mass_kg'], rng.uniform(.18, .50), f'Property RNG mismatch for cube {cid}', atol=0, rtol=0)
        close(cube['friction'], rng.uniform(.4, 1.), f'Property RNG mismatch for cube {cid}', atol=0, rtol=0)
        close(cube['combined_friction'], (cube['friction'] + .3) / 2,
              f'Combined friction mismatch for cube {cid}', atol=0, rtol=0)
        require(cube['split'] == ('train' if cid < 200 else 'test'), 'Registry split mismatch')
    for p in plans:
        rng = np.random.default_rng(np.random.SeedSequence([split['seed'], p['cube_id'], p['episode_id'], 200]))
        values = [rng.uniform(4., 7.), rng.uniform(-np.pi, np.pi), rng.uniform(-.006, .006),
                  *rng.uniform(-.10, .10, size=2), rng.uniform(-np.pi, np.pi)]
        saved = [p['cap_n'], p['direction_rad'], p['contact_offset_m'], *p['start_xy'], p['yaw_rad']]
        close(saved, values, 'Shove RNG must use its separate property-blind stream', atol=1e-14, rtol=0)
        old=old_plans[(p['cube_id'],p['episode_id'])]
        require({k:v for k,v in p.items() if k!='cap_n'} == {k:v for k,v in old.items() if k!='cap_n'},
                'Only the force-cap range may change in the rerun plans')
        close(p['cap_n'],4.+3.*(old['cap_n']-2.)/2.,'Force uniform quantile changed',atol=1e-14,rtol=0)
        require(4. <= p['cap_n'] <= 7.,'Cap outside requested range')

    identities = {(r['cube_id'], r['episode_id']) for r in rows}
    require(len(identities) == expected, 'Duplicate saved episode')
    require(identities == {(i, e) for i in range(10) for e in range(10)}, 'Collected beyond or short of the authorized ten cubes')
    files = {f.name for f in (data / 'episodes').glob('*.npz')}
    expected_files = {f'cube_{c:03d}_episode_{e:02d}.npz' for c, e in identities}
    require(files == expected_files, 'Extra or missing saved episode file')
    require({f.name for f in (data / 'raw').glob('*.npz')} == expected_files, 'Extra or missing raw file')

    full_cap_ratios = []
    normal_discrepancies = []
    normal_projection_errors = []
    maximum_coast_force = 0.
    randomized = []
    moving_coast = 0
    full_moving_coast = 0
    moving_all = 0
    retained_steps = 0
    coast_steps = 0
    trimmed_steps = 0
    launches = 0
    permissive_launches = 0
    maximum_distance = 0.
    episode_lengths = []
    contact_diagnostics=[]
    stopped = 0
    for row in rows:
        key = row['cube_id'], row['episode_id']
        p = plan_index[key]
        name = f'cube_{key[0]:03d}_episode_{key[1]:02d}.npz'
        for kind in ['episodes', 'raw']:
            require(digest(data / kind / name) == row[f'{kind}_sha256'], f'Changed {kind} trace: {key}')
        with np.load(data / 'episodes' / name, allow_pickle=False) as ep, np.load(data / 'raw' / name, allow_pickle=False) as raw:
            require(set(ep.files) == {'cube_id','episode_id','step','phase','state','action','next_state'}, 'Unexpected model-data field')
            require(not (set(raw.files) & forbidden), 'Property label or identity in physical raw arrays')
            n=len(ep['state'])
            require(21 <= n <= 216,'Trimmed episode must retain all push steps and five coast tail steps')
            require(ep['state'].shape == ep['next_state'].shape == (n,21),f'State shape: {key}')
            require(ep['action'].shape == (n,8),f'Action shape: {key}')
            require(np.array_equal(ep['cube_id'],np.full(n,key[0])),'Wrong cube identifier')
            require(np.array_equal(ep['episode_id'],np.full(n,key[1])),'Wrong episode identifier')
            require(np.array_equal(ep['step'],np.arange(n)),'Step sequence mismatch')
            require(np.array_equal(ep['phase'],['push']*16+['coast']*(n-16)),'Trimmed phase layout mismatch')
            require(np.array_equal(ep['state'][1:], ep['next_state'][:-1]), 'Discontinuous model transitions')
            state = raw['state']; phase = raw['phase']; commands = raw['command_force']
            dt = float(raw['physics_dt_s']); release = int(raw['release_index'])
            require(state.shape == (1981, 21) and phase.shape == (1980,), 'Raw fixed horizon changed')
            require(dt == .001 and release == 980, 'Physics timing mismatch')
            require(np.array_equal(phase, [0]*800+[1]*100+[2]*80+[3]*1000), 'Raw phase lengths changed')
            for array in [state, commands, ep['state'], ep['next_state'], ep['action']]:
                require(np.isfinite(array).all(), f'Non-finite physical value: {key}')
            selected = state[900::5].astype(np.float32)
            require(np.array_equal(ep['state'], selected[:n]) and np.array_equal(ep['next_state'], selected[1:n+1]),
                    'Model transitions do not match raw step boundaries')
            direction = np.array([np.cos(p['direction_rad']), np.sin(p['direction_rad']), 0.])
            expected_commands = np.zeros((1980,3))
            expected_commands[800:900] = .03*direction
            expected_commands[900:980] = .95*p['cap_n']*direction
            close(commands, expected_commands, 'Commands differ from the property-independent plan', atol=0, rtol=0)
            park = np.array([*p['start_xy'], .02])-1.5*direction
            expected_actions = np.zeros((n,8))
            expected_actions[:16,:3] = .95*p['cap_n']*direction
            expected_actions[:,3] = p['cap_n']; expected_actions[:16,4] = 1.
            expected_actions[:,5:] = park
            require(np.array_equal(ep['action'], expected_actions.astype(np.float32)), 'Saved actions include non-command information')
            randomized.append([p['cap_n'],p['direction_rad'],p['contact_offset_m'],*p['start_xy'],p['yaw_rad']])

            # Independently reconstruct initial pose and sphere geometry.
            yaw = p['yaw_rad']
            rotation = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            local_direction = rotation.T @ direction[:2]
            face = int(np.argmax(np.abs(local_direction)))
            normal = np.zeros(2); normal[face] = -np.sign(local_direction[face])
            local_point = .02*normal; local_point[1-face] = p['contact_offset_m']
            center = np.array([*p['start_xy'], .02])
            sphere_start = center + np.r_[rotation @ local_point,0.] + .008*np.r_[rotation @ normal,0.] -.003*direction
            close(state[0,:3],center,'Cube initial position disagrees with independent plan',atol=1e-8)
            close(state[0,15:18],sphere_start,'Pusher initial contact geometry mismatch',atol=1e-8)
            close(state[0,18:21],.005*direction,'Pusher approach velocity mismatch',atol=1e-8)
            close(state[0,3:9],np.array([np.cos(yaw),np.sin(yaw),0.,-np.sin(yaw),np.cos(yaw),0.]),
                  'Cube initial yaw mismatch',atol=3e-7)

            # Pusher translation has no constraints, gravity, damping, or other contacts.
            # The intentional velocity reset at release is not a contact impulse.
            actual_force = commands[:release] - .005*np.diff(state[:release+1,18:21],axis=0)/dt
            peak = float(np.linalg.norm(actual_force,axis=1).max())
            ratio = peak/p['cap_n']
            require(ratio <= 1.00001, f'Actual full contact force exceeds cap: {key}: {ratio}')
            full_cap_ratios.append(ratio)
            require('other_pusher_contact_count' in raw.files, 'Need explicit exclusion of non-cube pusher contacts')
            require(np.all(raw['other_pusher_contact_count'] == 0), 'Pusher contacted another body')
            require(np.all(raw['other_pusher_normal_impulse_norm'] == 0),
                    'Other pusher proximity contacts must have exactly zero normal impulse')
            close(raw['pusher_step_velocity_before'][:release],state[:release,18:21],
                  'Pre-step velocity samples do not align with raw state',atol=0,rtol=0)
            close(raw['pusher_step_velocity_before'][release:],[0,0,0],
                  'Release velocity reset or parking is inconsistent',atol=0,rtol=0)
            if 'transmitted_force_n' in raw.files:
                close(raw['transmitted_force_n'][:release],actual_force,'Saved full-force sensor mismatch',atol=1e-10)
                coast_force = float(np.linalg.norm(raw['transmitted_force_n'][release:],axis=1).max())
            elif 'contact_force_vector_n' in raw.files:
                close(raw['contact_force_vector_n'][:release],actual_force,'Saved full-force sensor mismatch',atol=1e-10)
                coast_force = float(np.linalg.norm(raw['contact_force_vector_n'][release:],axis=1).max())
            else:
                raise AssertionError('Raw trace needs the full transmitted force vector')
            close(raw['contact_force_n'],np.linalg.norm(raw['transmitted_force_n'],axis=1),
                  'Saved force magnitude differs from full vector norm',atol=1e-10)
            maximum_coast_force = max(maximum_coast_force,coast_force)
            require(coast_force < 1e-8, 'Nonzero force after release')
            close(row['max_contact_force_n'],peak,'Reported peak is not actual full force',atol=1e-9)
            close(row['cap_ratio'],ratio,'Reported cap ratio mismatch',atol=1e-9)

            # Normal-only API export must agree in its own normal direction.
            api_impulse = raw['reported_normal_impulse'][:release]
            norm = np.linalg.norm(api_impulse,axis=1)
            mask = norm > 1e-8
            require(np.max(raw['contact_point_count']) <= 1, 'Normal projection check assumes one sphere-face point')
            close(raw['reported_normal_force_n'],np.linalg.norm(raw['reported_normal_impulse'],axis=1)/dt,
                  'Normal-only API scalar aggregation mismatch',atol=1e-10)
            unit = api_impulse[mask]/norm[mask,None]
            projection = np.abs(np.sum(actual_force[mask]*dt*unit,axis=1))
            error = np.abs(projection-norm[mask])
            normal_projection_errors.append(float(error.max()) if len(error) else 0.)
            require(not len(error) or error.max() < 2e-7, f'Normal impulse projection inconsistent: {key}')
            normal_discrepancies.append(float(np.max(np.linalg.norm(actual_force,axis=1)-norm/dt)))
            require(np.max(np.linalg.norm(raw['reported_normal_impulse'][release:],axis=1)) < 1e-12,'API reports coast contact')
            require(np.all(raw['contact_point_count'][release:] == 0),'Pusher remains near cube after withdrawal')

            resting = (np.linalg.norm(state[:,9:12],axis=1)<.001)&(np.linalg.norm(state[:,12:15],axis=1)<.05)
            windows = np.convolve(resting[release+1:].astype(int),np.ones(100,dtype=int),mode='valid')
            candidates = np.flatnonzero(windows == 100)
            require(len(candidates)>0, 'No confirmed rest within fixed coast')
            stop = release+1+int(candidates[0])
            if stop == release+1 and resting[release]:stop = release
            require(row['stopped'] and row['stop_raw_index'] == stop, 'Reported rest onset mismatch')
            stopped += 1
            close(row['time_to_stop_s'],(stop-release)*dt,'Time to stop includes confirmation or wrong index')
            close(row['launch_speed_m_s'],np.linalg.norm(state[release,9:11]),'Launch speed mismatch')
            close(row['stopping_distance_m'],np.linalg.norm(state[stop,:2]-state[release,:2]),'Stopping distance mismatch')
            close(row['stopping_xy_m'],state[stop,:2],'Stopping position mismatch')
            close(row['terminal_speed_m_s'],np.linalg.norm(state[-1,9:12]),'Terminal speed mismatch')
            close(row['terminal_angular_speed_rad_s'],np.linalg.norm(state[-1,12:15]),'Terminal angular speed mismatch')
            expected_keep=min(216,int(np.ceil((stop-900)/5))+5)
            require(n == expected_keep,f'Trim endpoint differs from first confirmed stop plus five steps: {key}')
            require((900+n*5)-stop >= 25 and (900+n*5)-stop < 30,
                    'Trim must retain 25 to29ms beyond stop onset')
            observed_moving=(np.linalg.norm(ep['state'][:,9:12],axis=1)>.001)|(np.linalg.norm(ep['state'][:,12:15],axis=1)>.05)
            moving_coast += int(observed_moving[16:].sum())
            moving_all += int(observed_moving.sum())
            full=(np.linalg.norm(selected[:-1,9:12],axis=1)>.001)|(np.linalg.norm(selected[:-1,12:15],axis=1)>.05)
            full_moving_coast += int(full[16:].sum())
            require(not np.any(full[n:]),'Trim removed a moving transition beyond the confirmed rest tail')
            launches += int(row['launch_speed_m_s']>.02)
            permissive_launches += int(row['launch_speed_m_s']>.001)
            maximum_distance=max(maximum_distance,float(row['stopping_distance_m']))
            push_speed=np.linalg.norm(state[900:981,9:11],axis=1)
            push_force=np.linalg.norm(actual_force[900:980],axis=1)
            forceful=np.flatnonzero(push_force>.1)
            contact_diagnostics.append(dict(launched_at_release=bool(row['launch_speed_m_s']>.02),
                                            moved_during_push=bool(push_speed.max()>.02),
                                            peak_push_speed_m_s=float(push_speed.max()),
                                            mean_push_force_n=float(push_force.mean()),
                                            last_forceful_push_step=int(forceful[-1]) if len(forceful) else -1,
                                            forceful_push_steps=int(len(forceful))))
            retained_steps += n;coast_steps += n-16;trimmed_steps += 216-n;episode_lengths.append(n)
            for field,expected_value in [('untrimmed_steps',216),('retained_steps',n),('trimmed_steps',216-n),
                                         ('retained_coast_steps',n-16),
                                         ('retained_moving_coast_steps',int(observed_moving[16:].sum())),
                                         ('untrimmed_coast_steps',200),
                                         ('untrimmed_moving_coast_steps',int(full[16:].sum()))]:
                require(row[field]==expected_value,f'Reported {field} mismatch')

    randomized = np.asarray(randomized)
    distinct = np.array([len(np.unique(randomized[:,i])) for i in range(6)])
    require(np.all(distinct == expected),'Every collected shove must randomize all six nuisance coordinates')
    require(retained_steps+trimmed_steps==expected*216,'Retained and trimmed counts do not sum to raw transition count')
    contact_groups={}
    for launched,name in [(False,'not_launched_at_release'),(True,'launched_at_release')]:
        group=[d for d in contact_diagnostics if d['launched_at_release']==launched]
        contact_groups[name]=dict(episodes=len(group),moved_during_push=sum(d['moved_during_push'] for d in group),
                                 median_peak_push_speed_m_s=float(np.median([d['peak_push_speed_m_s'] for d in group])) if group else None,
                                 median_mean_push_force_n=float(np.median([d['mean_push_force_n'] for d in group])) if group else None,
                                 median_last_forceful_push_time_ms=float(np.median([d['last_forceful_push_step']+1 for d in group])) if group else None,
                                 median_forceful_push_steps=float(np.median([d['forceful_push_steps'] for d in group])) if group else None)
    return dict(status='passed',cubes=10,episodes=expected,transitions=retained_steps,
                untrimmed_transitions=expected*216,trimmed_transitions=trimmed_steps,
                trimmed_fraction=trimmed_steps/(expected*216),
                episode_length_min=min(episode_lengths),episode_length_max=max(episode_lengths),
                push_transitions=expected*16,coast_transitions=coast_steps,moving_coast_transitions=moving_coast,
                moving_coast_fraction=moving_coast/coast_steps,
                untrimmed_coast_transitions=expected*200,untrimmed_moving_coast_transitions=full_moving_coast,
                untrimmed_moving_coast_fraction=full_moving_coast/(expected*200),
                moving_all_transitions=moving_all,launch_threshold_m_s=.02,launches=launches,
                launch_fraction=launches/expected,launch_target_passed=launches/expected>=.9,
                launches_over_1mm_s=permissive_launches,maximum_coast_distance_m=maximum_distance,
                contact_loss_diagnostic=dict(forceful_threshold_n=.1,push_duration_ms=80,
                                             moved_during_push=sum(d['moved_during_push'] for d in contact_diagnostics),
                                             release_launches=launches,groups=contact_groups,
                                             interpretation='Movement during push is not the release-launch criterion. A shove may move early and stop after losing contact before scheduled release.'),
                all_stopped=stopped==expected,maximum_actual_contact_force_over_cap=max(full_cap_ratios),
                maximum_coast_contact_force_n=maximum_coast_force,
                maximum_normal_projection_impulse_error_Ns=max(normal_projection_errors),
                maximum_normal_only_force_underestimate_n=max(normal_discrepancies),
                unique_randomized_values=dict(zip(['cap','direction','contact_offset','start_x','start_y','yaw'],distinct.tolist())),
                checks=['original and probe source/data seals','only authorized ten cubes collected',
                        'separate property and per-shove RNG streams','physical properties used only by setup setter',
                        'state/action whitelist and raw-to-model transition alignment','same cube identities and non-cap nuisance settings as v3',
                        'caps uniformly remapped from2–4N to4–7N using unchanged random quantiles',
                        'all push steps retained; coast trimmed five steps after independently confirmed rest',
                        'trim removes only stationary steps; reported counts independently recomputed',
                        'actions exactly reconstructible from property-blind plans','full 3D transmitted force from pusher momentum',
                        'zero impulse from non-cube pusher proximity contacts','normal-only API projection agrees with momentum sensor',
                        'cap compliance and zero force after release','independent stop-window and endpoint metrics'],
                limitation='This is a data/physics audit, not the Oracle-vs-Plain prediction gate. Trimmed file length is outcome-dependent and must never be a model feature or label-revealing mask. No model training or added simulation.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--expected-episodes',type=int,default=100)
    args=parser.parse_args()
    root=args.root.resolve()
    try:
        result=audit(root,args.expected_episodes)
    except (AssertionError,KeyError,FileNotFoundError) as error:
        result=dict(status='failed',reason=str(error))
        (root/'results/pilot_audit.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        raise
    (root/'results/pilot_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
