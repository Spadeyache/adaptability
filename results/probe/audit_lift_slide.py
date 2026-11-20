"""Read-only inventory; the only output is inventory_lift_slide.json beside this script."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/lift_slide_belief'

MEANING = {
    'cube_id':'Persistent cube identifier, bookkeeping only; same IDs repeat across related variants.',
    'task':'Task code: 0 lift, 1 slide/slow_slide, 2 shove where present; bookkeeping only.',
    'episode_id':'Zero-based episode identifier within cube and task; bookkeeping only.',
    'step':'Zero-based issued-control transition index within episode.',
    'state':'Pre-action observed state43; ordered components are in state43_schema.',
    'next_state':'Observed state43 after one control action (five PhysX substeps).',
    'action':'Issued normalized translation/opening/strength command5, extended by Fx,Fy,force-mode for command8; see action_schemas.',
    'true_mass':'True simulator cube mass, kg. Privileged checking/target label, excluded from ordinary state/action inputs.',
    'true_friction':'True simulator cube friction coefficient. Privileged checking/target label, excluded from ordinary state/action inputs.',
    'cube_moving':'Pre-action cube linear speed exceeds 0.01 m/s.',
    'phase':'0 approach/setup, 1 active lift/push/force pulse, 2 release/coast for shoves.',
    'contact_force_substeps_n':'Five per-control-step, three-axis robot-to-cube reported normal-contact force vectors in N; tangential friction impulses excluded.',
    'contact_impulse_ns':'Sum of the five reported contact force vectors divided by100Hz, in N s.',
    'contact_force_max_n':'Largest horizontal norm among five reported normal-contact force vectors, N.',
    'force_start_step':'First force-pulse control step, or -1 for nonshove tasks; repeated episode metadata.',
    'shove_release_step':'First step after the six-command force pulse, or -1; repeated metadata.',
    'stop_step':'Inclusive final transition index used for endpoint scoring; fixed horizon minus1.',
    'observed_stopped':'Terminal half-second satisfies cube linear/angular speed and no robot-contact thresholds; false on nonshove tasks.',
    'touched':'Approach reached the configured reported contact threshold; repeated episode metadata.',
    'reset_seed':'Simulator reset seed, not a physical model input.',
    'motion_direction':'Sampled unit horizontal motion direction, two world-axis components.',
    'lift_speed_m_s':'Sampled scripted lift speed, m/s; repeated episode parameter.',
    'lift_height_m':'Sampled scripted lift height, m; repeated episode parameter.',
    'slide_speed_m_s':'Sampled scripted slow-push speed, m/s; repeated episode parameter.',
    'relative_position':'Cube center minus TCP position, rotated into the TCP frame, m.',
    'relative_velocity':'Causal backward difference of TCP-frame relative_position at20Hz; reset sample is zero, m/s.',
    'relative_velocity_instantaneous':'Diagnostic native instantaneous TCP-frame relative velocity including angular-frame correction, m/s; not part of projection51.',
    'finger_position_error':'Two finger drive-target minus measured joint-position errors, m.',
    'tcp_pose':'TCP world position3 plus quaternion(w,x,y,z)4; m and unitless.',
    'tcp_linear_velocity':'World TCP linear velocity inferred from local Jacobian times joint velocity, m/s.',
    'tcp_angular_velocity':'World TCP angular velocity inferred from local Jacobian times joint velocity, rad/s.',
    'finger_target_position':'Two per-finger commanded drive target positions, m.',
    'finger_contact_normal_n':'Per-finger reported contact force projected on signed local closing axis, clamped nonnegative, N.',
    'finger_drive_cap_n':'Two realized finger drive force limits at the state measurement, N.',
    'issued_grip_force_cap_n':'Two realized drive limits after issuing this row action, N; actuator cap, not guaranteed contact normal.',
    'grasp_acquired':'Episode diagnostic: acquired airborne bilateral grasp under the declared stable-gap criterion.',
    'slip_after_acquisition':'Episode diagnostic: translational slip after acquisition with declared rotation filter.',
    'rotating_loss_after_acquisition':'Episode diagnostic: grasp loss with rotation above the declared filter.',
    'acquisition_step':'First transition completing three consecutive acquired-grasp samples, or -1.',
    'slip_step':'First post-acquisition downward-gap slip event, or -1.',
    'control_step':'20Hz control index for each100Hz diagnostic sample.',
    'physics_substep':'Native substep index0..4 within control action.',
    'force_mode':'Whether the robot is executing the Cartesian force mode.',
    'command':'Issued command8 copied into each100Hz diagnostic sample.',
    'cartesian_wrench':'Commanded TCP force3 and torque3, N and N m, at a native substep.',
    'torque_components':'Four7-joint contributions: horizontal force, vertical feedback, orientation feedback, passive compensation; N m.',
    'applied_generalized_force':'Nine applied generalized arm/finger forces; N m for arm and N for fingers.',
    'before_qvel':'Nine joint velocities before native substep, rad/s or m/s.',
    'before_tcp_pose':'World TCP position+quaternion before native substep.',
    'before_cube_velocity':'World cube linear velocity before native substep, m/s.',
    'after_cube_velocity':'World cube linear velocity after native substep, m/s.',
    'robot_to_cube_force':'Sum of three robot-link reported normal-contact vectors, N.',
    'robot_to_cube_force_by_link':'Three robot-link three-axis reported normal-contact vectors, N.',
    'table_to_cube_force':'Table/cube reported normal-contact vector, N.',
    'arm_drive_stiffness':'Seven position-drive stiffness settings.',
    'arm_drive_damping':'Seven position-drive damping settings.',
    'arm_drive_force_limit':'Seven joint-drive torque limits, N m.',
    'finger_drive_force_limit':'Two finger drive-force limits, N.',
    'candidate_index':'Index of the fixed13-force candidate grid; scalar metadata in trajectory archives.',
    'selected_force_n':'Candidate horizontal force magnitude, N; scalar metadata in trajectory archives.',
    'seed':'World-model optimizer seed in derived result tables, not simulator step identity.',
    'condition':'Belief/planning comparison condition label.',
    'force_n':'Chosen or grid candidate horizontal force magnitude, N.',
    'planner_status':'Finite predicted-candidate status or explicit fallback status.',
    'target_xy':'Fixed target world XY coordinate, m.',
    'target_distance_m':'Predeclared target distance along sampled shove direction, m.',
    'tolerance_m':'Declared landing-success distance tolerance, m.',
    'predicted_endpoint_xy':'World-model predicted terminal cube XY coordinate, m; derived prediction.',
    'predicted_target_error_m':'Predicted terminal distance to target, m; derived prediction.',
    'predicted_final_speed_m_s':'Predicted terminal cube speed, m/s.',
    'valid_candidates':'Number of candidates with finite predicted rollouts.',
    'candidate_target_errors_m':'Predicted endpoint errors for all13 fixed force candidates, m.',
    'candidate_divergence_steps':'First divergence step per candidate, -1 if absent.',
    'actual_endpoint_xy':'Measured future endpoint for selected executed candidate, m; evaluation outcome, not pre-action feature.',
    'actual_stopped':'Observed terminal rest for selected executed candidate.',
    'actual_target_error_m':'Measured future target error, m; privileged evaluation outcome.',
    'target_success':'Observed rest plus actual target error within tolerance; evaluation outcome.',
    'origin_xy':'Measured cube XY at force onset, m.',
    'direction':'Horizontal unit direction for signed-distance diagnostics.',
    'predicted_signed_distance_mm':'Predicted endpoint displacement along direction, mm.',
    'actual_signed_distance_mm':'Measured endpoint displacement along direction, mm; future evaluation outcome.',
    'full_state_mse':'Normalized full-state one-step squared-error aggregate.',
    'cube_xy_mse':'Squared cube XY prediction error aggregate as defined by rollout diagnostic producer.',
    'cube_velocity_mse':'Squared cube velocity prediction error aggregate as defined by producer.',
    'cube_xy_rmse_mm':'Physical cube XY root-mean-square error, mm.',
    'cube_velocity_rmse_mm_s':'Physical cube velocity root-mean-square error, mm/s.',
    'moving_steps':'Number of eligible moving transitions in a derived score.',
    'donor_cube_id':'Wrong-belief donor cube bookkeeping.',
    'prior_mean':'Mean of learned training-cube latent vectors.',
    'prior_sd':'Standard deviation of learned training-cube latent vectors.',
    'linear_decoder':'5-by2 linear decoder fitted with true training mass/friction targets; privileged supervised diagnostic parameter.',
    'z_after_lift':'Inferred4D latent after each of five lift support episodes; learned artifact, not a raw measurement.',
    'z_lifts_only':'Inferred4D latent/prefixes from observed lift support episodes.',
    'z_slides_only':'Inferred4D latent/prefixes from observed slide support episodes.',
    'z_both':'Inferred4D latent/prefixes from observed lift+slide support episodes.',
    'z_shove_ceiling':'Inferred4D latent fitted to other same-cube shove episodes; evaluation ceiling artifact.',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def schema(path):
    result={}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                version=np.lib.format.read_magic(stream)
                shape,order,dtype=np.lib.format._read_array_header(stream,version)
            key=name.removesuffix('.npy')
            meaning=MEANING.get(key)
            if meaning is None and key.startswith('next_'):
                meaning='Post-action version of '+key[5:]+': '+MEANING[key[5:]]
            if meaning is None: raise KeyError(f'Unclassified field {key}')
            result[key]=dict(shape=list(shape),dtype=str(dtype),meaning=meaning)
    return result


def correlation(values):
    values=np.asarray(list(values),float).reshape(-1,2)
    reason=None
    if len(values)<2: reason='Fewer than two distinct physical cube/configuration records.'
    elif np.std(values[:,0])==0 or np.std(values[:,1])==0: reason='At least one physical label is constant.'
    return dict(n_unique_physical_records=len(values),pearson=None if reason else float(np.corrcoef(values.T)[0,1]),
                undefined_reason=reason,mass_kg_range=[float(values[:,0].min()),float(values[:,0].max())] if len(values) else None,
                friction_range=[float(values[:,1].min()),float(values[:,1].max())] if len(values) else None)


def classify(path):
    p=path.relative_to(BASE).as_posix()
    for prefix,name in [('nov13/data/','nov13_main'),('three_task/data/','three_task_main'),
        ('slip_probe/paired/fixed/data/','paired_fixed'),('slip_probe/paired/random/data/','paired_random'),
        ('slip_probe/smoke_pair/fixed/data/','smoke_fixed'),('slip_probe/smoke_pair/random/data/','smoke_random'),
        ('slip_probe/pilot/','pilot_constant_cap'),('slip_probe/pilot_step45/','pilot_staged_cap'),
        ('three_task/study/target_control/trajectories/','target_candidate_cache'),
        ('three_task/diagnostics/oracle/trajectories/','oracle_complement_cache'),
        ('three_task/diagnostics/force_cap/','force_cap_native_traces')]:
        if p.startswith(prefix): return name
    if p.startswith('nov13/study/'): return 'nov13_belief_banks'
    if p.startswith('three_task/study/seed_'): return 'three_task_belief_banks'
    if p.startswith('slip_probe/paired/study/fixed_grip/'): return 'paired_fixed_belief_banks'
    if p.startswith('slip_probe/paired/study/random_grip/'): return 'paired_random_belief_banks'
    if '/target_control/' in p: return 'target_trials' if path.name=='trials.npz' else 'target_planner_choices'
    if '/diagnostics/rollout/' in p: return 'rollout_'+path.stem
    raise ValueError(f'Unclassified NPZ: {path}')


GENERATION={
    'nov13_main':'Frozen RL-policy lifts with constant40g policy mass guess; scripted positional pushes; additive Gaussian action noise SD0.05;50commands/episode.',
    'three_task_main':'Scripted random-speed/height lift; random-speed/direction unilateral slow push; random1–4N horizontal robot-wrench shove for0.3s then withdrawal/coast. Fixed30.0625N grip.',
    'paired_fixed':'Same three-task script and seed42 physics/motion as three_task_main; fixed30.0625N grip; additional observed sidechannels.',
    'paired_random':'Same paired scene/motion seeds; fixed30.0625N acquisition then truth-independent stratified log-force draw .25–30.0625N atstep45 for lift/slowpush; shoves shared unchanged.',
    'smoke_fixed':'Two-cube smoke version of paired_fixed; IDs0/1 already occur in full training set.',
    'smoke_random':'Two-cube smoke version of paired_random; IDs0/1 already occur in full training set.',
    'pilot_constant_cap':'Fullfactorial4mass/friction corners×3resetseeds×8gripcaps×2tasks; same original lift/slowpush; cap held throughout episode.',
    'pilot_staged_cap':'Same fullfactorial pilot; fixed30.0625N untilstep45 then selectedcap independent of physics.',
    'target_candidate_cache':'Actual replayed shove candidates selected by frozen model+inferredbelief planner; candidate grid1–4N is fixed and physics-independent; true physics used only for simulator reset.',
    'oracle_complement_cache':'Remaining force-grid candidates absent from target cache, executed to enumerate all13candidates for150heldout setups; fixed candidate commands, not oracle-selected controls.',
    'force_cap_native_traces':'100Hz diagnostic interventions on2heldout setups: baseline, zero-horizontal, novertical, noorientation, horizontal-only, lowfinger-cap, zeroarmvelocity, softwithdrawal. Not ordinary training trajectories.',
}


def main():
    files=sorted(p for p in BASE.rglob('*') if p.is_file())
    groups=defaultdict(lambda:dict(files=[],schemas={},rows=0,episode_lengths=Counter(),task_episodes=Counter(),
        split_rows=Counter(),split_episodes=Counter(),physical={},split_physical=defaultdict(dict),derived_records=0,
        label_constancy_violations=[],episode_keys=set(),reset_seeds=set(),variants=Counter()))
    canonical={}
    for p in sorted((BASE/'three_task/data').glob('*/*.npz')):
        with np.load(p,allow_pickle=False) as d:canonical[int(d['cube_id'][0])]=(float(d['true_mass'][0]),float(d['true_friction'][0]))
    used_fields=set()
    for p in (f for f in files if f.suffix=='.npz'):
        name=classify(p);g=groups[name];fields=schema(p);used_fields.update(fields)
        normalized={k:dict(v,shape=(['N']+v['shape'][1:] if v['shape'] and ('state' in fields or name=='force_cap_native_traces') and v['shape'][0]==fields.get('state',fields.get('control_step',{})).get('shape',[None])[0] else v['shape'])) for k,v in fields.items()}
        signature=hashlib.sha256(json.dumps(normalized,sort_keys=True).encode()).hexdigest()[:16]
        g['schemas'].setdefault(signature,dict(fields=normalized,files=0))['files']+=1
        record=dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,schema=signature)
        g['files'].append(record)
        with np.load(p,allow_pickle=False) as d:
            if 'state' in fields:
                n=fields['state']['shape'][0];g['rows']+=n;record['rows']=n
                split=p.parent.name if p.parent.name in ('train','test') else ('inherited_test' if name.endswith('_cache') else 'diagnostic_only')
                g['split_rows'][split]+=n
                if name.startswith('pilot_'):
                    m=re.fullmatch(r'task_(\d+)_m_([\d.]+)_mu_([\d.]+)_seed_(\d+)_cap_([\d.]+)\.npz',p.name)
                    task,mass,mu,reset,cap=m.groups();task=int(task);key=f'pilot_mass{mass}_friction{mu}'
                    g['physical'][key]=(float(mass),float(mu));g['split_physical'][split][key]=(float(mass),float(mu))
                    g['reset_seeds'].add(int(reset));g['episode_lengths'][n]+=1;g['task_episodes'][str(task)]+=1
                    g['split_episodes'][split]+=1;g['episode_keys'].add(p.name)
                    record.update(task=task,mass_kg=float(mass),friction=float(mu),reset_seed=int(reset),grip_cap_n=float(cap))
                elif 'true_mass' in fields:
                    cube=d['cube_id'];task=d['task'];episode=d['episode_id'];step=d['step']
                    keys=np.column_stack((cube,task,episode));u,index,counts=np.unique(keys,axis=0,return_index=True,return_counts=True)
                    for key,start,count in zip(u,index,counts):
                        g['episode_keys'].add(tuple(map(int,key)));g['episode_lengths'][int(count)]+=1
                        g['task_episodes'][str(int(key[1]))]+=1;g['split_episodes'][split]+=1
                        if not np.array_equal(step[start:start+count],np.arange(count)):raise AssertionError(p)
                    for c in np.unique(cube):
                        selected=cube==c;mass=np.unique(d['true_mass'][selected]);mu=np.unique(d['true_friction'][selected])
                        if len(mass)!=1 or len(mu)!=1:g['label_constancy_violations'].append(dict(file=str(p),cube_id=int(c)))
                        pair=(float(mass[0]),float(mu[0]));key=f'seed42_cube{int(c)}'
                        if key in g['physical'] and g['physical'][key]!=pair:g['label_constancy_violations'].append(dict(file=str(p),cube_id=int(c),across_files=True))
                        g['physical'][key]=pair;g['split_physical'][split][key]=pair
                        if not np.allclose(pair,canonical[int(c)],rtol=0,atol=4e-8):raise AssertionError('Related cube physics differs')
                    record['cube_ids']=np.unique(cube).astype(int).tolist()
                else:
                    cube=int(d['cube_id'][0]);ep=int(d['episode_id'][0]);candidate=int(d['candidate_index'])
                    key=f'seed42_cube{cube}';g['physical'][key]=canonical[cube];g['split_physical']['inherited_test'][key]=canonical[cube]
                    g['episode_lengths'][n]+=1;g['task_episodes']['2']+=1;g['split_episodes']['inherited_test']+=1
                    g['episode_keys'].add((cube,ep,candidate));record.update(cube_id=cube,episode_id=ep,candidate_index=candidate)
            elif name=='force_cap_native_traces':
                n=fields['control_step']['shape'][0];g['rows']+=n;g['episode_lengths'][n]+=1
                g['split_rows']['inherited_test']+=n;g['split_episodes']['inherited_test']+=1
                companion=json.loads(p.with_suffix('.json').read_text());cube=companion['cube_id'];key=f'seed42_cube{cube}'
                g['physical'][key]=(companion['mass_kg'],companion['friction']);g['split_physical']['inherited_test'][key]=g['physical'][key]
                g['variants'][companion['variant']]+=1;g['task_episodes']['2']+=1;g['episode_keys'].add(p.name)
                record.update(rows=n,cube_id=cube,episode_id=companion['episode_id'],variant=companion['variant'])
            elif 'cube_id' in fields:
                record['derived_records']=fields['cube_id']['shape'][0];g['derived_records']+=record['derived_records']
                for cube in np.unique(d['cube_id']):
                    key=f'seed42_cube{int(cube)}';g['physical'][key]=canonical[int(cube)]
                    g['split_physical']['inherited_test'][key]=canonical[int(cube)]
    for name,g in groups.items():
        g['path_roots']=sorted({str(Path(f['path']).parent) for f in g['files']})
        g['generation']=GENERATION.get(name,'Derived learned-belief or prediction/evaluation artifact; not a simulator transition dataset.')
        g['kind']='control_transitions' if name in GENERATION and name!='force_cap_native_traces' else ('native_substep_diagnostics' if name=='force_cap_native_traces' else 'derived_artifact')
        g['file_count']=len(g['files']);g['stored_episode_or_variant_count']=sum(g['episode_lengths'].values())
        g['unique_episode_or_variant_keys']=len(g.pop('episode_keys'));g['reset_seed_values']=sorted(g.pop('reset_seeds'))
        g['task_names']={'0':'lift','1':'slide' if name.startswith('nov13') else 'slow_slide','2':'shove'}
        g['sampling_hz']=100 if name=='force_cap_native_traces' else (20 if g['kind']=='control_transitions' else None)
        if g['kind']=='derived_artifact':g['rows_note']='rows=0 means no simulator transition rows; derived_records counts cube-bank or decision/metric rows across files.'
        g['mass_friction_correlation']=correlation(g['physical'].values())
        g['split_correlations']={split:correlation(values.values()) for split,values in g['split_physical'].items()}
        g['unique_physical_records']=[dict(key=k,true_mass_kg=v[0],true_friction=v[1]) for k,v in sorted(g.pop('physical').items())]
        g['split_physical_keys']={split:sorted(values) for split,values in g.pop('split_physical').items()}
        g['mass_friction_stored_and_constant_per_cube']=not g['label_constancy_violations'] if name in ('nov13_main','three_task_main','paired_fixed','paired_random','smoke_fixed','smoke_random') else None
        g['label_source']=('Per-row true_mass/true_friction columns, verified constant within cube across every stored row.' if g['mass_friction_stored_and_constant_per_cube'] is not None else
            'Filenames and pilot summary.json; labels are NOT columns inside NPZ.' if name.startswith('pilot_') else
            'Companion JSON per diagnostic trace.' if name=='force_cap_native_traces' else
            'External cube-ID join to three_task/data; no true-mass/friction columns in these NPZs.')
        if name.startswith('pilot_'):g['physical_identity_note']='No cube_id field. Correlation uses four distinct physics conditions, not 192 repeated trials or 12 reset poses in this group.'
        if name.startswith('smoke_'):g['split_overlap_note']='Smoke test cube1 is full-dataset TRAIN cube1; smoke splits cannot be treated as an additional independent held-out set when datasets are pooled.'
    csvs=[]
    for p in (f for f in files if f.suffix=='.csv'):
        with p.open() as f:reader=csv.DictReader(f);rows=list(reader);columns=reader.fieldnames
        csvs.append(dict(path=str(p.relative_to(ROOT)),rows=len(rows),columns=columns,
            kind='Derived evaluation/decoder/oracle table; not per-step simulator inputs',
            privileged_fields=[c for c in columns if c.startswith('true_') or 'oracle' in c or c.startswith('actual_')]))
    jsons=[]
    for p in (f for f in files if f.suffix=='.json'):
        obj=json.loads(p.read_text());tables=[]
        if isinstance(obj,dict):
            for key,value in obj.items():
                if isinstance(value,list) and value and isinstance(value[0],dict):
                    tables.append(dict(field=key,rows=len(value),columns=sorted({k for r in value if isinstance(r,dict) for k in r})))
        jsons.append(dict(path=str(p.relative_to(ROOT)),top_level_keys=list(obj) if isinstance(obj,dict) else None,
                          record_tables=tables,kind='Metadata, provenance, aggregate diagnostics, or tabulated diagnostic records'))
    original=subprocess.check_output(['git','show','0b00d22:experiments/lift_slide_belief/collect.py'],cwd=ROOT)
    old_manifest=json.loads((BASE/'nov13/data/dataset.json').read_text())
    assert hashlib.sha256(original).hexdigest()==old_manifest['code_sha256']['experiments/lift_slide_belief/collect.py']
    state_schema=json.loads((BASE/'three_task/data/dataset.json').read_text())['state']
    source_files=['experiments/lift_slide_belief/collect.py','experiments/lift_slide_belief/scripted.py',
        'experiments/lift_slide_belief/slip_data.py','experiments/lift_slide_belief/target_control.py',
        'experiments/lift_slide_belief/oracle_diagnostics.py','experiments/lift_slide_belief/force_cap_diagnostics.py',
        'experiments/lift_slide_belief/slip_study.py','experiments/lift_slide_belief/evaluation.py',
        'experiments/context_dynamics/collect.py','robot_adaptation/simulation/context_dynamics.py',
        'robot_adaptation/simulation/collection_lift.py']
    checks={}
    for p in BASE.rglob('dataset.json'):
        cfg=json.loads(p.read_text());checks[str(p.relative_to(ROOT))]={}
        for name,expected in cfg.get('code_sha256',{}).items():
            actual=sha(ROOT/name)
            if name=='experiments/lift_slide_belief/collect.py' and 'nov13' in str(p):actual=hashlib.sha256(original).hexdigest()
            checks[str(p.relative_to(ROOT))][name]=dict(expected=expected,verified=actual==expected)
    audit=dict(schema_version=1,generated_utc=datetime.now(timezone.utc).isoformat(),scope=str(BASE.relative_to(ROOT)),
        mode='Read-only inventory; no collection, model modification, fitting or training performed.',
        global_training_gate='STOPPED by parent due independently verified leakage in other historical dataset families; this inventory does not authorize training.',
        all_files=len(files),extension_counts=dict(Counter(p.suffix for p in files)),
        covered_npz_files=sum(g['file_count'] for g in groups.values()),groups=dict(groups),
        total_stored_control_transitions=sum(g['rows'] for g in groups.values() if g['kind']=='control_transitions'),
        total_native_diagnostic_samples=sum(g['rows'] for g in groups.values() if g['kind']=='native_substep_diagnostics'),
        counts_note='Stored rows include deliberately repeated trajectories and overlapping variants; they are not independent physics samples.',
        global_unique_seed42_cube_correlation=correlation(canonical.values()),
        state43_schema=state_schema,projection51=dict(state43=[0,43],relative_position=[43,46],causal_relative_velocity=[46,49],finger_position_error=[49,51]),
        action_schemas=dict(command5=['dx','dy','dz','finger_opening','strength'],command8=['dx','dy','dz','finger_opening','strength','tcp_force_x_n','tcp_force_y_n','cartesian_force_mode']),
        fields_covered=sorted(used_fields),csv_artifacts=csvs,json_artifacts=jsons,
        checkpoint_artifacts=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,kind='Trained model checkpoint, not a per-step dataset') for p in files if p.suffix=='.pt'],
        other_files=[str(p.relative_to(ROOT)) for p in files if p.suffix not in ('.npz','.csv','.json','.pt')],
        source_sha256={p:sha(ROOT/p) for p in source_files},manifest_source_verification=checks,
        original_collector=dict(git_object='0b00d22:experiments/lift_slide_belief/collect.py',sha256=hashlib.sha256(original).hexdigest(),exact_manifest_match=True),
        leakage_audit=dict(proven_direct_true_physics_in_main_state_or_issued_action=False,
            ordinary_boundary='Only explicit observed state43 (or its declared51D projection) and issued action5/8. cube/task/episode IDs, true labels, episode outcomes and future diagnostics are not model inputs.',
            direct_privileged_columns=['true_mass','true_friction'],
            policy_lift='Exact historical collector supplies actor_mass_guess_kg=.04 times gravity, never sampled true_mass, to policy_obs. Policy-only guessed weight is not saved inside state43.',
            scripted_motion='Mass and friction only instantiate simulator physics and checking labels. Lift speed/height, slide speed/direction and shove force draw from [seed,cube,1,task,episode] motion RNG; physics uses [seed,cube,0].',
            grip_controls='Random cap depends on independent [seed,cube,917,task,episode//5] RNG and fixed step45; pilot caps form the same fullfactorial sweep for every mass/friction condition. No mg/mu-derived cap.',
            force_controls='J-transpose wrench uses commanded horizontal force, fixed vertical/orientation targets, observed robot state and robot passive compensation; no true cube mass/friction is read in control.',
            candidate_caches='True physics resets the simulator. Candidate magnitudes use a fixed1–4N grid. Model planner chooses using measured ready state and learned beliefs, not labels. Oracle-complement cache enumerates remaining candidates independently of physics labels.',
            privileged_derived_artifacts=['linear_decoder is fitted with true training physics labels; not an ordinary observation.',
                'oracle_force_n/oracle_candidate_index in oracle CSVs use actual future counterfactual simulator outcomes; do not use as blind-input features or deployable control rules.',
                'actual_endpoint/target_error/success and episode stop/slip/acquisition summaries reveal future outcomes; not pre-action features.',
                'Diagnostic beliefs are learned/adapted artifacts, not raw sensors; ceiling beliefs already use other same-cube shove observations.'],
            observed_responses='Mass/friction effects in measured forces, positions, velocities, gaps and contact timing are intended physical evidence, not direct label leakage.',
            source_references=['git0b00d22 collect.py:record_cube,measure; currentcollect.py:physics,measure,record_cube',
                'context_dynamics.py:PandaMotion.state,set_episode,step; context_dynamics/collect.py:policy_obs',
                'scripted.py:record_episode,shove_actions,_force_substep; slip_data.py:measured_state,grip_cap_for_episode',
                'target_control.py:_plan_seed,_simulate_variant; oracle_diagnostics.py:oracle_case; slip_study.py:evaluate_readouts']),
        split_cautions=['IDs0..199 are train and200..229 test in all fullmain variants. They identify the same sampled physical cubes across nov13/three_task/paired arms.',
            'Smoke IDs0/1 duplicate fulltraining cubes, including smoke testcube1.',
            'All target/oracle/force diagnostic cache cubeIDs inherit fulltest membership; do not silently promote them into training.',
            'Pilots have factorial physics conditions but no independent train/test cube split.',
            'Deduplicate by canonical physicalcube, not byfile,episode,trajectoryvariant,modelseed or griparm before correlations/splitting.'])
    out=Path(__file__).with_name('inventory_lift_slide.json')
    out.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out),npz_files=audit['covered_npz_files'],groups=len(groups),
        stored_control_transitions=audit['total_stored_control_transitions'],native_samples=audit['total_native_diagnostic_samples'],
        correlation=audit['global_unique_seed42_cube_correlation'],local_direct_leak=False)))


if __name__=='__main__':main()
