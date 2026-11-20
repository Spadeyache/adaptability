"""Read-only inventory of legacy collection/conditioning trajectory families.

Writes only results/probe/inventory_other.json. Does not simulate, train, import
simulator packages, or modify old data. Run with .venv/bin/python this_file.py.
"""
from __future__ import annotations
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROOTS = ['failure_collection','failure_discovery','physics_conditioning','interaction_adaptation','maniskill_pretrained']
PHYSICAL = ['mass_kg','object_static_friction','object_dynamic_friction','finger_static_friction','finger_dynamic_friction','table_static_friction','table_dynamic_friction','shape','size_xyz_m','com_offset_m','inertia_principal_kg_m2','linear_damping','angular_damping','object_restitution','finger_restitution','table_restitution']

def rel(p): return str(Path(p).relative_to(ROOT))
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_json(p): return json.loads(p.read_text())
def scalar(x):
    a=np.asarray(x)
    return a.item() if a.ndim==0 else a.tolist()
def array(z,k):
    with z.open(k+'.npy') as f:return np.lib.format.read_array(f,allow_pickle=False)
def header(z,k):
    with z.open(k+'.npy') as f:
        v=np.lib.format.read_magic(f)
        sh,_,dt=np.lib.format._read_array_header(f,v)
    return list(sh),str(dt)
def hist(values):return {str(k):int(v) for k,v in sorted(collections.Counter(values).items(),key=lambda x:str(x[0]))}
def finite_corr(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None

def cadence(k,sh,t,phys):
    if k.startswith(('label__','probe__mode','probe__supplied_parameters')) or len(sh)==0:return 'episode_constant_or_label'
    if 'contact_point_offsets' in k:return 'ragged_contact_offsets_physics_plus_one'
    if k.startswith('sim__contact_'):return 'ragged_contact_points'
    if k in ['input__goal_position_world_m']:return 'episode_constant'
    if sh and sh[0]==t+1:return 'state_T_plus_1'
    if sh and sh[0]==t:return 'transition_T'
    if sh and sh[0]==phys:return 'physics_T_times_substeps'
    return 'metadata_or_other'

def short_manifest(parent):
    candidates=[parent/'manifest.json',parent.parent/'manifest.json']
    for p in candidates:
        if p.exists():
            d=load_json(p)
            keys=['checkpoint','checkpoint_sha256','checkpoint_label','status','task','task_description','task_semantics','method','policy','estimator','policy_input','observations','actions','input_fields','input_sizes','settings','config','plan','diagnostic_subset','success_definition','estimator_input_boundary']
            out={k:d[k] for k in keys if k in d}
            out['path']=rel(p);out['sha256']=digest(p)
            if isinstance(out.get('plan'),list) and len(out['plan'])>40:
                out['plan']={'count':len(out['plan']),'first':out['plan'][:2]}
            # Training provenance may be sizable, but retain relevant action configuration.
            tm=d.get('training_manifest',d.get('policy_manifest',{}))
            if isinstance(tm,dict):
                out['policy_training_settings']={k:tm.get('settings',{}).get(k) for k in ['masses_kg','grip_force_limits_n','friction_profile','condition_on_friction','cube_friction_coefficients','goal_clearance_m'] if k in tm.get('settings',{})}
                out['policy_input_fields']=tm.get('input_fields')
            return out
    return None

def row_metadata(parent):
    result={}
    for n in ['attempts.jsonl','rows.jsonl']:
        p=parent.parent/n if parent.name=='episodes' else parent/n
        if p.exists():
            for line in p.open():
                r=json.loads(line)
                result[str(r['episode_id'])]=r
    return result

def npz_family(parent,files):
    records=[];fields={};schemas=collections.Counter();seen_schema=set();metadata=row_metadata(parent)
    for p in files:
        with zipfile.ZipFile(p) as z:
            keys=tuple(sorted(n[:-4] for n in z.namelist() if n.endswith('.npy')))
            ash,_=header(z,'input__action_applied');t=ash[0]
            psh,_=header(z,'meta__physics_time_s');phys=psh[0]
            schema=str(scalar(array(z,'meta__schema_version')));schemas[schema]+=1
            if keys not in seen_schema:
                for k in keys:
                    sh,dt=header(z,k)
                    spec={'dtype':dt,'example_shape':sh,'cadence':cadence(k,sh,t,phys),'example_file':rel(p)}
                    if k not in fields:fields[k]=spec
                    elif fields[k]['dtype']!=dt or fields[k]['cadence']!=spec['cadence']:
                        fields[k].setdefault('variants',[]).append(spec)
                seen_schema.add(keys)
            # Each field's presence is checked for every file; shape examples are
            # inspected once per exact key-set schema, and action/physics lengths
            # plus all selected physical scalars are checked for every episode.
            for k in keys:fields[k]['files_present']=fields[k].get('files_present',0)+1
            props={k:scalar(array(z,'label__'+k)) for k in PHYSICAL if 'label__'+k in keys}
            mass=float(props['mass_kg']);mu=float(props['object_static_friction'])
            r={'path':rel(p),'steps':t,'action_dim':ash[1],'mass':mass,'mu':mu,'dynamic_mu':float(props['object_dynamic_friction']),
               'physics_samples':phys,'physical_setting':json.dumps(props,sort_keys=True),'seed':scalar(array(z,'label__seed')) if 'label__seed' in keys else None,
               'environment':str(scalar(array(z,'meta__environment')))}
            for k in ['label__condition_id','input__reported_mass_kg','probe__mode']:
                if k in keys:r[k]=scalar(array(z,k))
            if 'sim__policy_observation' in keys:
                obs=array(z,'sim__policy_observation');r['policy_dim']=obs.shape[-1]
                if obs.shape[-1]>42:
                    r['policy_weight_constant']=bool(np.all(obs[:,42]==obs[0,42]));r['policy_weight_n']=float(obs[0,42])
                    r['weight_equals_true_mass_g']=bool(np.allclose(obs[:,42],mass*9.81,rtol=1e-6,atol=1e-6))
            if 'probe__supplied_policy_inputs' in keys:
                obs=array(z,'probe__supplied_policy_inputs');r['supplied_dim']=obs.shape[-1]
                r['supplied_weight_n']=float(obs[0,42]);r['supplied_friction']=float(obs[0,45])
                r['supplied_parameters_constant']=bool(np.all(obs[:,[42,45]]==obs[0,[42,45]]))
            limits=array(z,'input__grip_motor_limit_n');r['grip_min']=float(limits.min());r['grip_max']=float(limits.max())
            if p.stem in metadata:
                r['paired_row']={k:metadata[p.stem][k] for k in ['method','mode','supplied_mass_kg','supplied_friction','actual_mass_kg','actual_friction','condition_index','condition','sequence_index','sequence_key','attempt','grid'] if k in metadata[p.stem]}
            records.append(r)
    props_group=collections.defaultdict(list)
    ids=collections.defaultdict(list)
    for r in records:
        props_group[r['physical_setting']].append(r)
        ids[str(r.get('label__condition_id',r['physical_setting']))].append(r)
    unique_pair=sorted({(r['mass'],r['mu']) for r in records})
    unique_props=[v[0] for v in props_group.values()]
    idconsistent=all(len({(r['mass'],r['mu'],r['dynamic_mu']) for r in group})==1 for group in ids.values())
    modes=hist(r.get('paired_row',{}).get('mode',r.get('paired_row',{}).get('method',r.get('probe__mode','unspecified'))) for r in records)
    reported=[r['input__reported_mass_kg'] for r in records if 'input__reported_mass_kg' in r]
    out={'path':rel(parent),'kind':'raw_recorded_episode_npz','episode_count':len(records),'file_pattern':'*.npz',
         'all_file_paths_sha256':hashlib.sha256('\n'.join(r['path'] for r in records).encode()).hexdigest(),
         'first_file':records[0]['path'],'last_file':records[-1]['path'],'schema_versions':dict(schemas),'key_set_count':len(seen_schema),'fields':fields,
         'steps_histogram':hist(r['steps'] for r in records),'total_control_steps':sum(r['steps'] for r in records),'physics_samples_histogram':hist(r['physics_samples'] for r in records),
         'action_dimensions':hist(r['action_dim'] for r in records),'policy_observation_dimensions':hist(r.get('policy_dim') for r in records),
         'independent_cube_id_present':False,'identity_semantics':'Fresh simulator episodes indexed by physical condition/reset seed; no persistent independently sampled cube identity. Condition hash is physical/intervention setting identifier.',
         'condition_id_count':len(ids),'condition_ids_from_field':any('label__condition_id' in r for r in records),'episodes_per_condition_histogram':hist(len(v) for v in ids.values()),
         'unique_physical_setting_count':len(props_group),'episodes_per_physical_setting_histogram':hist(len(v) for v in props_group.values()),
         'condition_to_parameters_single_valued':idconsistent,'physical_setting_label_correlation':'Setting/condition IDs identify parameter values; do not use as predictor features. Splitting reset episodes alone reuses physical settings.',
         'masses_kg':sorted({r['mass'] for r in records}),'object_static_friction_values':sorted({r['mu'] for r in records}),'object_dynamic_friction_values':sorted({r['dynamic_mu'] for r in records}),
         'unique_mass_friction_pairs':unique_pair,'mass_friction_pearson_across_unique_settings':finite_corr([r['mass'] for r in unique_props],[r['mu'] for r in unique_props]),
         'mass_friction_cartesian_grid_complete':len(unique_pair)==len({r['mass'] for r in records})*len({r['mu'] for r in records}),
         'truth_storage':'Episode scalar label__mass_kg, label__object_static_friction, label__object_dynamic_friction, plus explicit material/body metadata. All files read; scalar per-episode storage. Owning collectors set properties at reset, not during step.',
         'per_step_truth_constancy':{'policy_weight_constant_episodes':sum(r.get('policy_weight_constant',False) for r in records),'policy_weight_episodes_checked':sum('policy_weight_constant' in r for r in records),'probe_supplied_parameters_constant_episodes':sum(r.get('supplied_parameters_constant',False) for r in records)},
         'reported_mass_values_kg':sorted(set(reported)),'reported_mass_equal_true_count':sum(np.isclose(r['input__reported_mass_kg'],r['mass'],rtol=1e-6,atol=1e-8) for r in records if 'input__reported_mass_kg' in r),
         'policy_weight_equal_true_mass_g_count':sum(r.get('weight_equals_true_mass_g',False) for r in records),
         'modes_or_methods':modes,'realized_grip_motor_limit_range_n':[min(r['grip_min'] for r in records),max(r['grip_max'] for r in records)],
         'environments':hist(r['environment'] for r in records),'reset_seed_count':len({r['seed'] for r in records}),'reset_seed_range':[min(r['seed'] for r in records),max(r['seed'] for r in records)],
         'existing_split':'No train/test cube split stored in these raw episode fields or family path. Evaluation interventions/grids and reset seeds are not an independent-cube training split.',
         'manifest':short_manifest(parent)}
    # Evidence rows are compact, exposing every distinct mode without copying arrays.
    out['mode_examples']={}
    for r in records:
        mode=r.get('paired_row',{}).get('mode',r.get('paired_row',{}).get('method',r.get('probe__mode','unspecified')))
        if mode not in out['mode_examples']:
            out['mode_examples'][mode]={k:v for k,v in r.items() if k!='physical_setting'}
    if 'failure_collection' in str(parent):
        out['generator']='robot_adaptation.collection.stock' if 'stock_short' in str(parent) else 'robot_adaptation.collection.varied; fixed learned policy, deterministic actions; no action exploration noise'
        out['task']='Stock random-XYZ pick/carry goal' if 'stock_short' in str(parent) else 'Vertical pick, lift and hold directly above spawn; no slide task'
        out['grip']='Native fourth action controls finger opening, motor cap100N' if 'stock_short' in str(parent) else 'Learned fifth action controls gripper motor force cap; fourth controls finger opening; supplied mass fixed40g'
        out['label_leak_status']='Truth available in label__/sim metadata but no direct true mass/friction supplied to this collection policy; 40g field is constant across unequal real masses, not a measured weight. Raw labels/metadata must be excluded.'
    elif 'checkpoint_comparison' in str(parent) or 'checkpoint_probe_smoke' in str(parent):
        out['generator']='experiments.physics_conditioning.checkpoint_probe; deterministic46-input5-action conditioned policy; six interventions'
        out['task']='Native training carry geometry, randomized XY and floating goal Z; CPU fresh scenes'
        out['grip']='Learned fifth action force cap, configured .25–40 N; observed limits inventoried'
        out['label_leak_status']='PROVEN privileged modes: supplied weight column42, friction column45. true/fixed_friction/fixed_mass/wrong_friction/wrong_mass retain at least one true or invertible-label coordinate. fixed_both alone uses constants .04kg/.5; modes must not be pooled as unknown-physics behavior.'
    else:
        out['generator']='experiments.interaction_adaptation.evaluate retries or transfer; deterministic46-input5-action policy; estimator updates between attempts, fresh scene each attempt'
        out['task']='Native carry or vertical pick-and-hold as environment field specifies; no slide task'
        out['grip']='Learned fifth action force cap, usually configured .25–40 N'
        out['label_leak_status']='Mixed methods: true retries explicitly supply true mass/friction to policy. Fixed/transfer use known constants; adaptive uses earlier observed attempts. input__reported_mass_kg and policy observation42 reveal supplied mass. Friction supplied at appended policy45 is in paired attempts.jsonl; raw sim__policy_observation stores only45 pre-append values.'
    return out

def nested_fields(obj,prefix=''):
    out={}
    if isinstance(obj,dict):
        for k,v in obj.items():out.update(nested_fields(v,prefix+'.'+k if prefix else k))
    else:
        shape=list(np.asarray(obj).shape) if isinstance(obj,list) else []
        out[prefix]={'python_type':type(obj).__name__,'example_shape':shape}
    return out

def jsonl_family(p):
    rows=[json.loads(s) for s in p.open()]; fields={}
    for row in rows:fields.update(nested_fields(row))
    kind='per_episode_summary_already_represented_by_npz' if p.name in ['rows.jsonl','attempts.jsonl'] else 'optimizer_metrics_not_trajectories' if p.name=='trust_region.jsonl' else 'raw_batched_policy_step_trace' if p.name=='traces.jsonl' else 'raw_episode_step_trace'
    out={'path':rel(p),'kind':kind,'records':len(rows),'all_fields':fields,'sha256':digest(p)}
    if p.name=='traces.jsonl':
        groups=collections.defaultdict(list)
        for r in rows:groups[(r['seed'],r['mode'])].append(r)
        nenv=len(rows[0]['observation'])
        out.update(record_layout='One JSON record per simulator step containing a batch of physical settings; no terminal next observation is saved after final action.',batch_size=nenv,
                   episode_count=len(groups)*nenv,steps_per_episode_histogram=hist(len(v) for v in groups.values()),modes=sorted({r['mode'] for r in rows}),seeds=sorted({r['seed'] for r in rows}),
                   observation_dimension=len(rows[0]['observation'][0]),action_dimension=len(rows[0]['action'][0]),
                   truth_inside_observation={'mass_index_zero_based':42,'mass_encoding':'true kg × gravity','friction_index_zero_based':45,'friction_encoding':'true coefficient'},
                   action_dependency='Policy receives supplied vector. True mode retains truths. Wrong friction=1.2−truth. Six-mode source also wrong mass=.54−truth; these remain label functions.',
                   constancy_checked=True,per_setting_constant_weight_and_friction=all(np.all(np.asarray([r['observation'] for r in v])[:,:,[42,45]]==np.asarray(v[0]['observation'])[:,[42,45]]) for v in groups.values()),
                   manifest=short_manifest(p.parent),existing_split='Evaluation seeds and crossed physical settings; no persistent cube train/test split')
        pairs=[]
        for g,v in groups.items():
            for ob in v[0]['observation']:pairs.append((ob[42]/9.81,ob[45]))
        out['mass_friction_pairs_from_first_observation']=sorted(set(pairs))
        out['recorded_episodes_vs_summary']='Only trace modes and first seed are serialized; episodes.json can contain additional aggregate outcomes, which are not full trajectories.'
    return out

def pretrained_family(parent,files):
    fields={};lengths=[];masses=[];const=[];dimensions=[];hashes=[]
    for p in files:
        rows=[json.loads(x) for x in p.open()]
        states=[rows[0]['state']]+[r['after'] for r in rows[1:]]
        for r in rows:fields.update(nested_fields(r))
        lengths.append(len(rows)-1);mm=[s['cube_mass_kg'] for s in states];masses.append(mm[0]);const.append(len(set(mm))==1)
        dimensions.append(len(states[0]['policy_observation']));hashes.append(digest(p))
    return {'path':rel(parent),'kind':'raw_jsonl_episodes','episode_count':len(files),'steps_histogram':hist(lengths),'total_control_steps':sum(lengths),'all_fields':fields,'all_file_hashes':dict(zip([p.name for p in files],hashes)),
            'masses_kg':sorted(set(masses)),'episodes_per_mass_histogram':hist(masses),'mass_constant_all_steps':all(const),'friction_stored':False,
            'independent_cube_id_present':False,'task':parent.name,'generator':'robot_adaptation/policy/examples/maniskill_pick_cube/run_pretrained.py; official42-input4-action pretrained PPO deterministic policy',
            'grip':'Native finger-opening command, no independent learned grip-strength output',
            'policy_observation_dimensions':hist(dimensions),'label_leak_status':'cube_mass_kg is a separate nested state field: direct label if naively flattened as a learner input. Policy vector deliberately removes appended weight and uses original42 PickCube inputs; policy actions do not directly receive mass/friction.',
            'parameter_correlation':'Custom-lift masses cycle by episode index; episode/reset seed can identify assigned mass. Official task is a single nominal mass. No sampled cube IDs.',
            'existing_split':'Evaluation only; no cube train/test split. Two20-episode runs share same seeds/settings; exact matching trace hashes reported separately.',
            'manifest':short_manifest(parent)}

def archives(root):
    entries=[]
    for parent in sorted(p for p in root.iterdir() if p.is_dir()):
        files=[p for p in parent.rglob('*') if p.is_file()]
        e={'path':rel(parent),'file_extensions':hist(p.suffix for p in files),'kind':'derived_visual_review_not_new_simulation_dataset'}
        p=parent/'manifest.json'
        if p.exists():
            d=load_json(p); e.update(manifest_path=rel(p),manifest_sha256=digest(p),source_dataset=d.get('dataset'),episode_count=d.get('episode_count'),source_file_count=len(d.get('source_files',[])))
            e['source_raw_paths']=sorted({str(r.get('file')) for r in d.get('source_files',[])})
        entries.append(e)
    return entries

def supplement(result):
    """Small provenance/summary-only additions; reusable without rescanning NPZs."""
    result['workspace_root']=str(ROOT)
    result['path_semantics']='All paths are relative to workspace_root unless explicitly absolute.'
    result['field_alignment']={
        'state':'T+1 samples including reset, state[t] is before action[t].',
        'transition':'T samples, command[t] and its observed response ending at state[t+1].',
        'physics':'T*K samples, K=sim_hz/control_hz; samples occur after each physics substep.',
        'label':'Scalar/constant labels describe episode physical parameters or derived episode outcomes; never blanket-flatten labels into predictor inputs.',
        'units':'SI: kg, m, s, N, radians. Quaternions wxyz. Joint positions first7 rad, final2 m.',
        'unknowns':'No audit of real-robot sensor feasibility, noise or calibration. No independent measurements of true physical parameters; these are simulator labels/readbacks. This inventory is not a scientific split approval.'}
    result['aggregate_only_evaluation_files']=[]
    for p in sorted((ROOT/'runs/physics_conditioning').rglob('episodes.json')):
        rows=load_json(p);fields={}
        for r in rows:fields.update(nested_fields(r))
        summary=load_json(p.parent/'summary.json')
        result['aggregate_only_evaluation_files'].append({'path':rel(p),'kind':'per_episode_outcomes_not_full_trajectories','episode_outcomes':len(rows),
            'all_fields':fields,'modes':hist(r['mode'] for r in rows),'seeds':sorted({r['seed'] for r in rows}),
            'physics_pairs':sorted({(r['mass_kg'],r['friction']) for r in rows}),
            'horizon':summary.get('horizon'),'summary_path':rel(p.parent/'summary.json'),
            'valid_for_inference':summary.get('valid_for_inference'),'invalid_reason':summary.get('invalid_reason'),
            'limitation':'These outcome rows may exceed the serialized first-seed/mode step traces; do not count them as additional recoverable state-action trajectories.'})
    result['diagnostic_smoke_nontrajectory_files']=[{
        'path':'runs/physics_conditioning/checkpoint_probe_smoke/native_cpu_equivalence.json',
        'kind':'50-step paired numerical-difference trace; not state/action arrays',
        'all_fields':nested_fields(load_json(ROOT/'runs/physics_conditioning/checkpoint_probe_smoke/native_cpu_equivalence.json')['trace'][0])}]
    file_hashes=collections.defaultdict(list)
    for family in result['pretrained_episode_families']:
        for name,h in family['all_file_hashes'].items():file_hashes[h].append(family['path']+'/'+name)
    result['pretrained_identical_trace_archives']=[v for v in file_hashes.values() if len(v)>1]
    result['pretrained_unique_byte_identical_episode_count']=len(file_hashes)
    result['published_pretrained_metadata_note']='models/trajectory*.json and manifest.published_metadata describe external official demonstrations; no corresponding downloaded official training trajectory arrays are present here. Local evaluation JSONL traces are counted separately.'
    for family in result['raw_npz_families']:
        family['physical_setting_signature_fields']=['label__'+k for k in PHYSICAL]
        family['physical_setting_count_note']='Signature includes stored body/material properties above, excluding goal/spawn/reset. Thus multiple condition IDs may share one physical body setting; this is not the number of independently sampled persistent cubes.'
        family['equality_count_note']='Reported-mass/weight equality counts include coincidences where a fixed prior equals actual mass; equality alone does not establish leakage. Action-generation source and intervention mode determine privileged truth use.'
        metadata=row_metadata(ROOT/family['path'])
        family['grid_labels']=hist(r.get('grid','none') for r in metadata.values()) if metadata else {}
        if any(x!='none' for x in family['grid_labels']):
            family['existing_physics_grid_partition']='training grid versus intermediate off-grid physics as grid field/manifest plan; same evaluation reset-seed blocks. This is not a raw train/test episode split.'
    result['remaining_unknowns']=[
        'Collection manifests preserve generator provenance, but this audit did not replay simulators or reexecute each historical frozen source snapshot.',
        'Except explicit recorded invariance checks, scalar episode labels plus owning source establish constant-parameter episodes; no per-step physical-material readback arrays exist.',
        'Whole-source-array corruption checks and numerical finite validation were not run; this is a full family/field inventory with every-file scalar/length coverage.',
        'Realized grip caps are logged for NPZ episodes. Pretrained JSONL exports do not log a separate per-step motor force cap; their native controller setting must come from environment configuration.',
        'Pretrained JSONL exports do not store true friction. No friction should be invented from mass or filename; historical task friction rules require source-specific reconstruction if used later.'
    ]


def main():
    result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':[str(Path('runs')/r) for r in ROOTS],
            'mode':'READ_ONLY_INVENTORY_NO_TRAINING','global_gate':'STOP: proven direct target metadata in candidate inputs and privileged action-generation modes. No new training performed.',
            'scan_method':'Every NPZ ZIP member list inspected; action/physics shapes and selected labels/weights/caps read for every episode. All field headers inspected per exact key-set schema; not a whole-array corruption audit. Every JSONL record parsed; all nested field paths enumerated. Media/weights/source snapshots inventoried separately.',
            'raw_npz_families':[],'jsonl_artifacts':[],'pretrained_episode_families':[],'derived_archives':[], 'nontrajectory_artifacts':[]}
    npzgroups=collections.defaultdict(list)
    for rootname in ROOTS:
        root=ROOT/'runs'/rootname
        files=[p for p in root.rglob('*') if p.is_file()]
        result['nontrajectory_artifacts'].append({'path':rel(root),'file_extensions':hist(p.suffix for p in files),'note':'PT files are model checkpoints, Python files frozen source, plots/media are derived. NPZ and JSONL trajectories are enumerated explicitly below; training manifests/optimizer metrics do not retain raw PPO rollout buffers.'})
        for p in files:
            if p.suffix=='.npz':npzgroups[p.parent].append(p)
        if rootname=='failure_discovery':result['derived_archives']=archives(root)
        if rootname=='maniskill_pretrained':
            pg=collections.defaultdict(list)
            for p in files:
                if p.name.endswith('_steps.jsonl'):pg[p.parent].append(p)
            for par,ff in sorted(pg.items()):result['pretrained_episode_families'].append(pretrained_family(par,sorted(ff)))
        else:
            for p in sorted(files):
                if p.suffix=='.jsonl':result['jsonl_artifacts'].append(jsonl_family(p))
    for parent,files in sorted(npzgroups.items(),key=lambda kv:len(kv[1])):
        result['raw_npz_families'].append(npz_family(parent,sorted(files)))
        print('Scanned',rel(parent),len(files),flush=True)
    result['direct_leak_evidence']=[
        {'path':'runs/interaction_adaptation/initial_policy_probe/episodes/condition2_sequence0_true_attempt0.npz','input__reported_mass_kg':.5,'label__mass_kg':.5,'sim__policy_observation_shape':[51,45],'column42_constant_weight_n':4.90500020980835,
         'source':['experiments/interaction_adaptation/evaluate.py:175-183','robot_adaptation/simulation/collection_lift.py:50-54','experiments/physics_conditioning/evaluate.py:25-35'],
         'finding':'Explicit supplied truth appears both as scalar input and weight coordinate42; policy actions were generated from true mass and friction. Excluding the truth columns alone does not remove privileged behavior policy.'},
        {'paths':['runs/physics_conditioning/*/traces.jsonl','runs/physics_conditioning/checkpoint_comparison_20261026/*/episodes/*.npz'],'fields':['observation[:,42]','observation[:,45]','probe__supplied_policy_inputs[:,42]','probe__supplied_policy_inputs[:,45]'],
         'finding':'Actual observations contain mass×gravity and friction. Supplied intervention vectors use true, fixed, or invertibly wrong values. Only fixed_both removes both true label coordinates from the action policy. Raw observation still contains truths in GPU traces.',
         'source':['experiments/physics_conditioning/evaluate.py:25-35','experiments/physics_conditioning/evaluate.py:113-130','experiments/physics_conditioning/checkpoint_probe.py:supplied_parameters']},
        {'paths':['runs/maniskill_pretrained/pick_cube/*/*/episode_*_steps.jsonl'],'field':'before.cube_mass_kg / after.cube_mass_kg / state.cube_mass_kg','finding':'Separate truth metadata nested under state; direct mass label if blanket flattening includes it. Original policy_observation42 excludes mass and friction.'}]
    result['causal_and_sampling_cautions']=[
        'Do not confuse ideal sensed force/gap/motion responses with explicit labels: their physics dependence is the intended identification signal.',
        'input__* alone is insufficient allowlisting for unknown-mass learning: input__reported_mass_kg is constant in fixed40g data, but truth in true retries.',
        'Response/interval-force/pose-derived-velocity arrays describe action[t] to state[t+1]; use only after that transition, never as pre-action features for its next-state target.',
        'sim__policy_observation forces43/44 are object-attributed pairwise sensor proxies; contact masks and ragged contacts also use object identity. Label metadata includes inertia/weight/COM/materials; excluded from unprivileged features.',
        'Dense design changes at most two factors per condition along signed diagonals; many factors are nominal. It is not an IID fully crossed physics population. Physical-setting or condition groups must be separated if claiming unseen-object generalization.',
        'Retry attempts are outcome-censored (stop at success), and adaptive methods depend on previous attempts; not independent matched episodes or fixed exposure counts.',
        'No new training is authorized after proven leakage until user chooses a safe subset/feature and behavior-policy protocol.']
    result['source_hashes']={s:digest(ROOT/s) for s in ['robot_adaptation/collection/schema.py','robot_adaptation/collection/recording.py','robot_adaptation/collection/collect.py','robot_adaptation/collection/design.py','robot_adaptation/collection/varied.py','robot_adaptation/collection/stock.py','robot_adaptation/simulation/collection_lift.py','robot_adaptation/simulation/training_lift.py','experiments/interaction_adaptation/evaluate.py','experiments/interaction_adaptation/transfer.py','experiments/physics_conditioning/evaluate.py','experiments/physics_conditioning/checkpoint_probe.py','robot_adaptation/policy/examples/maniskill_pick_cube/run_pretrained.py']}
    result['summary_counts']={'raw_npz_episodes':sum(x['episode_count'] for x in result['raw_npz_families']),'raw_npz_families':len(result['raw_npz_families']),'pretrained_jsonl_episodes':sum(x['episode_count'] for x in result['pretrained_episode_families']),'gpu_trace_episodes':sum(x.get('episode_count',0) for x in result['jsonl_artifacts']),'jsonl_artifact_files':len(result['jsonl_artifacts'])}
    supplement(result)
    result['audit_source_sha256']=digest(Path(__file__))
    out=ROOT/'results/probe/inventory_other.json'
    out.write_text(json.dumps(result,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
    print('Wrote',out,'counts',result['summary_counts'],flush=True)

if __name__=='__main__':main()
