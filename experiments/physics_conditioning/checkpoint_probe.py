"""CPU-only matched oracle-input interventions for frozen46-input policies.

The probe-local task preserves native PickCube goalXY and the saved training
floating-Z RNG rule. Every attempt rebuilds its scene. Simulator ground truth
is used deliberately for controlled policy-input interventions, not estimation.
"""
from __future__ import annotations

import os
for _thread_key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','LP_NUM_THREADS'):
    os.environ[_thread_key] = '1'

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.structs.pose import Pose

from experiments.interaction_adaptation.evaluate import supplied_observation
from experiments.physics_conditioning.evaluate import MODES,inject_parameters
from robot_adaptation.collection.collect import rollout,write_json,runtime_metadata,digest
from robot_adaptation.collection.provenance import source_hashes,snapshot_sources
from robot_adaptation.collection.schema import save_episode
from robot_adaptation.paths import ROOT
from robot_adaptation.policy.train import load_checkpoint
from robot_adaptation.simulation.collection_lift import WeightedCollectionEnv
from robot_adaptation.simulation.environment import CollectionEnv
from robot_adaptation.simulation.training_lift import reward_components

TRAINING_MASSES = (.04,.064,.1,.25,.5)
TRAINING_FRICTIONS = (.2,.5,1.)
INTERMEDIATE_MASSES = (.086,.132,.316)
INTERMEDIATE_FRICTIONS = (.28,.64,.91)
HORIZON = 50
SUSTAINED_STEPS = 10
LIFT_CLEARANCE_M = .03
TASK_SEMANTICS = 'CPU native PickCube goalXY with training floating goalZ and training reward; fresh scene every attempt'


def default_grids():
    return {name:[dict(mass_kg=mass,friction=mu) for mass in masses for mu in frictions]
            for name,masses,frictions in (
                ('training',TRAINING_MASSES,TRAINING_FRICTIONS),
                ('intermediate',INTERMEDIATE_MASSES,INTERMEDIATE_FRICTIONS))}


class ProbeCarryTask(WeightedCollectionEnv):
    """Recorder-compatible carrying task; no canonical simulator is modified."""
    collection_description = TASK_SEMANTICS

    def __init__(self,properties,*,reported_mass_kg,goal_clearance_m,grip_force_limits_n,grip_force_cost):
        self.goal_clearance_range_m = tuple(float(x) for x in goal_clearance_m)
        if (len(self.goal_clearance_range_m)!=2 or not np.isfinite(self.goal_clearance_range_m).all()
                or not .025 < self.goal_clearance_range_m[0] <= self.goal_clearance_range_m[1]):
            raise ValueError('Use the saved ordered floating-goal clearance range')
        self.grip_force_cost = float(grip_force_cost)
        if not np.isfinite(self.grip_force_cost) or self.grip_force_cost < 0:
            raise ValueError('grip_force_cost must be finite and nonnegative')
        super().__init__(properties,reported_mass_kg=reported_mass_kg,
                         goal_clearance_m=self.goal_clearance_range_m[0],
                         grip_force_limits_n=grip_force_limits_n)

    def _initialize_episode(self,env_idx,options):
        # Bypass WeightedCollectionEnv's vertical-goal override; preserve the
        # exact native goalXY RNG calls and training goalZ RNG call/order.
        CollectionEnv._initialize_episode(self,env_idx,options)
        self.agent.controller.controllers['gripper'].reset_env_idx = env_idx.clone()
        goal = self.goal_site.pose.p[env_idx].clone()
        low,high = self.goal_clearance_range_m
        goal[:,2] = self.cube_half_size + low + (high-low)*torch.rand(len(env_idx),device=self.device)
        self.goal_site.set_pose(Pose.create_from_pq(goal))

    def measured_compression(self):
        values=[]
        for link,sign in ((self.agent.finger1_link,1),(self.agent.finger2_link,-1)):
            force=self.scene.get_pairwise_contact_forces(link,self.cube)
            axis=sign*link.pose.to_transformation_matrix()[:,:3,1]
            value=(force*axis).sum(-1).clamp_min(0)
            values.append(torch.zeros_like(value) if self._contacts_stale else value)
        return torch.stack(values,dim=-1)

    def evaluate(self):
        info=super().evaluate()
        distance=(self.cube.pose.p-self.agent.tcp_pose.p).norm(dim=-1)
        goal_distance=(self.cube.pose.p-self.goal_site.pose.p).norm(dim=-1)
        speed=self.agent.robot.get_qvel()[...,:-2].norm(dim=-1)
        rotation=quaternion_to_matrix(self.cube.pose.q)
        clearance=self.cube.pose.p[:,2]-self.cube_half_size*rotation[:,2].abs().sum(-1)
        parts=reward_components(distance,goal_distance,info['is_grasped'],speed,
                                info['is_obj_placed'],info['is_robot_static'])
        forces=self.measured_compression()
        cost=self.grip_force_cost*(forces.mean(-1)/self.grip_force_limits_n[1]).clamp(0,1)
        return dict(info,distance_m=distance,goal_distance_m=goal_distance,clearance_m=clearance,
                    held_goal_success=info['success'],grip_force_cost=cost,
                    reward_total=parts['task_score']-cost,
                    grip_limit_n=self.agent.controller.controllers['gripper'].force_limit_n.clone(),**parts)

    def compute_normalized_dense_reward(self,obs,action,info):
        return info['reward_total']

    def compute_dense_reward(self,obs,action,info):
        return 5*info['reward_total']


def supplied_parameters(mode,mass_kg,friction):
    """Exactly the six existing GPU intervention semantics, in kg and mu."""
    if mode not in MODES:
        raise ValueError('Unknown intervention mode: '+str(mode))
    mass,mu=float(mass_kg),float(friction)
    if mode in ('fixed_mass','fixed_both'):mass=.04
    elif mode=='wrong_mass':mass=.54-mass
    if mode in ('fixed_friction','fixed_both'):mu=.5
    elif mode=='wrong_friction':mu=1.2-mu
    if not np.isfinite([mass,mu]).all() or mass<=0 or mu<=0:
        raise ValueError('Intervention parameters must remain positive and finite')
    return mass,mu


def policy_observation(observation,mass_kg,friction,gravity):
    """Only supplied weight index42 and appended friction index45 may change."""
    appended=supplied_observation(observation,friction)
    return inject_parameters(appended,mass_kg=mass_kg,friction=friction,gravity=gravity)


def extract_metrics(data):
    """Post-action endpoint metrics; T command intervals align to states1..T.

    Compression uses the policy's measured pairwise endpoint sensors43/44,
    matching the GPU evaluator. Interval-mean all-contact force is preserved
    separately in raw recorder fields and an optional supporting statistic.
    """
    actions=np.asarray(data['input__action_applied'])
    count=len(actions)
    grasp=np.asarray(data['sim__is_grasped'],dtype=bool)
    clearance=np.asarray(data['sim__object_clearance_m'],dtype=float)
    success=np.asarray(data['sim__task_success'],dtype=bool)
    observation=np.asarray(data['sim__policy_observation'],dtype=float)
    limits=np.asarray(data['input__grip_motor_limit_n'],dtype=float)
    rewards=np.asarray(data['sim__reward'],dtype=float)
    if (count<SUSTAINED_STEPS or grasp.shape!=(count+1,) or clearance.shape!=(count+1,)
            or success.shape!=(count+1,) or observation.shape!=(count+1,45)
            or limits.shape!=(count,2) or rewards.shape!=(count,)):
        raise ValueError('Episode state/transition timelines are not aligned')
    compression=observation[1:,43:45]
    if not all(np.isfinite(x).all() for x in (clearance,compression,limits,rewards)) or (compression < 0).any():
        raise ValueError('Nonfinite/negative measured force or episode metric')
    lifted=grasp[1:]&(clearance[1:]>LIFT_CLEARANCE_M)
    lifted_steps=int(lifted.sum())
    ever_lifted=np.maximum.accumulate(lifted)
    lost=bool(np.any(ever_lifted&~grasp[1:]))
    final=bool(success[-1] and grasp[-1])
    if final!=bool(np.asarray(data['label__held_goal_success_final'])):
        raise ValueError('Held-goal final label disagrees with state timeline')
    force_count=2*lifted_steps
    result=dict(final_success=final,sustained_success=bool(np.all(success[1:][-SUSTAINED_STEPS:]&grasp[1:][-SUSTAINED_STEPS:])),
                grasp_lost_after_lift=lost,lifted_ever=bool(lifted_steps),lifted_steps=lifted_steps,
                lifted_force_n=float(compression[lifted].sum()/force_count) if force_count else None,
                lifted_motor_limit_n=float(limits[lifted].sum()/force_count) if force_count else None,
                lifted_force_sum_n=float(compression[lifted].sum()),
                lifted_motor_limit_sum_n=float(limits[lifted].sum()),
                force_sample_count=force_count,force_step_count=lifted_steps,
                force_samples_per_step=2,reward_sum=float(rewards.sum()),
                success_any=bool(success[1:].any()),
                sustained_post_action_steps=SUSTAINED_STEPS)
    frequency=float(np.asarray(data['meta__control_frequency_hz']))
    if not np.isfinite(frequency) or frequency<=0:raise ValueError('Control frequency must be positive and finite')
    result['lifted_exposure_s']=lifted_steps/frequency
    result['sustained_exposure_s']=SUSTAINED_STEPS/frequency
    if 'input__finger_compression_mean_n' in data:
        interval=np.asarray(data['input__finger_compression_mean_n'],dtype=float)
        if interval.shape!=(count,2) or not np.isfinite(interval).all():
            raise ValueError('Invalid interval compression timeline')
        result['lifted_interval_force_n']=float(interval[lifted].sum()/force_count) if force_count else None
    return result


def assert_physics(physical,mass,mu):
    expected=[mass,mu,mu]
    observed=[physical['mass_kg'],physical['object_materials'][0]['static_friction'],physical['object_materials'][0]['dynamic_friction']]
    if not np.allclose(observed,expected,rtol=1e-6,atol=1e-8):
        raise RuntimeError('Actual mass/object material differs from the declared condition')
    if not np.allclose(physical['size_xyz_m'],[.04]*3,rtol=1e-6,atol=1e-8):
        raise RuntimeError('Object geometry differs from the fixed4cm cube')
    for materials,mu_fixed in ((physical['finger_materials'],.2),([physical['table_materials']],.3)):
        for shapes in materials:
            for shape in shapes:
                if not np.allclose([shape['static_friction'],shape['dynamic_friction']],[mu_fixed,mu_fixed],rtol=1e-6,atol=1e-8):
                    raise RuntimeError('Finger/table friction changed across interventions')


def source_extras():
    paths=[Path(__file__),ROOT/'experiments/physics_conditioning/evaluate.py',
           ROOT/'experiments/interaction_adaptation/evaluate.py',
           ROOT/'robot_adaptation/policy/model.py',ROOT/'robot_adaptation/policy/train.py',
           ROOT/'robot_adaptation/policy/trust_region.py',
           ROOT/'robot_adaptation/policy/examples/maniskill_pick_cube/ppo_upstream.py']
    paths.extend((ROOT/'robot_adaptation/estimators/interaction').glob('*.py'))
    return paths


def checkpoint_cycles(checkpoint,manifest):
    if checkpoint.stem.startswith('ckpt_') and checkpoint.stem[5:].isdigit():
        iteration=int(checkpoint.stem[5:])
        return iteration,max(iteration-1,0)
    if checkpoint.name=='final_ckpt.pt' and manifest.get('status')=='complete':
        return None,manifest.get('planned_iterations')
    return None,None


def run_probe(checkpoint,label,output,*,seed_start=73000,seed_count=10,threads=1,grids=None):
    """Run one checkpoint. Optional small grids exist for smoke checks only."""
    if threads!=1:raise ValueError('Each checkpoint probe must use exactly one CPU thread')
    if type(seed_count) is not int or seed_count<1 or type(seed_start) is not int:
        raise ValueError('Use an integer diagnostic seed and positive seed count')
    seeds=list(range(seed_start,seed_start+seed_count))
    if seed_start<0 or seeds[-1]>=2**31 or any(91000<=s<92000 or 54000<=s<55000 or 64000<=s<65000 for s in seeds):
        raise ValueError('Use diagnostic seeds, excluding the reserved final-test ranges')
    if not isinstance(label,str) or not label.strip():raise ValueError('Provide a nonempty checkpoint label')
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass # The embedding process may already have run Torch work.
    checkpoint=Path(checkpoint).expanduser().resolve()
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    policy,policy_manifest=load_checkpoint(checkpoint)
    if policy_manifest.get('observations')!=46 or policy_manifest.get('actions')!=5:
        raise ValueError('Checkpoint probe requires46-input/five-action weights')
    policy.cpu().eval().requires_grad_(False)
    if any(not bool(torch.isfinite(p).all()) for p in policy.state_dict().values()):
        raise ValueError('Checkpoint contains nonfinite model state')
    if policy.actor_mean[0].weight.shape[1]!=46 or policy.strength_mean[0].weight.shape[1]!=46:
        raise ValueError('Checkpoint model tensor interface differs from46inputs')
    settings=policy_manifest['settings']
    planned_grids=default_grids() if grids is None else grids
    if not planned_grids or any(not conditions for conditions in planned_grids.values()):
        raise ValueError('Provide nonempty physical grids')
    planned_grids={str(name):[dict(mass_kg=float(c['mass_kg']),friction=float(c['friction'])) for c in conditions] for name,conditions in planned_grids.items()}
    for conditions in planned_grids.values():
        for condition in conditions:supplied_parameters('true',condition['mass_kg'],condition['friction'])
    output=Path(output).expanduser().resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'episodes').mkdir()
    checkpoint_iteration,completed_cycles=checkpoint_cycles(checkpoint,policy_manifest)
    expected_rows=sum(len(c) for c in planned_grids.values())*len(seeds)*len(MODES)
    plan=dict(grids=planned_grids,seeds=seeds,modes=list(MODES),episode_steps=HORIZON,expected_rows=expected_rows)
    hashes=source_hashes(source_extras());snapshot_sources(output/'source',source_extras())
    manifest=dict(schema_version='cpu_checkpoint_probe_v1',status='running',checkpoint=str(checkpoint),
                  checkpoint_label=label,checkpoint_sha256=checkpoint_hash,checkpoint_iteration=checkpoint_iteration,
                  completed_rollout_cycles=completed_cycles,policy_manifest=policy_manifest,plan=plan,
                  planned_rows=expected_rows,completed_rows=0,diagnostic_subset=grids is not None,
                  task_semantics=TASK_SEMANTICS,goal_clearance_m=settings['goal_clearance_m'],
                  actual_physics=dict(finger_friction=.2,table_friction=.3,cube_size_m=.04,static_equals_dynamic=True),
                  source_sha256=hashes,runtime=runtime_metadata(),cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LP_NUM_THREADS')},
                  force_definition='Pairwise finger-object measured compression at each post-control endpoint, averaged across two fingers and qualifying grasped clearance>.03m steps; conditional on lift',
                  sustained_definition='All final10 post-action success+grasp samples;10 control intervals=.5s at20Hz',
                  grasp_loss_definition='Any loss of grasp detector after a grasped clearance>.03m sample; descriptive release/drop flag, not causal slip')
    write_json(output/'manifest.json',manifest)
    rows=[];started=time.monotonic()
    try:
        with (output/'rows.jsonl').open('x') as row_stream:
            for grid,conditions in planned_grids.items():
                for condition_index,condition in enumerate(conditions):
                    mass,mu=condition['mass_kg'],condition['friction']
                    properties=dict(mass_kg=mass,static_friction=mu,dynamic_friction=mu,finger_friction=.2,table_friction=.3,size_xyz_m=[.04]*3)
                    env=ProbeCarryTask(properties,reported_mass_kg=mass,goal_clearance_m=settings['goal_clearance_m'],grip_force_limits_n=settings['grip_force_limits_n'],grip_force_cost=settings['grip_force_cost'])
                    try:
                        if env.gpu_sim_enabled or torch.device(env.device).type!='cpu':raise RuntimeError('CPU-only probe constructed a non-CPU simulator')
                        gravity=float(np.linalg.norm(env.scene.px.get_config().gravity))
                        for seed in seeds:
                            expected_reset=None;expected_physics=None
                            for mode in MODES:
                                supplied_mass,supplied_mu=supplied_parameters(mode,mass,mu)
                                env.reported_mass_kg=supplied_mass
                                supplied_trace=[]
                                def act(observation):
                                    vector=policy_observation(observation,supplied_mass,supplied_mu,gravity)
                                    if not torch.equal(vector[:,:45],observation):
                                        raise RuntimeError('Supplied mass was not injected consistently by the environment')
                                    supplied_trace.append(vector[0].detach().cpu().numpy().copy())
                                    return policy.get_action(vector,deterministic=True)
                                with torch.inference_mode():
                                    data,_=rollout(env,act,seed,dict(episode_steps=HORIZON,contact_threshold_n=.05))
                                physical=json.loads(str(data['meta__physics_readback_json']))
                                assert_physics(physical,mass,mu)
                                reset=str(data['label__reset_state_hash'])
                                physics_fingerprint=digest(dict(physical=physical,controller=json.loads(str(data['meta__controller_config_json'])),simulation=json.loads(str(data['meta__simulation_config_json']))))
                                if expected_reset is None:expected_reset,expected_physics=reset,physics_fingerprint
                                if reset!=expected_reset or physics_fingerprint!=expected_physics:
                                    raise RuntimeError('Initial physical state/material/controller changed between input modes')
                                episode_id=f'{grid}_condition{condition_index:02d}_seed{seed}_{mode}'
                                trace=np.stack(supplied_trace)
                                supplied=dict(mass_kg=supplied_mass,friction=supplied_mu,weight_n=float(trace[0,42]),friction_float32=float(trace[0,45]),gravity_m_s2=gravity)
                                data.update(probe__supplied_policy_inputs=trace,probe__checkpoint_label=np.asarray(label),probe__checkpoint_sha256=np.asarray(checkpoint_hash),probe__mode=np.asarray(mode),probe__grid=np.asarray(grid),probe__reset_fingerprint=np.asarray(reset),probe__supplied_parameters_json=np.asarray(json.dumps(supplied,sort_keys=True)))
                                metrics=extract_metrics(data)
                                save_episode(output/'episodes'/(episode_id+'.npz'),data)
                                row=dict(checkpoint_label=label,grid=grid,mass_kg=mass,friction=mu,seed=seed,mode=mode,episode_id=episode_id,reset_fingerprint=reset,physics_fingerprint=physics_fingerprint,
                                         supplied_mass_kg=supplied_mass,supplied_friction=supplied_mu,supplied_weight_n=supplied['weight_n'],supplied_parameters=supplied,actual_physics=physical,**metrics)
                                row_stream.write(json.dumps(row,allow_nan=False)+'\n');row_stream.flush();rows.append(row)
                            print(json.dumps(dict(checkpoint_label=label,grid=grid,condition_index=condition_index,seed=seed,completed_rows=len(rows),planned_rows=expected_rows)),flush=True)
                    finally:env.close()
        summary={}
        for grid in planned_grids:
            summary[grid]={}
            for mode in MODES:
                selected=[r for r in rows if r['grid']==grid and r['mode']==mode]
                lifted=[r for r in selected if r['lifted_steps']>0]
                summary[grid][mode]=dict(episodes=len(selected),final_successes=sum(r['final_success'] for r in selected),final_success=float(np.mean([r['final_success'] for r in selected])),sustained_success=float(np.mean([r['sustained_success'] for r in selected])),episodes_with_lifted_contact=len(lifted),conditional_force_episode_mean_n=float(np.mean([r['lifted_force_n'] for r in lifted])) if lifted else None,total_lifted_steps=sum(r['lifted_steps'] for r in selected),total_force_samples=sum(r['force_sample_count'] for r in selected))
        result=dict(checkpoint_label=label,checkpoint_sha256=checkpoint_hash,completed_rollout_cycles=completed_cycles,rows=len(rows),planned_rows=expected_rows,metrics=summary,all_modes_matched_initial_physics=True,elapsed_seconds=time.monotonic()-started,scope='Diagnostic CPU native carry task, not final reserved assessment; force statistics are conditional on successful lifted contact')
        write_json(output/'summary.json',result)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=checkpoint_hash:raise RuntimeError('Checkpoint bytes changed during inference')
        if source_hashes(source_extras())!=hashes:raise RuntimeError('Probe source changed during collection; cross-checkpoint comparison would be ambiguous')
        manifest.update(status='complete',completed_rows=len(rows),elapsed_seconds=result['elapsed_seconds'])
        write_json(output/'manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='failed',completed_rows=len(rows),error=repr(error))
        write_json(output/'manifest.json',manifest)
        raise
    print('CPU checkpoint probe saved:',output,flush=True)
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed-start',type=int,default=73000)
    parser.add_argument('--seed-count',type=int,default=10)
    parser.add_argument('--threads',type=int,choices=(1,),default=1)
    args=parser.parse_args(argv)
    return run_probe(args.checkpoint,args.label,args.output,seed_start=args.seed_start,seed_count=args.seed_count,threads=args.threads)


if __name__=='__main__':main()
