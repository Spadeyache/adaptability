"""Free-XYZ physical pusher; full contact force from measured momentum change."""
from types import SimpleNamespace
import numpy as np
import sapien
import sapien.physx as px
from clean.native import set_cube_properties
from clean.shove_gate.script import (geometry,action,PHYSICS_DT,MODEL_DT,APPROACH_S,PRELOAD_S,
                                    PUSH_S,COAST_S,PUSHER_MASS,PUSHER_RADIUS,HALF_SIDE,TRIM_AFTER_STOP_STEPS)

STATE_COLUMNS=(['cube_x','cube_y','cube_z']+
               [f'cube_rotation_{i}' for i in range(6)]+
               ['cube_vx','cube_vy','cube_vz','cube_wx','cube_wy','cube_wz',
                'pusher_x','pusher_y','pusher_z','pusher_vx','pusher_vy','pusher_vz'])


def simulate(properties,plan):
    px.set_scene_config(gravity=[0,0,-9.81],enable_enhanced_determinism=True,enable_tgs=True,enable_pcm=True)
    scene=sapien.Scene([px.PhysxCpuSystem()]);scene.set_timestep(PHYSICS_DT)
    scene.add_ground(0,render=False,material=px.PhysxMaterial(.3,.3,0.))
    g=geometry(plan);yaw=plan['yaw_rad']
    b=scene.create_actor_builder()
    b.add_box_collision(half_size=[HALF_SIDE]*3,material=px.PhysxMaterial(.5,.5,0))
    b.initial_pose=sapien.Pose(g['cube_center'],[np.cos(yaw/2),0,0,np.sin(yaw/2)])
    cube=b.build(name='cube');body=cube.find_component_by_type(px.PhysxRigidDynamicComponent)
    set_cube_properties(SimpleNamespace(cube=SimpleNamespace(_bodies=[body]),cube_half_size=HALF_SIDE),properties)
    b=scene.create_actor_builder()
    b.add_sphere_collision(radius=PUSHER_RADIUS,material=px.PhysxMaterial(0,0,0))
    b.initial_pose=sapien.Pose(g['pusher_start'])
    pusher=b.build(name='pusher');pb=pusher.find_component_by_type(px.PhysxRigidDynamicComponent)
    pb.set_mass(PUSHER_MASS);pb.set_disable_gravity(True)
    pb.set_locked_motion_axes([False,False,False,True,True,True])
    for obj in [body,pb]:
        obj.set_sleep_threshold(0);obj.set_linear_damping(0);obj.set_angular_damping(0)
    pb.set_linear_velocity(.005*g['direction'])
    def observe():
        pose=cube.pose;rotation=pose.to_transformation_matrix()[:3,:2].T.ravel()
        return np.r_[pose.p,rotation,body.linear_velocity,body.angular_velocity,pusher.pose.p,pb.linear_velocity].astype(np.float64)
    states=[observe()];forces=[];impulses=[];counts=[];commands=[];phases=[]
    transmitted=[];reported=[];other_contacts=[];other_impulses=[];before_velocities=[]
    def step(phase,force):
        before=np.asarray(pb.linear_velocity,dtype=np.float64).copy()
        if phase!=3:pb.add_force_torque(force,[0,0,0])
        scene.step()
        net=np.zeros(3);bound=0.;count=0;other=0;other_impulse=0.
        for contact in scene.physx_system.get_contacts():
            if pb in contact.bodies and body not in contact.bodies:
                for point in contact.points:
                    magnitude=float(np.linalg.norm(point.impulse))
                    other_impulse+=magnitude
                    other+=int(magnitude>1e-12)
            if body in contact.bodies and pb in contact.bodies:
                for point in contact.points:
                    impulse=np.asarray(point.impulse,dtype=np.float64)
                    net+=impulse;bound+=float(np.linalg.norm(impulse));count+=1
        # PhysX extractContacts exports normal impulse only. The pusher has
        # no translational constraints, gravity or damping, so its measured
        # momentum change gives the full (normal+tangential) transmitted force.
        actual=np.asarray(force)-PUSHER_MASS*(np.asarray(pb.linear_velocity)-before)/PHYSICS_DT
        transmitted.append(actual);before_velocities.append(before)
        reported.append(bound/PHYSICS_DT);other_contacts.append(other);other_impulses.append(other_impulse)
        states.append(observe());forces.append(float(np.linalg.norm(actual)));impulses.append(net)
        counts.append(count);commands.append(force);phases.append(phase)
    for phase,duration,magnitude in [(0,APPROACH_S,0.),(1,PRELOAD_S,.03),(2,PUSH_S,.95*plan['cap_n'])]:
        for _ in range(round(duration/PHYSICS_DT)):step(phase,magnitude*g['direction'])
    release=len(phases)
    # Preplanned withdrawal is an action, not a target derived from object motion.
    pb.set_linear_velocity([0,0,0]);pb.set_kinematic(True)
    pusher.set_pose(sapien.Pose(g['park']))
    for _ in range(round(COAST_S/PHYSICS_DT)):step(3,np.zeros(3))
    raw=dict(state=np.asarray(states),phase=np.asarray(phases,dtype=np.int8),
             command_force=np.asarray(commands),reported_normal_impulse=np.asarray(impulses),
             reported_normal_force_n=np.asarray(reported),transmitted_force_n=np.asarray(transmitted),
             pusher_step_velocity_before=np.asarray(before_velocities),
             other_pusher_contact_count=np.asarray(other_contacts,dtype=np.int16),
             other_pusher_normal_impulse_norm=np.asarray(other_impulses),
             contact_force_n=np.asarray(forces),contact_point_count=np.asarray(counts,dtype=np.int16),
             release_index=np.array(release),physics_dt_s=np.array(PHYSICS_DT))
    start=round((APPROACH_S+PRELOAD_S)/PHYSICS_DT)
    stride=round(MODEL_DT/PHYSICS_DT)
    selected=np.asarray(states)[start::stride].astype(np.float32)
    phase=np.array(['push']*round(PUSH_S/MODEL_DT)+['coast']*round(COAST_S/MODEL_DT))
    n=len(phase)
    episode=dict(cube_id=np.full(n,plan['cube_id'],dtype=np.int16),
                 episode_id=np.full(n,plan['episode_id'],dtype=np.int16),step=np.arange(n,dtype=np.int32),
                 phase=phase,state=selected[:-1],action=np.stack([action(plan,p) for p in phase]),next_state=selected[1:])
    return episode,raw


def trim_episode(episode,raw,metrics):
    """Keep five model steps after confirmed rest onset; never trim diagnostics.

    Stop confirmation uses the full raw coast. The retained sequence length is
    an outcome, not a command, and must never enter prediction-model features.
    """
    if not metrics['stopped']:
        return episode,dict(untrimmed_steps=len(episode['step']),retained_steps=len(episode['step']),trimmed_steps=0)
    start=round((APPROACH_S+PRELOAD_S)/PHYSICS_DT)
    stride=round(MODEL_DT/PHYSICS_DT)
    keep=min(len(episode['step']),int(np.ceil((metrics['stop_raw_index']-start)/stride))+TRIM_AFTER_STOP_STEPS)
    trimmed={name:array[:keep] for name,array in episode.items()}
    coast=trimmed['phase']=='coast'
    moving=(np.linalg.norm(trimmed['state'][:,9:12],axis=1)>.001)|(np.linalg.norm(trimmed['state'][:,12:15],axis=1)>.05)
    full_coast=episode['phase']=='coast'
    full_moving=(np.linalg.norm(episode['state'][:,9:12],axis=1)>.001)|(np.linalg.norm(episode['state'][:,12:15],axis=1)>.05)
    return trimmed,dict(untrimmed_steps=len(episode['step']),retained_steps=keep,
                        trimmed_steps=len(episode['step'])-keep,retained_coast_steps=int(coast.sum()),
                        retained_moving_coast_steps=int(np.sum(coast&moving)),
                        untrimmed_coast_steps=int(full_coast.sum()),untrimmed_moving_coast_steps=int(np.sum(full_coast&full_moving)))


def measure(raw,plan):
    states=raw['state'];r=int(raw['release_index']);forces=raw['contact_force_n']
    resting=(np.linalg.norm(states[:,9:12],axis=1)<.001)&(np.linalg.norm(states[:,12:15],axis=1)<.05)
    windows=np.convolve(resting[r+1:].astype(int),np.ones(100,dtype=int),mode='valid')
    candidates=np.flatnonzero(windows==100)
    s=r+1+int(candidates[0]) if len(candidates) else None
    if s==r+1 and resting[r]:s=r
    end=s if s is not None else len(states)-1
    normals=np.cross(states[:,3:6],states[:,6:9])
    tilt=np.degrees(np.arccos(np.clip(normals[:,2],-1,1)))
    return dict(stopped=s is not None,stop_raw_index=s,release_raw_index=r,
                time_to_stop_s=(end-r)*PHYSICS_DT,
                launch_speed_m_s=float(np.linalg.norm(states[r,9:11])),
                stopping_distance_m=float(np.linalg.norm(states[end,:2]-states[r,:2])),
                stopping_xy_m=states[end,:2].tolist(),
                max_contact_force_n=float(forces.max()),cap_ratio=float(forces.max()/plan['cap_n']),
                max_coast_contact_force_n=float(forces[r:].max()),
                other_pusher_contact_count=int(raw['other_pusher_contact_count'].sum()),
                push_contact_fraction=float(np.mean(forces[raw['phase']==2]>1e-4)),
                max_tilt_deg=float(tilt.max()),min_z_m=float(states[:,2].min()),max_z_m=float(states[:,2].max()),
                terminal_speed_m_s=float(np.linalg.norm(states[-1,9:12])),
                terminal_angular_speed_rad_s=float(np.linalg.norm(states[-1,12:15])))
