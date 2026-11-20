"""Direct COM force on a free cube; no pusher."""
from types import SimpleNamespace
import numpy as np
import sapien
import sapien.physx as px
from clean.native import set_cube_properties
from clean.shove_gate.script import action,PHYSICS_DT,MODEL_DT,COAST_S,SETTLE_S,HALF_SIDE,TRIM_AFTER_STOP_STEPS
STATE_COLUMNS=(['cube_x','cube_y','cube_z']+[f'cube_rotation_{i}' for i in range(6)]+
               ['cube_vx','cube_vy','cube_vz','cube_wx','cube_wy','cube_wz'])

def simulate(properties,plan):
    px.set_scene_config(gravity=[0,0,-9.81],enable_enhanced_determinism=True,enable_tgs=True,enable_pcm=True)
    scene=sapien.Scene([px.PhysxCpuSystem()]);scene.set_timestep(PHYSICS_DT)
    scene.add_ground(0,render=False,material=px.PhysxMaterial(.3,.3,0.))
    b=scene.create_actor_builder()
    b.add_box_collision(half_size=[HALF_SIDE]*3,material=px.PhysxMaterial(.5,.5,0.))
    yaw=plan['yaw_rad'];b.initial_pose=sapien.Pose([*plan['start_xy'],HALF_SIDE],[np.cos(yaw/2),0,0,np.sin(yaw/2)])
    cube=b.build(name='cube');body=cube.find_component_by_type(px.PhysxRigidDynamicComponent)
    set_cube_properties(SimpleNamespace(cube=SimpleNamespace(_bodies=[body]),cube_half_size=HALF_SIDE),properties)
    body.set_sleep_threshold(0);body.set_linear_damping(0);body.set_angular_damping(0)
    for _ in range(round(SETTLE_S/PHYSICS_DT)):scene.step()
    def observe():
        pose=cube.pose
        return np.r_[pose.p,pose.to_transformation_matrix()[:3,:2].T.ravel(),body.linear_velocity,body.angular_velocity].astype(np.float64)
    states=[observe()];actions=[];phases=[]
    pulse=action(plan,'push');zero=action(plan,'coast')
    release=plan['duration_steps']*round(MODEL_DT/PHYSICS_DT)
    for step in range(release+round(COAST_S/PHYSICS_DT)):
        phase=0 if step<release else 1
        applied=pulse if phase==0 else zero
        body.add_force_torque(applied,[0,0,0])
        scene.step()
        states.append(observe());actions.append(applied);phases.append(phase)
    raw=dict(state=np.asarray(states),action=np.asarray(actions),phase=np.asarray(phases,dtype=np.int8),
             release_index=np.array(release),physics_dt_s=np.array(PHYSICS_DT))
    stride=round(MODEL_DT/PHYSICS_DT);selected=raw['state'][::stride].astype(np.float32)
    n=len(selected)-1;phase=np.where(raw['phase'][::stride]==0,'push','coast')
    episode=dict(cube_id=np.full(n,plan['cube_id'],dtype=np.int16),episode_id=np.full(n,plan['episode_id'],dtype=np.int16),
                 step=np.arange(n,dtype=np.int32),phase=phase,state=selected[:-1],action=raw['action'][::stride],next_state=selected[1:])
    return episode,raw

def measure(raw,plan):
    states=raw['state'];r=int(raw['release_index'])
    resting=(np.linalg.norm(states[:,9:12],axis=1)<.001)&(np.linalg.norm(states[:,12:15],axis=1)<.05)
    candidates=np.flatnonzero(np.convolve(resting[r+1:].astype(int),np.ones(100,dtype=int),mode='valid')==100)
    stop=r+1+int(candidates[0]) if len(candidates) else None
    if stop==r+1 and resting[r]:stop=r
    end=stop if stop is not None else len(states)-1
    normal=np.cross(states[:,3:6],states[:,6:9]);tilt=np.degrees(np.arccos(np.clip(normal[:,2],-1,1)))
    return dict(stopped=stop is not None,stop_raw_index=stop,release_raw_index=r,
                time_to_stop_s=(end-r)*PHYSICS_DT,launch_speed_m_s=float(np.linalg.norm(states[r,9:11])),
                stopping_distance_m=float(np.linalg.norm(states[end,:2]-states[r,:2])),stopping_xy_m=states[end,:2].tolist(),
                max_applied_force_n=float(np.linalg.norm(raw['action'],axis=1).max()),
                max_coast_force_n=float(np.linalg.norm(raw['action'][r:],axis=1).max()),
                max_tilt_deg=float(tilt.max()),min_z_m=float(states[:,2].min()),max_z_m=float(states[:,2].max()),
                terminal_speed_m_s=float(np.linalg.norm(states[-1,9:12])),terminal_angular_speed_rad_s=float(np.linalg.norm(states[-1,12:15])))

def trim_episode(episode,raw,metrics):
    n=len(episode['step']);stride=round(MODEL_DT/PHYSICS_DT)
    keep=min(n,int(np.ceil(metrics['stop_raw_index']/stride))+TRIM_AFTER_STOP_STEPS) if metrics['stopped'] else n
    trimmed={key:value[:keep] for key,value in episode.items()}
    moving=(np.linalg.norm(episode['state'][:,9:12],axis=1)>.001)|(np.linalg.norm(episode['state'][:,12:15],axis=1)>.05)
    coast=episode['phase']=='coast'
    return trimmed,dict(untrimmed_steps=n,retained_steps=keep,trimmed_steps=n-keep,
                        retained_coast_steps=int(coast[:keep].sum()),retained_moving_coast_steps=int((coast[:keep]&moving[:keep]).sum()),
                        untrimmed_coast_steps=int(coast.sum()),untrimmed_moving_coast_steps=int((coast&moving).sum()))
