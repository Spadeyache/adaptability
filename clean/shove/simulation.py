"""Physical actuator and sensor-only trajectory collector, with no learned model."""
from types import SimpleNamespace
import numpy as np
import sapien
import sapien.physx as px
from clean.native import set_cube_properties
from clean.shove.script import commands

STATE_COLUMNS=['cube_x','cube_y','cube_z','cube_vx','cube_vy','cube_vz',
               'cube_wx','cube_wy','cube_wz','cube_qw','cube_qx','cube_qy','cube_qz',
               'pusher_x','pusher_y','pusher_z','pusher_vx','pusher_vy','pusher_vz']


def simulate(properties, settings):
    """Only the setup setter consumes properties. Program and stopping use sensors."""
    px.set_scene_config(gravity=[0,0,-9.81], enable_enhanced_determinism=True,
                        enable_tgs=True, enable_pcm=True)
    scene=sapien.Scene([px.PhysxCpuSystem()])
    scene.set_timestep(settings.dt)
    scene.add_ground(0,render=False,material=px.PhysxMaterial(.3,.3,0))
    b=scene.create_actor_builder()
    b.add_box_collision(half_size=[.02]*3,material=px.PhysxMaterial(.5,.5,0))
    b.initial_pose=sapien.Pose([0,0,.02])
    cube=b.build(name='cube')
    body=cube.find_component_by_type(px.PhysxRigidDynamicComponent)
    set_cube_properties(SimpleNamespace(cube=SimpleNamespace(_bodies=[body]),cube_half_size=.02),properties)
    b=scene.create_actor_builder()
    b.add_sphere_collision(radius=.008,material=px.PhysxMaterial(0,0,0))
    b.initial_pose=sapien.Pose([-.031,0,.02])
    pusher=b.build(name='pusher')
    pb=pusher.find_component_by_type(px.PhysxRigidDynamicComponent)
    pb.set_mass(.005)
    pb.set_disable_gravity(True)
    pb.set_locked_motion_axes([False,True,True,True,True,True])
    for component in [body,pb]:
        component.set_sleep_threshold(0)
        component.set_linear_damping(0)
        component.set_angular_damping(0)
    pb.set_linear_velocity([settings.approach_speed_m_s,0,0])

    def observe():
        return np.concatenate([cube.pose.p,body.linear_velocity,body.angular_velocity,cube.pose.q,
                               pusher.pose.p,pb.linear_velocity]).astype(np.float64)

    states=[observe()]; phases=[];actions=[];impulses=[];bounds=[];counts=[];floor_bounds=[]
    def step(phase,force):
        if phase!=3:pb.add_force_torque([force,0,0],[0,0,0])
        scene.step()
        net=np.zeros(3);upper=0.;count=0;floor=0.
        for contact in scene.physx_system.get_contacts():
            if body not in contact.bodies:continue
            if pb in contact.bodies:
                for point in contact.points:
                    v=np.asarray(point.impulse,dtype=np.float64)
                    net+=v;upper+=float(np.linalg.norm(v));count+=1
            else:
                floor+=sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
        phases.append(phase);actions.append(force);impulses.append(net)
        bounds.append(upper/settings.dt);counts.append(count);floor_bounds.append(floor/settings.dt)
        states.append(observe())

    for phase,force in commands(settings):step(phase,force)
    release_index=len(phases)
    # Instant tool withdrawal is an ideal release. Never change cube state.
    pb.set_linear_velocity([0,0,0]);pb.set_kinematic(True)
    pusher.set_pose(sapien.Pose(pusher.pose.p-np.array([.1,0,0])))
    count=0;stop_index=None
    confirmation=round(settings.stop_confirmation_s/settings.dt)
    def at_rest(state):
        return (np.linalg.norm(state[3:6])<settings.stop_speed_m_s and
                np.linalg.norm(state[6:9])<settings.stop_angular_speed_rad_s)
    for _ in range(round(settings.maximum_slide_s/settings.dt)):
        step(3,0.)
        count=count+1 if at_rest(states[-1]) else 0
        if count>=confirmation:
            stop_index=len(states)-confirmation
            if stop_index==release_index+1 and at_rest(states[release_index]):stop_index=release_index
            break
    state=np.asarray(states)
    return dict(state=state,phase=np.asarray(phases,dtype=np.int8),
                action=np.asarray(actions),contact_impulse=np.asarray(impulses),
                contact_force_bound_n=np.asarray(bounds),contact_point_count=np.asarray(counts,dtype=np.int16),
                floor_contact_force_bound_n=np.asarray(floor_bounds),time_s=np.arange(len(states))*settings.dt,
                release_index=np.asarray(release_index),stop_index=np.asarray(-1 if stop_index is None else stop_index),
                state_columns=np.asarray(STATE_COLUMNS))


def measure(trace,settings):
    state=trace['state'];r=int(trace['release_index']);s=int(trace['stop_index'])
    end=s if s>=0 else len(state)-1
    speed=float(np.linalg.norm(state[r,3:5]))
    distance=float(np.linalg.norm(state[end,:2]-state[r,:2]))
    force=trace['contact_force_bound_n']
    tilt=np.degrees(np.arccos(np.clip(1-2*(state[:,10]**2+state[:,11]**2),-1,1)))
    return dict(launch_speed_m_s=speed,stopping_distance_m=distance,
                time_to_stop_s=(end-r)*settings.dt,
                max_contact_force_n=float(force.max()),
                mean_push_contact_force_n=float(force[trace['phase']==2].mean()),
                cap_ratio=float(force.max()/settings.cap_n),stopped=s>=0,
                release_contact_force_n=float(force[r]),max_slide_contact_force_n=float(force[r:].max()),
                max_tilt_deg=float(tilt.max()),min_cube_z_m=float(state[:,2].min()),max_cube_z_m=float(state[:,2].max()),
                max_contact_points=int(trace['contact_point_count'].max()),
                force_free_deceleration_m_s2=speed**2/(2*distance) if speed>.02 and distance>.0001 else None)
