"""Archived development actuator check; not the sealed collection program."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import sapien
import sapien.physx as px
from clean.native import set_cube_properties


def run(mass, friction, cap, dt=.001, duration=.08, pusher_mass=.005):
    px.set_scene_config(gravity=[0, 0, -9.81], enable_enhanced_determinism=True,
                        enable_tgs=True, enable_pcm=True)
    scene = sapien.Scene([px.PhysxCpuSystem()])
    scene.set_timestep(dt)
    scene.add_ground(0, render=False, material=px.PhysxMaterial(.3, .3, 0))
    b = scene.create_actor_builder()
    b.add_box_collision(half_size=[.02]*3, material=px.PhysxMaterial(.5, .5, 0))
    b.initial_pose = sapien.Pose([0, 0, .02])
    cube = b.build(name='cube')
    body = cube.find_component_by_type(px.PhysxRigidDynamicComponent)
    set_cube_properties(SimpleNamespace(cube=SimpleNamespace(_bodies=[body]),cube_half_size=.02),
                        {'mass_kg': mass, 'friction': friction})
    body.set_sleep_threshold(0)
    b = scene.create_actor_builder()
    b.add_sphere_collision(radius=.008, material=px.PhysxMaterial(0, 0, 0))
    b.initial_pose = sapien.Pose([-.031, 0, .02])
    pusher = b.build(name='pusher')
    pb = pusher.find_component_by_type(px.PhysxRigidDynamicComponent)
    pb.set_mass(pusher_mass)
    pb.set_disable_gravity(True)
    pb.set_locked_motion_axes([False, True, True, True, True, True])
    pb.set_sleep_threshold(0)
    body.set_linear_damping(0); body.set_angular_damping(0)
    pb.set_linear_damping(0); pb.set_angular_damping(0)
    rows=[]
    def step(phase, command):
        if phase != 3:
            pb.add_force_torque([command,0,0],[0,0,0])
        scene.step()
        impulse=np.zeros(3); normsum=0.; points=0
        for c in scene.physx_system.get_contacts():
            if body in c.bodies and pb in c.bodies:
                for pt in c.points:
                    impulse+=np.asarray(pt.impulse)
                    normsum+=np.linalg.norm(pt.impulse)
                    points+=1
        rows.append([phase,command,*cube.pose.p,*body.linear_velocity,*body.angular_velocity,
                     *pusher.pose.p,*pb.linear_velocity,*impulse/dt,normsum/dt,points])
        return normsum/dt
    # Fixed low-speed approach servo, observing only pusher speed and contact.
    for _ in range(1000):
        command=np.clip((.005-pb.linear_velocity[0])*.5,-.1,.1)
        if step(0,command)>.02:
            break
    for _ in range(100):
        step(1,.03)
    for _ in range(round(duration/dt)):
        step(2,.95*cap)
    # Remove contact by disabling and parking the retracted pusher at release.
    pb.set_linear_velocity([0,0,0])
    pb.set_kinematic(True)
    pusher.set_pose(sapien.Pose(pusher.pose.p-np.array([.1,0,0])))
    for _ in range(4000):
        step(3,0)
        if len(rows)>300 and np.linalg.norm(body.linear_velocity)<.001 and np.linalg.norm(body.angular_velocity)<.05:
            if all(np.linalg.norm(r[5:8])<.001 for r in rows[-100:]):break
    return np.asarray(rows)


if __name__=='__main__':
    root=Path(__file__).resolve().parent/'development'
    root.mkdir(exist_ok=True)
    summary=[]; cubes=[]
    for i,(mass,mu,cap) in enumerate([(m,.7,f) for m in [.04,.27,.5] for f in [2.,3.,4.]]):
        rows=run(mass,mu,cap)
        np.savez_compressed(root/f'trial_{i:03d}.npz',trace=rows)
        push=rows[rows[:,0]==2];slide=rows[rows[:,0]==3]
        cubes.append(dict(cube_id=i,mass_kg=mass,friction=mu))
        summary.append(dict(trial=i,cube_id=i,cap_n=cap,
            max_force_n=float(rows[:,-2].max()),push_mean_force_n=float(push[:,-2].mean()),
            launch_speed=float(push[-1,5]),distance=float(slide[-1,2]-push[-1,2]),
            stop_time_s=len(slide)*.001,max_spin=float(np.linalg.norm(rows[:,8:11],axis=1).max()),
            final_z=float(rows[-1,4])))
    (root/'cubes.json').write_text(json.dumps({'cubes':cubes},indent=2)+'\n')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
