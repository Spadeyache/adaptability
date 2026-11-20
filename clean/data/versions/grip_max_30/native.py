"""Simulator adapter. Cube physics can only be installed through the setter."""
import os
from pathlib import Path
os.environ.setdefault('MS_ASSET_DIR',str(Path(__file__).resolve().parents[1]/'.maniskill/assets'))
os.environ.setdefault('MS_SKIP_ASSET_DOWNLOAD_PROMPT','1')
os.environ.setdefault('VK_ICD_FILENAMES','/usr/share/vulkan/icd.d/lvp_icd.x86_64.json')
import numpy as np
import sapien
import sapien.physx as physx
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from clean.script import INITIAL_QPOS

def make_env(seed):
    env=PickCubeEnv(num_envs=1,robot_uids='panda',sim_backend='physx_cpu',render_backend='none',
        control_mode='pd_joint_pos',obs_mode='none',reward_mode='none',robot_init_qpos_noise=0,
        enhanced_determinism=True)
    reset_env(env,seed)
    return env

def reset_env(env,seed):
    """Reset measured kinematics and nuisance materials before installing cube properties."""
    env.reset(seed=int(seed))
    env.agent.robot.set_qpos(torch.tensor(INITIAL_QPOS,dtype=torch.float32)[None])
    env.agent.robot.set_qvel(torch.zeros((1,9)))
    env.agent.controller.reset()
    # Controller.reset resets buffers but leaves old native drive targets.
    # Set deterministic targets without advancing physics so initial sensor
    # errors cannot retain the preceding episode's last gripper command.
    for joint,target in zip(env.agent.robot._objs[0].get_active_joints(),INITIAL_QPOS):
        joint.set_drive_target(float(target))
    # Fixed nuisance materials, identical for every cube and episode.
    for obj,coefficient in [(env.agent.finger1_link,.2),(env.agent.finger2_link,.2),(env.table_scene.table,.3)]:
        for body in obj._bodies:
            for shape in body.get_collision_shapes():shape.physical_material=physx.PhysxMaterial(coefficient,coefficient,0.)
    return env

def set_cube_properties(env,properties):
    """The sole simulator mass/friction/inertia setter; never called by the script."""
    body=env.cube._bodies[0]
    mass=float(properties['mass_kg']);friction=float(properties['friction'])
    body.set_mass(mass)
    side=2*float(env.cube_half_size)
    body.set_inertia(np.full(3,mass*side**2/6))
    body.set_cmass_local_pose(sapien.Pose())
    for shape in body.get_collision_shapes():shape.physical_material=physx.PhysxMaterial(friction,friction,0.)
    if not np.isclose(body.mass,mass):raise ValueError('Mass installation failed')
    for shape in body.get_collision_shapes():
        material=shape.physical_material
        if not np.allclose([material.static_friction,material.dynamic_friction],friction):raise ValueError('Material installation failed')

def execute(env,command):
    for joint in env.agent.robot._objs[0].get_active_joints()[-2:]:
        joint.set_drive_properties(joint.stiffness,joint.damping,float(command[8]),joint.drive_mode)
    env.step(np.asarray(command[:8],dtype=np.float32))
