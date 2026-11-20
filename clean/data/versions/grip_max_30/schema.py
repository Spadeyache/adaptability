"""Explicit observable-state schema; no simulator physics metadata."""
import numpy as np

COLUMNS=[]
for name,n,unit,source in [
    ('joint_position',9,'rad (0–6), m (7–8)','robot.qpos'),
    ('joint_velocity',9,'rad/s (0–6), m/s (7–8)','robot.qvel'),
    ('joint_effort',9,'N m (0–6), N (7–8)','incoming child-link joint wrench: torque X for revolute, force X for prismatic'),
    ('gripper_opening',1,'m','sum of measured finger joint positions'),
    ('finger_position_error',2,'m','issued finger drive target minus measured finger position'),
    ('cube_position',3,'m','cube entity world pose position'),
    ('cube_rotation',6,'unitless','first two columns of cube pose rotation matrix'),
    ('cube_linear_velocity',3,'m/s','cube rigid-body linear velocity'),
    ('cube_angular_velocity',3,'rad/s','cube rigid-body angular velocity'),
    ('cube_relative_position',3,'m','TCP rotation transpose times (cube position minus TCP position)')]:
    COLUMNS += [dict(name=f'{name}_{i}',unit=unit,source=source) for i in range(n)]

ACTION_COLUMNS=[f'arm_joint_target_{i}_rad' for i in range(7)]+['gripper_command_normalized','grip_force_cap_per_finger_N']

def observe(env,first=False):
    robot=env.agent.robot._objs[0];body=env.cube._bodies[0]
    joints=robot.get_active_joints();q=np.asarray(robot.qpos);v=np.asarray(robot.qvel)
    effort=np.zeros(9)
    if not first:
        wrench=robot.get_link_incoming_joint_forces()
        ids=[j.get_child_link().index for j in joints]
        effort[:7]=wrench[ids[:7],3];effort[7:]=wrench[ids[7:],0]
    error=np.asarray([float(np.asarray(j.drive_target).item()) for j in joints[-2:]])-q[-2:]
    pose=body.entity_pose;tcp=env.agent.tcp._bodies[0].entity_pose
    rotation=pose.to_transformation_matrix()[:3,:2].T.ravel()
    relative=tcp.to_transformation_matrix()[:3,:3].T@(pose.p-tcp.p)
    state=np.concatenate([q,v,effort,[q[-2:].sum()],error,pose.p,rotation,body.linear_velocity,body.angular_velocity,relative]).astype(np.float32)
    if len(state)!=len(COLUMNS) or not np.isfinite(state).all():raise ValueError('Invalid whitelisted state')
    return state
