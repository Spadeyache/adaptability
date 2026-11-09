"""Episode-specific Panda physics on ManiSkill's CPU Pick, Push and Pull tasks."""
import numpy as np
import sapien
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.envs.tasks.tabletop.pull_cube import PullCubeEnv

from .collection_lift import WeightedCollectionEnv
from .environment import CollectionEnv

TASKS = ('PickCube-v1', 'PushCube-v1', 'PullCube-v1', 'poke')
CONSTANTS = ('mass_kg', 'cube_friction', 'table_friction', 'grip_scale', 'damping_scale',
             'stiffness_scale', 'delay_steps', 'com_x_m', 'com_y_m', 'com_z_m')
STATE = {'qpos': (0,9), 'qvel': (9,18), 'tcp_pose_wxyz': (18,25), 'gripper_width': (25,26),
         'cube_pose_wxyz': (26,33), 'cube_linear_velocity': (33,36), 'cube_angular_velocity': (36,39)}


def make_env(task, cfg, limits):
    if task in ('PickCube-v1', 'poke'):
        return WeightedCollectionEnv({'finger_friction': cfg['finger_friction']},
            reported_mass_kg=.04, goal_clearance_m=cfg['goal_clearance_m'], grip_force_limits_n=limits)
    cls = {'PushCube-v1': PushCubeEnv, 'PullCube-v1': PullCubeEnv}[task]
    return cls(num_envs=1, robot_uids='panda', sim_backend='physx_cpu', render_backend='cpu',
               control_mode='pd_ee_delta_pos', obs_mode='state', reward_mode='none', enhanced_determinism=True)


class PandaMotion:
    """Use the task's CPU Pinocchio IK and native PhysX drives without rendering or rewards.

    The native tasks use noninterpolated relative EE translation in robot-root axes.
    Skipping tensor packing around the single-robot controller matters for 2M steps.
    The two finger drives follow the original policy's opening and strength interface.
    """
    def __init__(self, env, limits, finger_friction):
        self.env = env
        env.reconfiguration_freq = 0
        self.obj = env.cube if hasattr(env, 'cube') else env.obj
        self.body = self.obj._bodies[0]
        visual = self.body.entity.find_component_by_type(sapien.render.RenderBodyComponent)
        if visual is not None:
            for shape in visual.render_shapes:
                for part in shape.parts:
                    part.material.base_color = [1., 0., 0., 1.]
        self.arm = env.agent.controller.controllers['arm']
        arm_cfg = self.arm.config
        assert arm_cfg.frame == 'root_translation' and not arm_cfg.interpolate and not arm_cfg.use_target
        assert arm_cfg.use_delta and arm_cfg.normalize_action
        self.ee_scale = np.float32(arm_cfg.pos_upper)
        assert np.allclose(arm_cfg.pos_lower, -self.ee_scale)
        self.ik = self.arm.kinematics
        self.robot = env.agent.robot._objs[0]
        self.arm_joints = [joint._objs[0] for joint in self.arm.joints]
        self.base_drives = [(j.stiffness, j.damping, j.force_limit, j.drive_mode) for j in self.arm_joints]
        gripper = env.agent.controller.controllers['gripper']
        self.fingers = [joint._objs[0] for joint in gripper.joints]
        self.width_range = (gripper.config.lower, gripper.config.upper)
        self.force_range = limits
        self.finger_kd = (gripper.config.stiffness, gripper.config.damping)
        for link in (env.agent.finger1_link, env.agent.finger2_link):
            CollectionEnv._material(link._bodies[0], finger_friction, finger_friction, 0.)
        self.tcp = env.agent.tcp._bodies[0]
        self.table = env.table_scene.table._bodies[0]
        self.side_m = float(self.body.get_collision_shapes()[0].half_size[0])*2

    def set_episode(self, values, seed):
        self.env.reset(seed=int(seed))
        mass, cube_mu, table_mu, grip, damping, stiff, delay = values[:7]
        com = values[7:]
        self.body.set_mass(mass)
        self.body.set_cmass_local_pose(sapien.Pose(p=com))
        self.body.set_inertia(np.full(3, mass * self.side_m**2 / 6))
        CollectionEnv._material(self.body, cube_mu, cube_mu, 0.)
        CollectionEnv._material(self.table, table_mu, table_mu, 0.)
        for joint, (k, d, cap, mode) in zip(self.arm_joints, self.base_drives):
            joint.set_drive_properties(k*stiff, d*damping, cap, mode)
        self.grip_scale = grip
        self.set_grip_cap(self.force_range[1] * grip)
        self.base_inverse = self.robot.root.pose.inv()
        self.delay = int(delay)
        self.pending = [np.zeros(5, dtype=np.float32) for _ in range(self.delay)]
        # Verify the actual scene after each independent reset.
        assert np.allclose(self.body.cmass_local_pose.p, com, atol=1e-8)
        assert np.isclose(self.body.mass, mass)
        assert np.allclose(self.body.inertia, mass*self.side_m**2/6, atol=1e-11)
        for body, expected in [(self.body, cube_mu), (self.table, table_mu)]:
            for material in CollectionEnv._materials(body):
                assert np.allclose([material['static_friction'], material['dynamic_friction']], expected)
        for joint, (k,d,_,_) in zip(self.arm_joints, self.base_drives):
            assert np.allclose([joint.stiffness, joint.damping], [k*stiff, d*damping])
        assert all(np.isclose(j.force_limit, self.force_range[1]*grip) for j in self.fingers)

    def set_grip_cap(self, cap):
        for joint in self.fingers:
            joint.set_drive_properties(*self.finger_kd, cap, 'force')

    def state(self):
        tcp, cube = self.tcp.entity_pose, self.body.entity_pose
        qpos = self.robot.qpos
        return np.concatenate((qpos, self.robot.qvel, tcp.p, tcp.q, [qpos[-2:].sum()],
                               cube.p, cube.q, self.body.linear_velocity, self.body.angular_velocity)).astype(np.float32)

    def step(self, issued):
        self.pending.append(issued)
        applied = self.pending.pop(0)
        qpos = self.robot.qpos
        tcp_local = self.base_inverse * self.tcp.entity_pose
        target = sapien.Pose(tcp_local.p + self.ee_scale * applied[:3], tcp_local.q)
        result, success, _ = self.ik.pmodel.compute_inverse_kinematics(self.ik.end_link_idx,
                target, initial_qpos=qpos, active_qmask=self.ik.qmask, max_iterations=100)
        desired = result if success else qpos
        for joint, position in zip(self.arm_joints, desired[self.ik.pmodel_controlled_joint_indices]):
            joint.set_drive_target(float(position))
        close, open_ = self.width_range
        width = close + (applied[3]+1)*.5*(open_-close)
        low, high = self.force_range
        cap = (low + (applied[4]+1)*.5*(high-low))*self.grip_scale
        self.set_grip_cap(cap)
        for finger in self.fingers:
            finger.set_drive_target(width)
        for _ in range(self.env._sim_steps_per_control):
            self.env.scene.step()
        if hasattr(self.env, '_contacts_stale'):
            self.env._contacts_stale = False
        return applied
