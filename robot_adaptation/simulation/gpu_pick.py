"""Randomized PickCube-v1 scenes for one synchronized GPU batch of episodes."""
import numpy as np
import sapien
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv

from .environment import CollectionEnv
from .grip_controller import install_strength_controller
from .training_lift import normal_forces


def set_surface(body, coefficient):
    CollectionEnv._material(body, coefficient, coefficient, 0.)
    materials = CollectionEnv._materials(body)
    for material in materials:
        np.testing.assert_allclose(
            [material['static_friction'], material['dynamic_friction']], coefficient, rtol=2e-6
        )
    return materials[0]['static_friction']


class RandomizedPickGPU(PickCubeEnv):
    """Install physical constants before GPU initialization and keep them for one rollout."""

    def __init__(self, requested, guess_kg, force_limits, finger_friction, scene_spacing_m):
        self.requested = requested
        self.guess_kg = guess_kg
        self.grip_force_limits_n = force_limits
        self.finger_friction = finger_friction
        # One independent native body and articulation per row, with no cameras.
        super().__init__(num_envs=len(requested), robot_uids='panda',
                         sim_backend='physx_cuda', render_backend='none',
                         control_mode='pd_ee_delta_pos', obs_mode='state', reward_mode='none',
                         reconfiguration_freq=0, enhanced_determinism=True,
                         sim_config={'spacing': scene_spacing_m, 'gpu_memory_config': {
                             'found_lost_pairs_capacity': 2**20,
                             'max_rigid_contact_count': 2**18, 'max_rigid_patch_count': 2**16,
                             'total_aggregate_pairs_capacity': 2**14,
                             'found_lost_aggregate_pairs_capacity': 2**14}})
        if not self.gpu_sim_enabled or self.device.type != 'cuda':
            raise RuntimeError('PickCube requires CUDA PhysX for this collection')
        if self.sim_config.spacing != scene_spacing_m:
            raise RuntimeError('Parallel scene spacing differs from the collection protocol')

    def _load_agent(self, options):
        super()._load_agent(options)
        install_strength_controller(self)
        for finger in (self.agent.finger1_link, self.agent.finger2_link):
            for body in finger._bodies:
                set_surface(body, self.finger_friction)

    def _load_scene(self, options):
        super()._load_scene(options)
        self.realized = np.empty_like(self.requested)
        gripper = self.agent.controller.controllers['gripper']
        strength = torch.as_tensor(self.requested[:, 3], device=self.device, dtype=torch.float32)
        for name in ('action_space_low', 'action_space_high'):
            bounds = getattr(gripper, name).expand(self.num_envs, -1).clone()
            bounds[:, 1] *= strength
            setattr(gripper, name, bounds)
        gripper._apply_limits(gripper.action_space_high[:, 1])
        gravity = np.linalg.norm(self.scene.px.get_config().gravity)
        self.reported_weight = torch.as_tensor(self.guess_kg * gravity, device=self.device,
                                               dtype=torch.float32).reshape(-1, 1)
        arm = self.agent.controller.controllers['arm']
        self.arm_base_gains = [(joint._objs[0].stiffness, joint._objs[0].damping) for joint in arm.joints]
        for i, row in enumerate(self.requested):
            mass, cube_mu, table_mu, grip, damping, stiffness, delay = row[:7]
            cube = self.cube._bodies[i]
            half = cube.get_collision_shapes()[0].half_size
            np.testing.assert_allclose(half, [.02]*3, atol=1e-7)
            inertia = np.full(3, mass*(2*half[0])**2/6)
            cube.set_mass(mass)
            cube.set_cmass_local_pose(sapien.Pose(p=row[7:]))
            cube.set_inertia(inertia)
            np.testing.assert_allclose(cube.inertia, inertia, rtol=2e-5)
            actual_cube_mu = set_surface(cube, cube_mu)
            actual_table_mu = set_surface(self.table_scene.table._bodies[i], table_mu)
            for joint in arm.joints:
                native = joint._objs[i]
                base_k, base_d = native.stiffness, native.damping
                native.set_drive_properties(base_k*stiffness, base_d*damping,
                                            native.force_limit, native.drive_mode)
                actual_k, actual_d = native.stiffness/base_k, native.damping/base_d
                np.testing.assert_allclose([actual_k,actual_d], [stiffness,damping], rtol=2e-6)
            grip_scales = [joint._objs[i].force_limit/self.grip_force_limits_n[1] for joint in gripper.joints]
            np.testing.assert_allclose(grip_scales, grip, rtol=2e-6)
            # Delay is a software actuator queue. Its realized timing is checked
            # against every issued/applied command after the file is reopened.
            self.realized[i] = [cube.mass, actual_cube_mu, actual_table_mu, grip_scales[0],
                                actual_d, actual_k, delay, *cube.cmass_local_pose.p]
        np.testing.assert_allclose(self.realized, self.requested, rtol=2e-6, atol=1e-8)

    def check_reset_physics(self):
        """Re-read all native labels after the episode reset and GPU initialization."""
        arm = self.agent.controller.controllers['arm']
        gripper = self.agent.controller.controllers['gripper']
        for i, row in enumerate(self.realized):
            cube = self.cube._bodies[i]
            np.testing.assert_allclose([cube.mass, *cube.cmass_local_pose.p],
                                       [row[0], *row[7:]], rtol=2e-6, atol=1e-8)
            for body, expected in ((cube, row[1]), (self.table_scene.table._bodies[i], row[2])):
                for surface in CollectionEnv._materials(body):
                    np.testing.assert_allclose([surface['static_friction'], surface['dynamic_friction']],
                                               expected, rtol=2e-6)
            for joint, (base_k, base_d) in zip(arm.joints, self.arm_base_gains):
                native = joint._objs[i]
                np.testing.assert_allclose([native.stiffness/base_k, native.damping/base_d],
                                           [row[5], row[4]], rtol=2e-6)
            limits = [joint._objs[i].force_limit for joint in gripper.joints]
            np.testing.assert_allclose(limits, row[3]*self.grip_force_limits_n[1], rtol=2e-6)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        self._contacts_stale = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _after_control_step(self):
        super()._after_control_step()
        self._contacts_stale.zero_()

    def _get_obs_extra(self, info):
        extra = super()._get_obs_extra(info)
        extra['is_grasped'] = extra['is_grasped'] & ~self._contacts_stale
        forces = normal_forces(self)
        extra['object_weight_n'] = self.reported_weight
        extra['left_finger_force_n'] = forces[:, :1]
        extra['right_finger_force_n'] = forces[:, 1:]
        return extra

    def state(self):
        qpos = self.agent.robot.get_qpos()
        return torch.cat((qpos, self.agent.robot.get_qvel(), self.agent.tcp_pose.raw_pose,
                          qpos[:, -2:].sum(1, keepdim=True), self.cube.pose.raw_pose,
                          self.cube.linear_velocity, self.cube.angular_velocity), dim=1)

    def finger_limits(self):
        joints = self.agent.controller.controllers['gripper'].joints
        return np.asarray([[j.force_limit for j in joint._objs] for joint in joints], np.float32).T
