"""Frozen v1 collection task with an explicitly supplied mass estimate.

Training LiftTask and this independent-physics task intentionally retain distinct
semantics. The shared simulator does not import collection commands.
"""
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply
from .environment import CollectionEnv
from .grip_controller import install_strength_controller

class WeightedCollectionEnv(CollectionEnv):
    """45-input/5-action trained interface; reported weight never uses true mass.

    Goal is directly above the spawn: pick, lift, and hold, without lateral carry.
    Independent object materials replace the training task's mass/friction coupling.
    """
    episode_schema_version = 'pick_failure_v3'
    collection_description = 'WeightedCollectionEnv: held object within 2.5cm of vertical goal, arm static'

    def __init__(self, properties, *, reported_mass_kg, goal_clearance_m, grip_force_limits_n,
                 spawn_offset_m=None, spawn_yaw_offset_rad=0.0):
        self.spawn_offset_m = [0.,0.] if spawn_offset_m is None else list(spawn_offset_m)
        self.spawn_yaw_offset_rad = float(spawn_yaw_offset_rad)
        self.reported_mass_kg = float(reported_mass_kg)
        self.goal_clearance_m = float(goal_clearance_m)
        self.grip_force_limits_n = list(grip_force_limits_n)
        super().__init__(properties)

    def _load_agent(self, options):
        super()._load_agent(options)
        install_strength_controller(self)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        self.agent.controller.controllers['gripper'].reset_env_idx = env_idx
        if any(self.spawn_offset_m) or self.spawn_yaw_offset_rad:
            pose = self.cube.pose.raw_pose.clone()
            pose[:, :2] += torch.tensor(self.spawn_offset_m,device=self.device)
            angle = self.spawn_yaw_offset_rad/2
            yaw = torch.tensor([[np.cos(angle),0.,0.,np.sin(angle)]],device=self.device,dtype=pose.dtype)
            pose[:, 3:] = quaternion_multiply(yaw,pose[:, 3:])
            pose[:, 2] = self.support_height(pose[:, 3:])
            self.cube.set_pose(Pose.create(pose))
        goal = self.cube.pose.p.clone()
        goal[:, 2] += self.goal_clearance_m
        self.goal_site.set_pose(Pose.create_from_pq(goal))

    def _get_obs_extra(self, info):
        measurements = super()._get_obs_extra(info)
        gravity = float(np.linalg.norm(self.scene.px.get_config().gravity))
        measurements['object_weight_n'] = torch.full((1, 1), self.reported_mass_kg,
            device=self.device, dtype=torch.float32) * gravity
        forces = []
        for link, sign in ((self.agent.finger1_link, 1), (self.agent.finger2_link, -1)):
            force = self.scene.get_pairwise_contact_forces(link, self.cube)
            axis = sign * link.pose.to_transformation_matrix()[:, :3, 1]
            value = (force * axis).sum(-1, keepdim=True).clamp_min(0)
            forces.append(torch.zeros_like(value) if self._contacts_stale else value)
        measurements['left_finger_force_n'], measurements['right_finger_force_n'] = forces
        return measurements

    def evaluate(self):
        info = super().evaluate()
        info['success'] = info['success'] & info['is_grasped']
        return info

