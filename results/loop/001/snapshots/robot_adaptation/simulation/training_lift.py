"""THE ENVIRONMENT: carry and hold a cube, using only the grip force needed.

Keep the Panda arm and finger-opening controller. Add a per-finger motor strength
limit, two measured compression forces, and a small measured-force reward cost.
The optional hidden friction profile deliberately correlates surface friction
with actual mass; experiment metadata explicitly records both changing.
"""
import numpy as np
import sapien.physx as physx
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.common import flatten_state_dict
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from robot_adaptation.simulation.grip_controller import install_strength_controller

ENV_ID = "GripWeightedPickCube-v2"
HORIZON = 50
ORIGIN = "maniskill_grip_weighted_pick_cube_v2"
LEGACY_ORIGIN = "maniskill_weighted_pick_cube"
ENV_SETTINGS = {"masses_kg": [0.040, 0.064, 0.100, 0.250, 0.500],
                "goal_clearance_m": [0.10, 0.30], "grip_force_limits_n": [0.25, 40.0],
                "grip_force_cost": 0.05,
                # Deliberate simulator challenge: mass AND surface friction vary.
                "friction_profile": "mass_correlated_linear",
                "friction_mass_range_kg": [0.040, 0.500],
                "cube_friction_range": [1.0, 0.4],
                "finger_friction_coefficient": 0.2,
                "cube_friction_coefficients": [0.2, 0.5, 1.0]}
FRICTION_ORIGIN = "maniskill_grip_mass_friction_pick_cube_v3"
FRICTION_INPUT_INDEX = 45
WEIGHT_INPUT_INDEX = 42
INPUT_FIELDS = ["joint_positions", "joint_velocities", "is_grasped", "tcp_pose_world_m_wxyz",
                "goal_position_world_m", "cube_pose_world_m_wxyz", "tcp_to_cube_world_m",
                "cube_to_goal_world_m", "object_weight_n", "left_finger_force_n", "right_finger_force_n"]
INPUT_SIZES = [9, 9, 1, 7, 3, 7, 3, 3, 1, 1, 1]
FRICTION_INPUT_FIELDS = INPUT_FIELDS + ["object_friction_coefficient"]
FRICTION_INPUT_SIZES = INPUT_SIZES + [1]
ACTION_DESCRIPTION = "5 actions: native root-frame XYZ increments, finger opening (-1 closed, +1 open), per-finger motor strength limit (-1 minimum, +1 maximum)"


FRICTION_KEYS = ("friction_profile", "friction_mass_range_kg", "cube_friction_range", "finger_friction_coefficient", "cube_friction_coefficients")


def friction_settings(settings):
    """Resolve old manifests to native friction; new settings record all values."""
    values = {key: settings.get(key, ENV_SETTINGS[key]) for key in FRICTION_KEYS}
    values["friction_profile"] = settings.get("friction_profile", "native")
    if values["friction_profile"] not in ("native", "mass_correlated_linear", "independent_grid"):
        raise ValueError("Choose native, mass_correlated_linear, or independent_grid friction")
    bounds = values["friction_mass_range_kg"]
    coefficients = values["cube_friction_range"]
    if len(bounds) != 2 or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
        raise ValueError("Choose two ordered positive friction mass bounds")
    if len(coefficients) != 2 or not np.isfinite(coefficients).all() or not coefficients[0] >= coefficients[1] > 0:
        raise ValueError("Cube friction must decrease from a positive light coefficient to a positive heavy coefficient")
    finger = values["finger_friction_coefficient"]
    if not np.isfinite(finger) or finger <= 0:
        raise ValueError("Finger friction must be positive and finite")
    grid = values["cube_friction_coefficients"]
    if not grid or not np.isfinite(grid).all() or np.any(np.asarray(grid) <= 0):
        raise ValueError("Choose positive finite independent cube friction coefficients")
    return values


def cube_friction_at_mass(mass, settings):
    """Linear decrease with ACTUAL mass, clamped at the endpoint coefficients."""
    low, high = settings["friction_mass_range_kg"]
    light, heavy = settings["cube_friction_range"]
    fraction = np.clip((float(mass) - low) / (high - low), 0., 1.)
    return float(light + fraction * (heavy - light))


def friction_experiment(settings, masses):
    config = friction_settings(settings)
    changing = config["friction_profile"] == "mass_correlated_linear"
    if config["friction_profile"] == "independent_grid":
        return dict(settings=config, hidden_from_policy=not settings.get("condition_on_friction", False),
                    changes_with_actual_mass=False, independently_crossed=True,
                    conditions=[dict(mass_kg=float(mass), cube_static_friction=float(mu),
                                     cube_dynamic_friction=float(mu), finger_friction=config["finger_friction_coefficient"])
                                for mu in config["cube_friction_coefficients"] for mass in masses],
                    note="Mass and object friction are independently crossed; static equals dynamic; finger/table materials are fixed.")
    return dict(settings=config, hidden_from_policy=True, changes_with_actual_mass=changing,
                coefficients_by_mass=[dict(mass_kg=float(mass),
                    cube_static_friction=cube_friction_at_mass(mass, config) if changing else None,
                    cube_dynamic_friction=cube_friction_at_mass(mass, config) if changing else None,
                    finger_static_friction=config["finger_friction_coefficient"] if changing else None,
                    finger_dynamic_friction=config["finger_friction_coefficient"] if changing else None) for mass in masses],
                note=("Mass and surface friction BOTH change: heavier cubes have lower cube friction. "
                      "Finger friction is also reduced. This is a combined weight/friction stress test, "
                      "not isolated proof of weight adaptation.") if changing else "Original native materials; no mass-dependent friction adjustment.")


def reward_components(distance, goal_distance, grasped, joint_speed, at_goal, static):
    """PickCube reach/grasp/carry terms; success now also requires finger contact."""
    reach = (1 - torch.tanh(5 * distance)) / 5
    grasp = grasped.float() / 5
    move = (1 - torch.tanh(5 * goal_distance)) * grasped / 5
    still = (1 - torch.tanh(5 * joint_speed)) * at_goal / 5
    bonus = torch.where(at_goal & static & grasped, 1 - reach - grasp - move - still, 0)
    return dict(reach_score=reach, grasp_score=grasp, goal_score=move, static_score=still,
                success_bonus=bonus, task_score=reach + grasp + move + still + bonus)


@register_env(ENV_ID, max_episode_steps=HORIZON)
class LiftTask(PickCubeEnv):
    def __init__(self, *args, masses_kg=None, goal_clearance_m=None, grip_force_limits_n=None, grip_force_cost=None, friction_profile=None,
                 friction_mass_range_kg=None, cube_friction_range=None, finger_friction_coefficient=None,
                 cube_friction_coefficients=None, condition_on_friction=False, **kwargs):
        supplied = dict(friction_profile=friction_profile, friction_mass_range_kg=friction_mass_range_kg,
                        cube_friction_range=cube_friction_range, finger_friction_coefficient=finger_friction_coefficient,
                        cube_friction_coefficients=cube_friction_coefficients)
        self.friction_config = friction_settings({key: ENV_SETTINGS[key] if value is None else value
                                                 for key, value in supplied.items()})
        self.condition_on_friction = bool(condition_on_friction)
        if self.condition_on_friction and self.friction_config["friction_profile"] != "independent_grid":
            raise ValueError("Friction-conditioned training requires independent_grid physics")
        self._cpu_friction = float(self.friction_config["cube_friction_coefficients"][0])
        self.grip_force_limits_n = tuple(grip_force_limits_n or ENV_SETTINGS["grip_force_limits_n"])
        self.grip_force_cost = ENV_SETTINGS["grip_force_cost"] if grip_force_cost is None else grip_force_cost
        if (len(self.grip_force_limits_n) != 2 or not np.isfinite(self.grip_force_limits_n).all()
                or not 0 < self.grip_force_limits_n[0] < self.grip_force_limits_n[1]):
            raise ValueError("Choose ordered positive finite per-finger strength limits")
        if not np.isfinite(self.grip_force_cost) or self.grip_force_cost < 0:
            raise ValueError("grip_force_cost must be finite and nonnegative")
        self.masses_kg = tuple(masses_kg if masses_kg is not None else ENV_SETTINGS["masses_kg"])
        self.goal_clearance_m = tuple(goal_clearance_m if goal_clearance_m is not None else ENV_SETTINGS["goal_clearance_m"])
        if not self.masses_kg or any(not np.isfinite(m) or m <= 0 for m in self.masses_kg):
            raise ValueError("masses_kg must contain positive finite masses")
        if (len(self.goal_clearance_m) != 2 or not np.isfinite(self.goal_clearance_m).all()
                or self.goal_clearance_m[0] <= .025 or self.goal_clearance_m[1] < self.goal_clearance_m[0]):
            raise ValueError("Choose an ordered floating-goal clearance range above the 2.5 cm goal tolerance")
        kwargs.setdefault("num_envs", 1)
        kwargs.setdefault("obs_mode", "state")
        kwargs.setdefault("control_mode", "pd_ee_delta_pos")
        kwargs.setdefault("sim_backend", "physx_cpu")
        kwargs.setdefault("render_backend", "cpu")
        kwargs.setdefault("reward_mode", "normalized_dense")
        kwargs.setdefault("sensor_configs", {"shader_pack": "minimal"})
        kwargs.setdefault("human_render_camera_configs", {"shader_pack": "minimal"})
        if kwargs["control_mode"] != "pd_ee_delta_pos" or kwargs["reward_mode"] != "normalized_dense":
            raise ValueError("Use native arm/finger position control with the added strength action and normalized dense reward")
        self._cpu_episode = 0
        super().__init__(*args, **kwargs)

    def _load_agent(self, options):
        super()._load_agent(options)
        install_strength_controller(self)
        if self.friction_config["friction_profile"] in ("mass_correlated_linear", "independent_grid"):
            for link in (self.agent.finger1_link, self.agent.finger2_link):
                for body in link._objs:
                    self._set_friction(body, self.friction_config["finger_friction_coefficient"])

    @staticmethod
    def _set_friction(body, coefficient):
        # Replace material on each shape BEFORE GPU initialization. Preserve
        # restitution and collision geometry; only static/dynamic friction change.
        for shape in body.get_collision_shapes():
            old = shape.physical_material
            shape.physical_material = physx.PhysxMaterial(coefficient, coefficient, old.restitution)
            if not (np.isclose(shape.physical_material.static_friction, coefficient)
                    and np.isclose(shape.physical_material.dynamic_friction, coefficient)):
                raise RuntimeError("Physical friction assignment failed")

    def _apply_cube_friction(self, body, mass, index=0):
        if self.friction_config["friction_profile"] == "mass_correlated_linear":
            self._set_friction(body, cube_friction_at_mass(mass, self.friction_config))
        elif self.friction_config["friction_profile"] == "independent_grid":
            mu = (self.friction_config["cube_friction_coefficients"][(index // len(self.masses_kg)) % len(self.friction_config["cube_friction_coefficients"])]
                  if self.gpu_sim_enabled else self._cpu_friction)
            self._set_friction(body, mu)

    def friction_details(self):
        """Inspect real materials; estimates are supplied through observations separately."""
        materials = dict(cube=self.cube._bodies[0].get_collision_shapes()[0].physical_material,
                         left_finger=self.agent.finger1_link._objs[0].get_collision_shapes()[0].physical_material,
                         right_finger=self.agent.finger2_link._objs[0].get_collision_shapes()[0].physical_material,
                         table=self.table_scene.table._bodies[0].get_collision_shapes()[0].physical_material)
        return dict(profile=self.friction_config["friction_profile"], hidden_from_policy=not self.condition_on_friction,
                    **{name + "_" + kind: float(getattr(material, kind))
                       for name, material in materials.items() for kind in ("static_friction", "dynamic_friction")})

    @staticmethod
    def _set_mass(body, mass):
        inertia = np.array(body.inertia, copy=True) * float(mass) / body.mass
        body.set_mass(float(mass))
        body.set_inertia(inertia)
        if not np.isclose(body.mass, mass) or not np.allclose(body.inertia, inertia):
            raise RuntimeError("Physical mass/inertia assignment failed")

    def _load_scene(self, options):
        super()._load_scene(options)
        # GPU PhysX requires mass assignment BEFORE initialization. Each parallel
        # robot has one mass and its matching material, retained across resets.
        for index, body in enumerate(self.cube._bodies):
            mass = self.masses_kg[index % len(self.masses_kg)]
            self._set_mass(body, mass)
            self._apply_cube_friction(body, mass, index)

    def _initialize_episode(self, env_idx, options):
        # Reset moves bodies without stepping physics. PhysX contact queries can
        # still describe the previous episode: invalidate ONLY reset robots until
        # a fresh simulation step. Other robots retain their measured contacts.
        if not hasattr(self, '_contacts_stale') or self._contacts_stale.shape != (self.num_envs,):
            self._contacts_stale = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._contacts_stale[env_idx] = True
        self.agent.controller.controllers["gripper"].reset_env_idx = env_idx.clone()
        # CPU cycles the complete crossed grid; explicit reset options support matched evaluation.
        if not self.gpu_sim_enabled and self.friction_config["friction_profile"] == "independent_grid":
            grid = self.friction_config["cube_friction_coefficients"]
            mu = float(options.get("friction_coefficient", grid[(self._cpu_episode // len(self.masses_kg)) % len(grid)]))
            if not np.isfinite(mu) or mu <= 0:
                raise ValueError("friction_coefficient must be positive and finite")
            self._cpu_friction = mu
        # CPU inspection can select a different physical mass each attempt.
        if not self.gpu_sim_enabled and "mass_kg" not in options:
            options = {**options, "mass_kg": self.masses_kg[self._cpu_episode % len(self.masses_kg)]}
            self._cpu_episode += 1
        if "mass_kg" in options:
            mass = float(options["mass_kg"])
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError("mass_kg must be positive and finite")
            if self.gpu_sim_enabled:
                raise ValueError("For GPU simulation assign masses_kg when constructing the scene")
            for body in self.cube._bodies:
                self._set_mass(body, mass)
                # CPU inspection selects actual mass at reset. Its friction
                # follows that mass, never the true/nominal weight fed to policy.
                self._apply_cube_friction(body, mass)
        super()._initialize_episode(env_idx, options)
        goal = self.goal_site.pose.p[env_idx].clone()
        low, high = self.goal_clearance_m
        # Random XY from PickCube; Z is a floating target for the CUBE centre.
        # This range expresses bottom clearance for an upright cube.
        goal[:, 2] = self.cube_half_size + low + (high - low) * torch.rand(len(env_idx), device=self.device)
        self.goal_site.set_pose(Pose.create_from_pq(goal))

    def _after_control_step(self):
        super()._after_control_step()
        self._contacts_stale.zero_()

    def _get_obs_extra(self, info):
        measurements = super()._get_obs_extra(info)
        gravity = torch.as_tensor(self.scene.px.get_config().gravity, device=self.device).norm()
        # Preserve the original 43 entries; append ONLY the two new force measurements.
        measurements["object_weight_n"] = self.cube.mass.to(self.device).reshape(-1, 1) * gravity
        forces = normal_forces(self)
        measurements["left_finger_force_n"] = forces[:, :1]
        measurements["right_finger_force_n"] = forces[:, 1:2]
        if self.condition_on_friction:
            if self.gpu_sim_enabled:
                grid = torch.as_tensor(self.friction_config["cube_friction_coefficients"], device=self.device)
                indices = (torch.arange(self.num_envs, device=self.device) // len(self.masses_kg)) % len(grid)
                mu = grid[indices]
            else:
                mu = torch.full((self.num_envs,), self._cpu_friction, device=self.device)
            measurements["object_friction_coefficient"] = mu.reshape(-1, 1)
        return measurements

    def evaluate(self):
        info = super().evaluate()
        if hasattr(self, '_contacts_stale'):
            info['is_grasped'] = info['is_grasped'] & ~self._contacts_stale
        # A cube at the target must still be held; approaching it is not sufficient.
        info["success"] = info["success"] & info["is_grasped"]
        distance = (self.cube.pose.p - self.agent.tcp_pose.p).norm(dim=-1)
        goal_distance = (self.cube.pose.p - self.goal_site.pose.p).norm(dim=-1)
        speed = self.agent.robot.get_qvel()[..., :-2].norm(dim=-1)
        rotation = quaternion_to_matrix(self.cube.pose.q)
        clearance = self.cube.pose.p[:, 2] - self.cube_half_size * rotation[:, 2].abs().sum(-1)
        parts = reward_components(distance, goal_distance, info["is_grasped"], speed,
                                  info["is_obj_placed"], info["is_robot_static"])
        forces = normal_forces(self)
        # Penalize measured compression, not merely proximity or a requested cap.
        # The task reward remains dominant; dropping the cube loses its carry reward.
        cost = self.grip_force_cost * (forces.mean(-1) / self.grip_force_limits_n[1]).clamp(0, 1)
        return dict(info, distance_m=distance, goal_distance_m=goal_distance, clearance_m=clearance,
                    held_goal_success=info["success"], grip_force_cost=cost,
                    reward_total=parts["task_score"]-cost,
                    grip_limit_n=self.agent.controller.controllers["gripper"].force_limit_n.clone(), **parts)

    def compute_normalized_dense_reward(self, obs, action, info):
        return info["reward_total"]

    def compute_dense_reward(self, obs, action, info):
        return 5 * info["reward_total"]


def read_inputs(task, observation):
    """45 inputs: previous 43 unchanged, then left/right compression force in N."""
    vector = flatten_state_dict(observation, use_torch=True, device=task.device) if isinstance(observation, dict) else observation
    vector = vector.detach().clone().float()
    conditioned = getattr(task, "condition_on_friction", False)
    fields, sizes = (FRICTION_INPUT_FIELDS, FRICTION_INPUT_SIZES) if conditioned else (INPUT_FIELDS, INPUT_SIZES)
    if vector.shape != (task.num_envs, sum(sizes)) or not torch.isfinite(vector).all():
        raise ValueError(f"Expected {sum(sizes)} finite measurements per robot")
    inputs, start = {}, 0
    for name, size in zip(fields, sizes):
        inputs[name] = vector[:, start:start + size]
        start += size
    return inputs, vector


def normal_forces(task):
    """Measured compressive contact forces, one scalar per finger, in newtons."""
    left = task.scene.get_pairwise_contact_forces(task.agent.finger1_link, task.cube)
    right = task.scene.get_pairwise_contact_forces(task.agent.finger2_link, task.cube)
    la = task.agent.finger1_link.pose.to_transformation_matrix()[:, :3, 1]
    ra = -task.agent.finger2_link.pose.to_transformation_matrix()[:, :3, 1]
    forces = torch.stack([(left * la).sum(-1).clamp_min(0), (right * ra).sum(-1).clamp_min(0)], dim=-1)
    if hasattr(task, '_contacts_stale'):
        forces = forces.masked_fill(task._contacts_stale[:, None], 0.)
    return forces
