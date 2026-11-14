"""Scripted Panda contacts with a bounded Cartesian-force shove."""
import numpy as np
from scipy.spatial.transform import Rotation

from robot_adaptation.simulation.context_dynamics import PandaMotion


class ScriptedMotion(PandaMotion):
    """Keep the native position drives or apply a known horizontal TCP wrench."""

    def __init__(self, env, limits, finger_friction):
        super().__init__(env, limits, finger_friction)
        self.force_mode = False
        self.contacts = []
        self._native_scene_step = env.scene.step
        self.contact_links = [env.agent.finger1_link, env.agent.finger2_link,
                              env.agent.robot.links_map['panda_hand']]
        env.scene.step = self._physics_step

    def close(self):
        self.env.scene.step = self._native_scene_step
        self.env.close()

    def _physics_step(self):
        self._native_scene_step()
        # The query gives force on the robot. Negate to get force on the cube.
        force = -sum((self.env.scene.get_pairwise_contact_forces(link, self.obj)[0].numpy()
                      for link in self.contact_links), np.zeros(3))
        self.contacts.append(force)

    def set_episode(self, values, seed):
        self.robot.set_qf(np.zeros(9))
        self.force_mode = False
        super().set_episode(values, seed)

    def _force_substep(self, force_xy):
        positions, velocities = self.robot.qpos, self.robot.qvel
        jacobian = self.ik.pmodel.compute_single_link_local_jacobian(
            positions, self.ik.end_link_idx)[:, :7]
        rotation = self.tcp.entity_pose.to_transformation_matrix()[:3, :3]
        jacobian[:3] = rotation @ jacobian[:3]
        jacobian[3:] = rotation @ jacobian[3:]
        twist = jacobian @ velocities[:7]
        # Fixed references keep every control target observable from the mode.
        rotation_error = Rotation.from_matrix(np.diag([1., -1., -1.]) @ rotation.T).as_rotvec()
        vertical_force = np.clip(200 * (.023 - self.tcp.entity_pose.p[2]) - 10 * twist[2], -60, 60)
        torque = np.clip(6 * rotation_error - .5 * twist[3:], -20, 20)
        wrench = np.r_[force_xy, vertical_force, torque]
        # ManiSkill disables link gravity for the Panda's compensated actuators.
        effort = self.robot.compute_passive_force(gravity=False, coriolis_and_centrifugal=True)
        effort[:7] += jacobian.T @ wrench
        self.robot.set_qf(effort)
        self.env.scene.step()

    def step(self, action):
        self.contacts = []
        force_mode = bool(action[7])
        if force_mode != self.force_mode:
            for joint, (stiffness, damping, limit, mode) in zip(self.arm_joints, self.base_drives):
                joint.set_drive_properties(0 if force_mode else stiffness,
                                           0 if force_mode else damping, limit, mode)
            self.robot.set_qf(np.zeros(9))
            self.force_mode = force_mode
        if force_mode:
            self.set_grip_cap(self.force_range[0] + (action[4] + 1) * .5 *
                              (self.force_range[1] - self.force_range[0]))
            for finger in self.fingers:
                finger.set_drive_target(self.width_range[0] + (action[3] + 1) * .5 *
                                        (self.width_range[1] - self.width_range[0]))
            for _ in range(self.env._sim_steps_per_control):
                self._force_substep(action[5:7])
            self.env._contacts_stale = False
        else:
            super().step(action[:5])
        return np.asarray(self.contacts, dtype=np.float32)


def position_action(motion, target, opening):
    action = np.zeros(8, dtype=np.float32)
    action[:3] = np.clip((target - motion.tcp.entity_pose.p) / motion.ee_scale, -1, 1)
    action[3:5] = [opening, .5]
    return action


def shove_actions(direction, force_n, horizon, cfg):
    """Generate the exact open-loop force pulse, withdrawal and holding commands."""
    direction = np.asarray(direction, dtype=np.float64)
    if direction.shape != (2,) or not np.isclose(np.linalg.norm(direction), 1, atol=1e-6):
        raise ValueError('Shove direction must be a unit 2-vector')
    if not 0 <= force_n <= cfg['shove_force_cap_n']:
        raise ValueError('Requested force exceeds the collection cap')
    pulse = round(cfg['shove_duration_s'] * cfg['control_hz'])
    if horizon < pulse + cfg['retract_steps']:
        raise ValueError('Shove horizon must include withdrawal')
    actions = np.zeros((horizon, 8), dtype=np.float32)
    actions[:, 3:5] = [-1, .5]
    actions[:pulse, 5:7] = force_n * direction
    actions[:pulse, 7] = 1
    actions[pulse:pulse + cfg['retract_steps'], :2] = -cfg['retract_xy_action'] * direction
    actions[pulse:pulse + cfg['retract_steps'], 2] = cfg['retract_z_action']
    return actions


def record_episode(motion, cfg, task, seed, rng, measure, *, angle=None, direction=None, force_n=None):
    """Keep all attempts, including absent contact or an unmet rest criterion."""
    horizon = cfg['task_steps'][task]
    raw = motion.state()
    links = [joint.child_link.index for joint in motion.arm_joints + motion.fingers]
    initial = raw[26:29].copy()
    states = [measure(motion, raw, links, first=True)]
    actions, contacts, phases = [], [], []
    if direction is None:
        if angle is None:
            angle = rng.uniform(-np.pi, np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
    direction = np.asarray(direction, dtype=np.float64)
    lift_speed = rng.uniform(*cfg['lift_speed_m_s'])
    lift_height = rng.uniform(*cfg['lift_height_m'])
    slide_speed = rng.uniform(*cfg['slide_speed_m_s'])
    sampled_force = rng.uniform(*cfg['shove_force_range_n'])
    if force_n is None:
        force_n = sampled_force
    pulse_actions = None
    target = initial.copy()
    target[:2] -= cfg['approach_offset_m'] * direction
    target[2] = .12
    touch_step = -1
    force_start = -1
    release_step = -1
    touched = False
    impulse_steps = round(cfg['shove_duration_s'] * cfg['control_hz'])
    for step in range(horizon):
        phase = 0
        if task == 0:
            target = initial.copy()
            target[2] += .08 if step < 10 else .003
            opening = 1 if step < 20 else -1
            if step >= 30:
                phase = 1
                target[2] += min(lift_height, (step - 29) * lift_speed / cfg['control_hz'])
            action = position_action(motion, target, opening)
        elif touch_step < 0:
            if step >= 12:
                target[2] = .023
            if step >= 24:
                target[:2] += cfg['touch_speed_m_s'] * direction / cfg['control_hz']
            action = position_action(motion, target, -1)
        elif task == 1:
            phase = 1
            target[:2] += slide_speed * direction / cfg['control_hz']
            action = position_action(motion, target, -1)
        else:
            phase = 1 if step < release_step else 2
            action = pulse_actions[step - force_start]
        contact = motion.step(action)
        raw = motion.state()
        states.append(measure(motion, raw, links, first=False))
        actions.append(action)
        contacts.append(contact)
        phases.append(phase)
        if task > 0 and touch_step < 0 and step >= 24:
            touched = np.linalg.norm(contact[-1, :2]) > cfg['touch_force_threshold_n']
            # Reserve the pulse and a long free-coast period even on missed touch.
            if touched or step == cfg['touch_timeout_step']:
                touch_step = step
                if task == 2:
                    force_start = step + 1
                    release_step = force_start + impulse_steps
                    pulse_actions = shove_actions(direction, force_n, horizon - force_start, cfg)
    states = np.asarray(states, dtype=np.float32)
    contacts = np.asarray(contacts, dtype=np.float32)
    tail = round(cfg['rest_duration_s'] * cfg['control_hz'])
    rest = (task == 2 and release_step < horizon - tail
            and np.all(np.linalg.norm(states[-tail:, 28:31], axis=1) < cfg['speed_threshold_m_s'])
            and np.all(np.linalg.norm(states[-tail:, 31:34], axis=1) < cfg['rest_angular_speed_rad_s'])
            and np.all(np.linalg.norm(contacts[-tail:], axis=2) < cfg['rest_contact_threshold_n']))
    return dict(
        state=states[:-1], action=np.asarray(actions), next_state=states[1:],
        cube_moving=np.linalg.norm(states[:-1, 28:31], axis=1) > cfg['speed_threshold_m_s'],
        phase=np.asarray(phases, dtype=np.int8),
        contact_force_substeps_n=contacts,
        contact_impulse_ns=contacts.sum(axis=1) / cfg['physx_hz'],
        contact_force_max_n=np.linalg.norm(contacts[:, :, :2], axis=2).max(axis=1),
        force_start_step=np.full(horizon, force_start, np.int32),
        shove_release_step=np.full(horizon, release_step, np.int32),
        stop_step=np.full(horizon, horizon - 1, np.int32),
        observed_stopped=np.full(horizon, rest, bool),
        touched=np.full(horizon, touched, bool),
        reset_seed=np.full(horizon, seed, np.int64),
        motion_direction=np.tile(direction, (horizon, 1)),
        lift_speed_m_s=np.full(horizon, lift_speed, np.float32),
        lift_height_m=np.full(horizon, lift_height, np.float32),
        slide_speed_m_s=np.full(horizon, slide_speed, np.float32),
    )


def simulate_shove(mass, friction, reset_seed, direction, force_n, cfg=None):
    """Repeat a recorded shove setup and execute a chosen open-loop force pulse."""
    from robot_adaptation.simulation.context_dynamics import make_env
    from .collect import configuration, measure

    if cfg is None:
        cfg = configuration()
    env = make_env('PickCube-v1', cfg, cfg['grip_force_limits_n'])
    try:
        motion = ScriptedMotion(env, cfg['grip_force_limits_n'], cfg['finger_friction'])
        motion.set_episode(np.array([mass, friction, cfg['table_friction'], 1, 1, 1, 0, 0, 0, 0]),
                           reset_seed)
        return record_episode(motion, cfg, 2, reset_seed, np.random.default_rng(reset_seed),
                              measure, direction=direction, force_n=force_n)
    finally:
        motion.close()
