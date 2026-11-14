"""Matched grip-force experiments with TCP-relative cube motion and finger errors."""
import argparse
import fcntl
from functools import partial
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from .collect import EPISODES, configuration, measure, physics, start_worker
from .scripted import ScriptedMotion, record_episode
from robot_adaptation.simulation.context_dynamics import make_env


FIXED_GRIP_CAP_N = 30.0625
PILOT_CAPS_N = (.25, .5, 1., 2., 4., 8., 16., 30.)
EXTRA_STATE = ('relative_position', 'relative_velocity', 'finger_position_error')


class LoggedGripMotion(ScriptedMotion):
    """Write the requested grip strength into the same action array that is saved."""

    def __init__(self, env, cfg, grip_cap_n=None, cap_switch_step=0):
        super().__init__(env, cfg['grip_force_limits_n'], cfg['finger_friction'])
        self.grip_cap_n = grip_cap_n
        self.cap_switch_step = cap_switch_step
        self.step_index = 0
        self.sensor_states = []
        self.issued_caps = []

    def step(self, action):
        if self.grip_cap_n is not None:
            cap = FIXED_GRIP_CAP_N if self.step_index < self.cap_switch_step else self.grip_cap_n
            low, high = self.force_range
            if not low <= cap <= high:
                raise ValueError('Requested finger cap is outside the controller range')
            action[4] = 2 * (cap-low) / (high-low) - 1
        contact = super().step(action)
        self.issued_caps.append([joint.force_limit for joint in self.fingers])
        self.step_index += 1
        return contact

    def measured_state(self, motion, raw, links, first):
        state = measure(motion, raw, links, first)
        pose = self.tcp.entity_pose
        rotation = pose.to_transformation_matrix()[:3, :3]
        jacobian = self.ik.pmodel.compute_single_link_local_jacobian(self.robot.qpos, self.ik.end_link_idx)
        local_twist = jacobian @ self.robot.qvel
        tcp_velocity = rotation @ local_twist[:3]
        tcp_omega = rotation @ local_twist[3:]
        difference = self.body.entity_pose.p - pose.p
        relative_position = rotation.T @ difference
        relative_velocity_instantaneous = rotation.T @ (
            self.body.linear_velocity - tcp_velocity - np.cross(tcp_omega, difference))
        relative_velocity = np.zeros(3, dtype=np.float32)
        if self.sensor_states:
            relative_velocity = (relative_position-self.sensor_states[-1]['relative_position']) * self.env.control_freq
        target = np.asarray([np.asarray(joint.drive_target).item() for joint in self.fingers])
        finger_error = target - self.robot.qpos[-2:]
        normal = []
        for link, sign in ((self.env.agent.finger1_link, 1), (self.env.agent.finger2_link, -1)):
            reported_force = np.zeros(3) if first else self.env.scene.get_pairwise_contact_forces(link, self.obj)[0].numpy()
            closing_axis = sign * link._bodies[0].entity_pose.to_transformation_matrix()[:3, 1]
            normal.append(max(0., float(reported_force @ closing_axis)))
        self.sensor_states.append(dict(
            relative_position=relative_position,
            relative_velocity=relative_velocity,
            relative_velocity_instantaneous=relative_velocity_instantaneous,
            finger_position_error=finger_error,
            tcp_pose=np.r_[pose.p, pose.q],
            tcp_linear_velocity=tcp_velocity,
            tcp_angular_velocity=tcp_omega,
            finger_target_position=target,
            finger_contact_normal_n=np.asarray(normal),
            finger_drive_cap_n=np.asarray([joint.force_limit for joint in self.fingers])))
        return state


def record_probe(mass, friction, task, reset_seed, grip_cap_n, *, motion_seed=0, cfg=None,
                 motion_rng=None, cap_switch_step=0):
    """Record one fresh scene. Supply the original RNG after its reset-seed draw for pairing."""
    if cfg is None:
        cfg = configuration()
    rng = np.random.default_rng(motion_seed) if motion_rng is None else motion_rng
    env = make_env('PickCube-v1', cfg, cfg['grip_force_limits_n'])
    motion = LoggedGripMotion(env, cfg, None if task == 2 else grip_cap_n, cap_switch_step)
    try:
        motion.set_episode(np.array([mass, friction, cfg['table_friction'], 1, 1, 1, 0, 0, 0, 0]), reset_seed)
        result = record_episode(motion, cfg, task, reset_seed, rng, motion.measured_state)
        for field in motion.sensor_states[0]:
            values = np.asarray([row[field] for row in motion.sensor_states], dtype=np.float32)
            result[field] = values[:-1]
            result['next_' + field] = values[1:]
        result['issued_grip_force_cap_n'] = np.asarray(motion.issued_caps, dtype=np.float32)
        low, high = cfg['grip_force_limits_n']
        expected = low + (result['action'][:, 4] + 1) * .5 * (high-low)
        np.testing.assert_allclose(result['issued_grip_force_cap_n'], np.tile(expected[:, None], (1, 2)),
                                   rtol=0, atol=3e-6)
        return result
    finally:
        motion.close()


def project_rows(rows):
    """Append eight measured state values, preserving all row metadata."""
    projected = dict(rows)
    for prefix in ('', 'next_'):
        projected[prefix+'state'] = np.concatenate(
            [rows[prefix+'state'], *(rows[prefix+field] for field in EXTRA_STATE)], axis=1).astype(np.float32)
    return projected


def classify_lift(rows):
    """Separate established-grasp slip from failure to acquire an airborne grasp."""
    state = rows['next_state']
    height = state[:, 21] - rows['state'][0, 21]
    bilateral = (rows['next_finger_contact_normal_n'] > .1).all(axis=1)
    stable_relative_motion = np.linalg.norm(rows['next_relative_velocity'], axis=1) < .02
    acquired_sample = bilateral & (height >= .005) & stable_relative_motion
    consecutive = np.convolve(acquired_sample.astype(int), np.ones(3, dtype=int), mode='valid')
    candidates = np.flatnonzero(consecutive == 3)
    acquired = int(candidates[0]+2) if len(candidates) else -1
    vertical_gap = state[:, 21] - rows['next_tcp_pose'][:, 2]
    previous_vertical_gap = rows['state'][:, 21] - rows['tcp_pose'][:, 2]
    relative_vertical_speed = (vertical_gap-previous_vertical_gap) * 20
    drop = np.zeros(len(state))
    slip_step = -1
    rotation_change = 0.
    rotation_at_slip = 0.
    rotation_before_slip = 0.
    rotation_through_slip_window = 0.
    if acquired >= 0:
        drop[acquired:] = vertical_gap[acquired] - vertical_gap[acquired:]
        slipping = (drop >= .005) & (relative_vertical_speed < -.01)
        slipping[:acquired+1] = False
        events = np.flatnonzero(slipping)
        if len(events):
            slip_step = int(events[0])
        columns = state[:, 22:28].reshape(-1, 2, 3)
        rotation = np.stack((columns[:, 0], columns[:, 1], np.cross(columns[:, 0], columns[:, 1])), axis=2)
        relative = rotation[acquired:].transpose(0, 2, 1) @ rotation[acquired]
        cosine = np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1)
        angles = np.rad2deg(np.arccos(cosine))
        rotation_change = float(angles.max())
        if slip_step >= 0:
            rotation_at_slip = float(angles[slip_step-acquired])
            rotation_before_slip = float(angles[:slip_step-acquired+1].max())
            rotation_through_slip_window = float(angles[:slip_step-acquired+2].max())
    outcome = 'held_after_acquisition'
    if acquired < 0:
        outcome = 'failed_pickup'
    elif slip_step >= 0:
        outcome = 'rotating_loss_after_acquisition' if rotation_through_slip_window > 20 else 'slip_after_acquisition'
    return dict(outcome=outcome, acquisition_step=acquired, slip_step=slip_step,
                maximum_cube_rise_m=float(height.max()),
                maximum_downward_relative_drift_after_acquisition_m=float(drop.max()),
                maximum_cube_rotation_after_acquisition_deg=rotation_change,
                cube_rotation_at_first_slip_deg=rotation_at_slip,
                maximum_rotation_through_slip_onset_deg=rotation_before_slip,
                maximum_rotation_through_slip_and_next_step_deg=rotation_through_slip_window,
                rotation_over_20deg_after_acquisition=rotation_change > 20,
                bilateral_contact_steps=int(bilateral.sum()),
                maximum_finger_position_error_m=float(np.abs(rows['next_finger_position_error']).max()))


def _pilot_case(job, output, cfg, cap_switch_step):
    mass, friction, reset_seed, grip_cap_n, task = job
    rows = record_probe(mass, friction, task, reset_seed, grip_cap_n,
                        motion_seed=reset_seed, cfg=cfg, cap_switch_step=cap_switch_step)
    name = f'task_{task}_m_{mass:.2f}_mu_{friction:.1f}_seed_{reset_seed}_cap_{grip_cap_n:g}.npz'
    np.savez_compressed(output / name, **rows)
    report = dict(file=name, mass_kg=mass, friction=friction, reset_seed=reset_seed,
                  grip_cap_n=grip_cap_n, task=task, cap_switch_step=cap_switch_step)
    if task == 0:
        report.update(classify_lift(rows))
    else:
        displacement = rows['next_state'][-1, 19:21] - rows['state'][0, 19:21]
        report.update(
            horizontal_travel_m=float(np.linalg.norm(displacement)),
            signed_travel_m=float(displacement @ rows['motion_direction'][0]),
            contact_steps=int((rows['contact_force_max_n'] > .01).sum()),
            relative_position_change_m=float(np.linalg.norm(rows['next_relative_position'][-1] -
                                                           rows['relative_position'][0])),
            bilateral_contact_steps=int((rows['next_finger_contact_normal_n'] > .1).all(axis=1).sum()))
    return report


def pilot(output, workers=8, cap_switch_step=0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = configuration()
    protocol = dict(config=cfg, cap_switch_step=cap_switch_step, grip_caps_n=PILOT_CAPS_N,
                    masses_kg=[.04, .5], frictions=[.4, 1.], reset_seeds=[6101, 6102, 6103],
                    acquisition='Bilateral reported finger-normal projections above 0.1 N, cube rise at least 5 mm and cube-relative TCP speed below0.02m/s for3consecutivecontrolsteps',
                    slip='After acquisition, cube-TCP world vertical gap falls by at least 5 mm with relative vertical velocity below -0.01 m/s',
                    rotation='Maximum cube rotation from acquisition through the first slip event plus one control step must be at most20degrees for a clean slip label; later rotation is separately recorded',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    jobs = [(mass, friction, seed, cap, task)
            for task in (0, 1) for mass in (.04, .5) for friction in (.4, 1.)
            for seed in (6101, 6102, 6103) for cap in PILOT_CAPS_N]
    worker = partial(_pilot_case, output=output, cfg=cfg, cap_switch_step=cap_switch_step)
    with mp.get_context('spawn').Pool(workers, initializer=start_worker) as pool:
        records = list(pool.imap_unordered(worker, jobs))
    records.sort(key=lambda row: (row['task'], row['mass_kg'], row['friction'], row['reset_seed'], row['grip_cap_n']))
    counts = {}
    for record in records:
        if record['task'] == 0:
            counts[record['outcome']] = counts.get(record['outcome'], 0)+1
    report = dict(episodes=len(records), lift_outcomes=counts, records=records)
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(episodes=len(records), lift_outcomes=counts)), flush=True)
    return report


def grip_cap_for_episode(seed, cube, task, episode):
    """Cover five log-force bins in each block of five episodes, without physics labels."""
    edges = np.linspace(np.log(.25), np.log(FIXED_GRIP_CAP_N), 6)
    rng = np.random.default_rng([seed, cube, 917, task, episode // 5])
    caps = np.exp(rng.uniform(edges[:-1], edges[1:]))
    return float(rng.permutation(caps)[episode % 5])


def _paired_cube(cube, root, cfg):
    split = 'train' if cube < cfg['train_cubes'] else 'test'
    mass, friction = physics(cfg['seed'], cube, cfg)
    episodes = {'fixed': [], 'random': []}
    for task in range(3):
        for episode in range(EPISODES):
            fixed_rng = np.random.default_rng([cfg['seed'], cube, 1, task, episode])
            reset_seed = int(fixed_rng.integers(2**31))
            fixed = record_probe(mass, friction, task, reset_seed, FIXED_GRIP_CAP_N,
                                 motion_rng=fixed_rng, cfg=cfg, cap_switch_step=45)
            arms = {'fixed': fixed}
            if task == 2:
                arms['random'] = fixed
            else:
                random_rng = np.random.default_rng([cfg['seed'], cube, 1, task, episode])
                assert int(random_rng.integers(2**31)) == reset_seed
                cap = grip_cap_for_episode(cfg['seed'], cube, task, episode)
                arms['random'] = record_probe(mass, friction, task, reset_seed, cap,
                                              motion_rng=random_rng, cfg=cfg, cap_switch_step=45)
            for arm, rows in arms.items():
                n = len(rows['action'])
                labels = classify_lift(rows) if task == 0 else dict(acquisition_step=-1, slip_step=-1, outcome='')
                rows = dict(rows,
                            cube_id=np.full(n, cube, np.int32),
                            task=np.full(n, task, np.int8),
                            episode_id=np.full(n, episode, np.int16),
                            step=np.arange(n, dtype=np.int32),
                            true_mass=np.full(n, mass, np.float64),
                            true_friction=np.full(n, friction, np.float64),
                            grasp_acquired=np.full(n, labels['acquisition_step'] >= 0),
                            slip_after_acquisition=np.full(n, labels['outcome'] == 'slip_after_acquisition'),
                            rotating_loss_after_acquisition=np.full(n, labels['outcome'] == 'rotating_loss_after_acquisition'),
                            acquisition_step=np.full(n, labels['acquisition_step'], np.int32),
                            slip_step=np.full(n, labels['slip_step'], np.int32))
                episodes[arm].append(rows)
    for arm, parts in episodes.items():
        data = {key: np.concatenate([episode[key] for episode in parts]) for key in parts[0]}
        if not all(np.isfinite(value).all() for value in data.values()):
            raise ValueError(f'Nonfinite cube data for cube {cube}, arm {arm}')
        assert project_rows(data)['state'].shape[1] == 51
        path = root / arm / 'data' / split / f'cube_{cube:04}.npz'
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, **data)
        temporary.replace(path)
    return cube


def collect(root, seed=42, workers=8, train_cubes=200, test_cubes=30):
    """Collect paired fixed/random arms, sharing each unchanged shove trajectory."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = configuration(seed, train_cubes, test_cubes)
    cfg.update(grip_probe=dict(
        fixed_cap_n=FIXED_GRIP_CAP_N, randomized_tasks=[0, 1], cap_switch_step=45,
        random_cap_range_n=[.25, FIXED_GRIP_CAP_N], log_bins=5,
        log_bin_edges_n=np.exp(np.linspace(np.log(.25), np.log(FIXED_GRIP_CAP_N), 6)).tolist(),
        sampling='One log-uniform draw in each of5bins, independently permuted within episodes0..4 and5..9 per cube/task',
        cap_rng='SeedSequence [seed,cube_id,917,task,episode_id//5]',
        acquisition_cap_n=FIXED_GRIP_CAP_N,
        shared_shove_trajectories=True),
        raw_state_dimension=43, projected_state_dimension=51, projected_additions=list(EXTRA_STATE),
        relative_velocity='Causal control-interval difference (relative_position[t]-relative_position[t-1])*20; reset measurement zero',
        relative_velocity_instantaneous='R_tcp.T @ (v_cube-v_tcp-omega_tcp cross (p_cube-p_tcp)), retained as diagnostic only',
        finger_error='drive_target - joint_position, two independent finger coordinates',
        finger_normal='Reported normal-contact force on each finger projected onto its signed local Y axis',
        slip_criteria=dict(bilateral_normal_threshold_n=.1, airborne_height_m=.005,
                           acquisition_relative_speed_m_s=.02, consecutive_acquisition_steps=3,
                           downward_relative_gap_m=.005, downward_relative_velocity_m_s=.01,
                           maximum_rotation_through_event_and_next_step_deg=20))
    cfg['code_sha256']['experiments/lift_slide_belief/slip_data.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (root / '.collect.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for arm in ('fixed', 'random'):
            folder = root / arm / 'data'
            for split in ('train', 'test'):
                (folder / split).mkdir(parents=True, exist_ok=True)
            manifest = folder / 'dataset.json'
            protocol = dict(cfg, grip_arm=arm)
            if manifest.exists() and json.loads(manifest.read_text()) != protocol:
                raise ValueError(f'Existing {arm} data use another protocol')
            manifest.write_text(json.dumps(protocol, indent=2)+'\n')
        total = train_cubes+test_cubes
        missing = []
        for cube in range(total):
            split = 'train' if cube < train_cubes else 'test'
            if not all((root / arm / 'data' / split / f'cube_{cube:04}.npz').exists() for arm in ('fixed','random')):
                missing.append(cube)
        with mp.get_context('spawn').Pool(workers, initializer=start_worker) as pool:
            worker = partial(_paired_cube, root=root, cfg=cfg)
            for done, _ in enumerate(pool.imap_unordered(worker, missing), total-len(missing)+1):
                if done % 10 == 0 or done == total:
                    print(f'{done}/{total} paired cubes saved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--cap-switch-step', type=int, default=0)
    parser.add_argument('--collect-pair', action='store_true')
    parser.add_argument('--train-cubes', type=int, default=200)
    parser.add_argument('--test-cubes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.collect_pair:
        collect(args.output, args.seed, args.workers, args.train_cubes, args.test_cubes)
    else:
        pilot(args.output, args.workers, args.cap_switch_step)
