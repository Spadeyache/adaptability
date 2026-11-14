"""Record scripted lifts, slow slides and bounded-force robot shoves."""
import argparse
import fcntl
from functools import partial
import hashlib
from importlib.metadata import version
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / 'runs/failure_collection/varied_fixed40g/manifest.json'
EPISODES = 10
STATE = {
    'joint_positions': [0, 9], 'joint_velocities': [9, 18], 'gripper_width_m': [18, 19],
    'cube_position_m': [19, 22], 'cube_rotation_matrix_columns_1_and_2': [22, 28],
    'cube_linear_velocity_m_per_s': [28, 31], 'cube_angular_velocity_rad_per_s': [31, 34],
    'arm_joint_torque_nm': [34, 41], 'finger_joint_force_n': [41, 43],
}


def checkpoint():
    source = json.loads(SOURCE_MANIFEST.read_text())
    saved = Path(source['checkpoint'])
    # Resolve the retained checkpoint in the current checkout.
    path = ROOT.joinpath(*saved.parts[saved.parts.index('runs'):])
    if hashlib.sha256(path.read_bytes()).hexdigest() != source['checkpoint_sha256']:
        raise ValueError('Lift checkpoint differs from the recorded collection policy')
    return path, source['checkpoint_sha256']


def physics(seed, cube, cfg):
    rng = np.random.default_rng([seed, cube, 0])
    return rng.uniform(*cfg['mass_range_kg']), rng.uniform(*cfg['friction_range'])


def measure(motion, raw, links, first):
    # PhysX returns incoming wrenches in child-joint coordinates. All seven Panda
    # revolute joints and both prismatic joints use the local X axis. Contact
    # and joint-limit reactions contribute to these measured net joint efforts.
    effort = np.zeros(9, np.float32)
    if not first:
        wrench = motion.robot.get_link_incoming_joint_forces()
        effort[:7] = wrench[links[:7], 3]
        effort[7:] = wrench[links[7:], 0]
    rotation = motion.body.entity_pose.to_transformation_matrix()[:3, :2].T.ravel()
    return np.concatenate((raw[:18], raw[25:29], rotation, raw[33:39], effort))


def record_cube(cube, folder, cfg):
    from robot_adaptation.simulation.context_dynamics import make_env
    from .scripted import ScriptedMotion, record_episode

    split = 'train' if cube < cfg['train_cubes'] else 'test'
    path = folder / split / f'cube_{cube:04}.npz'
    mass, friction = physics(cfg['seed'], cube, cfg)
    constants = np.array([mass, friction, cfg['table_friction'], 1, 1, 1, 0, 0, 0, 0])
    limits = cfg['grip_force_limits_n']
    episodes = []
    for task in range(3):
        for ep in range(EPISODES):
            rng = np.random.default_rng([cfg['seed'], cube, 1, task, ep])
            seed = int(rng.integers(2**31))
            env = make_env('PickCube-v1', cfg, limits)
            motion = ScriptedMotion(env, limits, cfg['finger_friction'])
            try:
                if (env.control_freq, env.sim_config.sim_freq) != (cfg['control_hz'], cfg['physx_hz']):
                    raise ValueError('Unexpected simulator step duration')
                motion.set_episode(constants, seed)
                data = record_episode(motion, cfg, task, seed, rng, measure)
                n = len(data['action'])
                data.update(cube_id=np.full(n, cube, np.int32), task=np.full(n, task, np.int8),
                            episode_id=np.full(n, ep, np.int16), step=np.arange(n, dtype=np.int32),
                            true_mass=np.full(n, mass, np.float64),
                            true_friction=np.full(n, friction, np.float64))
                episodes.append(data)
            finally:
                motion.close()
    data = {key: np.concatenate([ep[key] for ep in episodes]) for key in episodes[0]}
    if not all(np.isfinite(a).all() for a in data.values()):
        raise ValueError(f'Nonfinite measurement for cube {cube}')
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez(stream, **data)
    tmp.replace(path)
    return cube


def start_worker():
    import torch
    torch.set_num_threads(1)


def load_rows(data_directory, split):
    if split not in ('train', 'test'):
        raise ValueError('Unknown cube split')
    folder = Path(data_directory)
    cfg = json.loads((folder / 'dataset.json').read_text())
    start = 0 if split == 'train' else cfg['train_cubes']
    count = cfg[split + '_cubes']
    chunks = {}
    for cube in range(start, start + count):
        with np.load(folder / split / f'cube_{cube:04}.npz', allow_pickle=False) as archive:
            for key in archive.files:
                chunks.setdefault(key, []).append(archive[key])
    return {k: np.concatenate(parts) for k, parts in chunks.items()}


def configuration(seed=42, train_cubes=200, test_cubes=30, steps=None):
    from robot_adaptation.simulation import context_dynamics
    from . import scripted
    if min(train_cubes, test_cubes) < 1 or seed < 0 or (steps is not None and steps < 120):
        raise ValueError('Positive cube counts, a nonnegative seed and at least 120 steps are required')
    path, digest = checkpoint()
    trained = json.loads(path.with_name('manifest.json').read_text())['settings']
    sources = [Path(__file__), Path(scripted.__file__), Path(context_dynamics.__file__)]
    return dict(
        schema_version=2, seed=seed, train_cubes=train_cubes, test_cubes=test_cubes,
        episodes_per_task=EPISODES, steps=steps, task_steps=[120, 120, 180] if steps is None else [steps]*3,
        control_hz=20, physx_hz=100, speed_threshold_m_s=.01,
        mass_range_kg=[min(trained['masses_kg']), max(trained['masses_kg'])],
        friction_range=sorted(trained['cube_friction_range']), sampling='independent uniform per cube',
        table_friction=.3, finger_friction=trained['finger_friction_coefficient'],
        goal_clearance_m=.20, grip_force_limits_n=trained['grip_force_limits_n'],
        range_reference_checkpoint=str(path.relative_to(ROOT)), range_reference_sha256=digest,
        task=['lift', 'slow_slide', 'shove'], state=STATE,
        action=['dx', 'dy', 'dz', 'finger_opening', 'strength', 'tcp_force_x_n', 'tcp_force_y_n',
                'cartesian_force_mode'],
        cube_moving='pre-action cube linear speed exceeds speed_threshold_m_s',
        initial_effort='zero before first simulated step of each episode',
        lift_speed_m_s=[.05, .10], lift_height_m=[.07, .15], slide_speed_m_s=[.02, .04],
        approach_offset_m=.065, touch_speed_m_s=.012, touch_force_threshold_n=.05,
        touch_timeout_step=90, shove_force_cap_n=4., shove_force_range_n=[1., 4.], shove_duration_s=.3,
        retract_steps=5, retract_xy_action=.10, retract_z_action=.20,
        rest_duration_s=.5, rest_angular_speed_rad_s=.1, rest_contact_threshold_n=.001,
        force_control=dict(z_m=.023, rotation_matrix=[[1,0,0],[0,-1,0],[0,0,-1]],
                           z_stiffness_n_m=200, z_damping_ns_m=10,
                           rotation_stiffness_nm_rad=6, rotation_damping_nms_rad=.5,
                           force_scope='horizontal commanded TCP wrench; impact forces measured separately'),
        code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        versions={k: version(k) for k in ('torch', 'numpy', 'sapien', 'mani_skill')})


def collect(data_directory, seed=42, workers=8, train_cubes=200, test_cubes=30, steps=None):
    cfg = configuration(seed, train_cubes, test_cubes, steps)
    if workers < 1:
        raise ValueError('workers must be positive')
    folder = Path(data_directory).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.collect.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = folder / 'dataset.json'
        if manifest.exists() and json.loads(manifest.read_text()) != cfg:
            raise ValueError('Output folder contains a different dataset protocol')
        if not manifest.exists():
            tmp = manifest.with_suffix('.tmp')
            tmp.write_text(json.dumps(cfg, indent=2) + '\n')
            tmp.replace(manifest)
        for split in ('train', 'test'):
            (folder / split).mkdir(exist_ok=True)
        count = train_cubes + test_cubes
        missing = [cube for cube in range(count)
                   if not (folder / ('train' if cube < train_cubes else 'test') / f'cube_{cube:04}.npz').exists()]
        if not missing:
            return
        worker = partial(record_cube, folder=folder, cfg=cfg)
        with mp.get_context('fork').Pool(min(workers, len(missing)), initializer=start_worker) as pool:
            for done, _ in enumerate(pool.imap_unordered(worker, missing), count - len(missing) + 1):
                if done % 10 == 0 or done == count:
                    print(f'{done}/{count} cubes saved', flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('data_directory', type=Path)
    cli.add_argument('--seed', type=int, default=42)
    cli.add_argument('--workers', type=int, default=8)
    cli.add_argument('--train-cubes', type=int, default=200)
    cli.add_argument('--test-cubes', type=int, default=30)
    cli.add_argument('--steps', type=int)
    collect(**vars(cli.parse_args()))
