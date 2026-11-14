"""Record noisy policy lifts and scripted pushes for cubes with persistent physics."""
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
    import torch
    from experiments.context_dynamics.collect import motion_command, policy_obs
    from robot_adaptation.simulation.context_dynamics import PandaMotion, make_env

    split = 'train' if cube < cfg['train_cubes'] else 'test'
    path = folder / split / f'cube_{cube:04}.npz'
    mass, friction = physics(cfg['seed'], cube, cfg)
    constants = np.array([mass, friction, cfg['table_friction'], 1, 1, 1, 0, 0, 0, 0])
    limits = cfg['grip_force_limits_n']
    env = make_env('PickCube-v1', cfg, limits)
    try:
        motion = PandaMotion(env, limits, cfg['finger_friction'])
        links = [j.child_link.index for j in motion.arm_joints + motion.fingers]
        if (env.control_freq, env.sim_config.sim_freq) != (cfg['control_hz'], cfg['physx_hz']):
            raise ValueError('Unexpected simulator step duration')
        weight_guess = cfg['actor_mass_guess_kg'] * np.linalg.norm(env.scene.px.get_config().gravity)
        horizon = cfg['steps']
        n = 2 * EPISODES * horizon
        data = dict(cube_id=np.full(n, cube, np.int32),
                    task=np.repeat(np.arange(2, dtype=np.int8), EPISODES * horizon),
                    episode_id=np.tile(np.repeat(np.arange(EPISODES, dtype=np.int16), horizon), 2),
                    step=np.tile(np.arange(horizon, dtype=np.int32), 2 * EPISODES),
                    state=np.empty((n, 43), np.float32), action=np.empty((n, 5), np.float32),
                    next_state=np.empty((n, 43), np.float32),
                    true_mass=np.full(n, mass, np.float32),
                    true_friction=np.full(n, friction, np.float32))
        with torch.inference_mode():
            for task in range(2):
                for ep in range(EPISODES):
                    rng = np.random.default_rng([cfg['seed'], cube, 1, task, ep])
                    motion.set_episode(constants, int(rng.integers(2**31)))
                    raw = motion.state()
                    state = measure(motion, raw, links, first=True)
                    anchor = raw[26:29].copy()
                    goal = env.goal_site.pose.p[0].numpy().copy()
                    noise = rng.normal(0, cfg['noise_sd'], size=(horizon, 5))
                    for step in range(horizon):
                        row = (task * EPISODES + ep) * horizon + step
                        if task == 0:
                            obs = policy_obs(motion, raw, weight_guess, goal, step == 0)
                            command = _actor.get_action(torch.from_numpy(obs[None]))[0].numpy()
                        else:
                            command = motion_command('PushCube-v1', step, raw, anchor, rng, None, None)
                        action = np.clip(command + noise[step], -1, 1).astype(np.float32)
                        data['state'][row] = state
                        data['action'][row] = action
                        motion.step(action)
                        raw = motion.state()
                        state = measure(motion, raw, links, first=False)
                        data['next_state'][row] = state
        data['cube_moving'] = np.linalg.norm(data['state'][:, 28:31], axis=1) > cfg['speed_threshold_m_s']
        if not all(np.isfinite(a).all() for a in data.values()):
            raise ValueError(f'Nonfinite measurement for cube {cube}')
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            np.savez(stream, **data)
        tmp.replace(path)
        return cube
    finally:
        env.close()


def start_worker(path):
    import torch
    from robot_adaptation.policy.frozen_v1 import load_agent
    global _actor
    torch.set_num_threads(1)
    _actor, _ = load_agent(path)


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


def collect(data_directory, seed=42, workers=8, train_cubes=200, test_cubes=30, steps=50):
    from robot_adaptation.simulation import context_dynamics
    from experiments.context_dynamics import collect as context_collect

    if min(workers, train_cubes, test_cubes, steps) < 1 or seed < 0:
        raise ValueError('Counts must be positive and seed must be nonnegative')
    path, digest = checkpoint()
    trained = json.loads(path.with_name('manifest.json').read_text())['settings']
    sources = [Path(__file__), Path(context_dynamics.__file__), Path(context_collect.__file__)]
    cfg = dict(seed=seed, train_cubes=train_cubes, test_cubes=test_cubes, episodes_per_task=EPISODES,
               steps=steps, control_hz=20, physx_hz=100, speed_threshold_m_s=.01, noise_sd=.05,
               mass_range_kg=[min(trained['masses_kg']), max(trained['masses_kg'])],
               friction_range=sorted(trained['cube_friction_range']), sampling='independent uniform per cube',
               table_friction=.3, finger_friction=trained['finger_friction_coefficient'],
               actor_mass_guess_kg=.04, goal_clearance_m=.20, grip_force_limits_n=trained['grip_force_limits_n'],
               policy_checkpoint=str(path.relative_to(ROOT)), policy_sha256=digest, task=['lift', 'slide'],
               action=['dx', 'dy', 'dz', 'finger_opening', 'strength'], state=STATE,
               cube_moving='pre-action cube linear speed exceeds speed_threshold_m_s',
               initial_effort='zero before first simulated step of each episode',
               code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               versions={k: version(k) for k in ('torch', 'numpy', 'sapien', 'mani_skill')})
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
        with mp.get_context('fork').Pool(min(workers, len(missing)), initializer=start_worker,
                                         initargs=(path,)) as pool:
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
    cli.add_argument('--steps', type=int, default=50)
    collect(**vars(cli.parse_args()))
