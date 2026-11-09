"""Collect reproducible episode shards for training or the fixed-body deployment test."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from robot_adaptation.policy.frozen_v1 import load_agent
from robot_adaptation.simulation.context_dynamics import CONSTANTS, STATE, TASKS, PandaMotion, make_env


def json_write(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def physics(rng, c, deployment):
    mass = np.exp(rng.uniform(*np.log(c['mass_kg'])))
    mu = rng.uniform(*c['cube_friction'])
    if deployment:
        return np.array([mass, mu, .3, .5, 1.5, 1., 0., 0., 0., 0.])
    params = [rng.uniform(*c[key]) for key in
              ('table_friction', 'grip_scale', 'damping_scale', 'stiffness_scale')]
    delay = rng.integers(c['delay_steps'][0], c['delay_steps'][1]+1)
    return np.r_[mass, mu, params, delay, rng.uniform(*c['com_m'], size=3)]


def policy_obs(motion, state, guessed_weight, goal, first):
    # Historical 45-input interface, including *measured* finger forces.
    # Only the frozen policy can see this vector.
    env = motion.env
    forces, grasped = [], True
    for link, sign in ((env.agent.finger1_link, 1), (env.agent.finger2_link, -1)):
        force = np.zeros(3) if first else env.scene.get_pairwise_contact_forces(link, motion.obj)[0].numpy()
        axis = link._bodies[0].entity_pose.to_transformation_matrix()[:3,1]*sign
        component = force @ axis
        magnitude = np.linalg.norm(force)
        forces.append(max(0., component))
        grasped = grasped and magnitude >= .5 and component >= magnitude*np.cos(np.deg2rad(85.))
    return np.r_[state[:18], grasped, state[18:25], goal, state[26:33],
                 state[26:29]-state[18:21], goal-state[26:29], guessed_weight, forces].astype(np.float32)


def motion_command(task, step, state, anchor, rng, direction, amplitude):
    cube, tcp = state[26:29], state[18:21]
    if task in ('PushCube-v1', 'PullCube-v1'):
        side = 1 if task == 'PushCube-v1' else -1
        # Approach behind the cube, descend, then sweep in the task's goal direction.
        target = anchor.copy()
        target[0] += side*(-.055 if step < 13 else min(.20, -.055+(step-12)*.014))
        target[2] = .09 if step < 6 else .023
        opening, strength = -.9, .65
    else:
        # Pokes include lateral taps as well as close/lift/shake/release phases.
        target = cube.copy() if step < 20 else anchor.copy()
        target[2] = .08 if step < 6 else .022
        opening = 1. if step < 13 or step >= 43 else -.95
        if step >= 20:
            target[2] += min(.12, (step-20)*.016)
            target[:2] += direction * amplitude * np.sin((step-20)*.65)
        strength = -.65 if step < 13 else np.sin(step*.32)
    return np.r_[np.clip((target-tcp)*12, -1, 1), opening, strength]


def grip_probe(step, state, anchor, direction, cap_profile):
    """Explore drive saturation while closing, lifting, unloading and regrasping."""
    cube, tcp = state[26:29], state[18:21]
    target = cube.copy()
    target[2] += .06 if step < 6 else .002
    opening, cap = 1., cap_profile[0]
    if 10 <= step < 41:
        opening = -1.
        if step >= 17:
            target = anchor.copy()
            target[2] += min(.14,.02*(step-16))
            target[:2] += .032*direction*np.sin((step-17)*.78)
        # Both increases and decreases are present and independent of true physics.
        cap = cap_profile[min(3,(step-10)//8)]
    return np.r_[np.clip(14*(target-tcp),-1,1), opening, cap]


def task_index(cfg, episode_id, deployment):
    if deployment:
        return 0
    if cfg.get('motion_variant') == 'grip_probe':
        return TASKS.index('poke')
    if cfg.get('motion_variant') == 'paired':
        pair = episode_id//2
        return (pair + pair//20) % len(TASKS)
    return episode_id % len(TASKS)


@torch.inference_mode()
def record(motion, policy, cfg, episode_id, deployment):
    # Use independent streams for the constants, motion and reset.
    root = [cfg['seed'], int(deployment), episode_id]
    prng, rng, srng = [np.random.default_rng(s) for s in np.random.SeedSequence(root).spawn(3)]
    seed = int(srng.integers(0, 2**31))
    task_id = task_index(cfg, episode_id, deployment)
    use_probe = not deployment and task_id == TASKS.index('poke') and cfg.get('motion_variant') in ('grip_probe','paired')
    task = TASKS[task_id]
    constants = physics(prng, cfg, deployment)
    motion.set_episode(constants, seed)
    guessed_kg = np.exp(rng.uniform(*np.log(cfg['mass_kg'])))
    guessed_weight = guessed_kg * np.linalg.norm(motion.env.scene.px.get_config().gravity)
    horizon = cfg['steps']
    segment = not deployment and rng.random() < .25
    segment_start = rng.integers(6, horizon//2+1) if segment else -1
    segment_end = min(horizon, segment_start+rng.integers(12, 24)) if segment else -1
    angle = rng.uniform(-np.pi, np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    amplitude = rng.uniform(.015, .055)
    actions = np.empty((horizon, 5), np.float32)
    applied = np.empty_like(actions)
    states = np.empty((horizon+1, 39), np.float32)
    states[0] = motion.state()
    anchor = states[0,26:29].copy()
    if task_id == 0:
        goal = motion.env.goal_site.pose.p[0].numpy().copy()
    noise = rng.normal(0, cfg['action_noise_sd'], size=actions.shape)
    random_cmd = np.zeros(5)
    if use_probe:
        # Uniform in log newtons before unknown robot strength scaling.
        caps_n = np.exp(rng.uniform(np.log(.30),np.log(25.),4))
        low, high = motion.force_range
        cap_profile = 2*(caps_n-low)/(high-low)-1
    for t in range(horizon):
        if segment_start <= t < segment_end:
            if (t-segment_start) % 4 == 0:
                random_cmd = rng.uniform(-1, 1, 5)
            cmd = random_cmd
        elif task_id == 0:
            obs = policy_obs(motion, states[t], guessed_weight, goal, t==0)
            cmd = policy.get_action(torch.from_numpy(obs[None]))[0].numpy()
        elif use_probe:
            cmd = grip_probe(t,states[t],anchor,direction,cap_profile)
        else:
            cmd = motion_command(task, t, states[t], anchor, rng, direction, amplitude)
        actions[t] = np.clip(cmd + noise[t], -1, 1)
        applied[t] = motion.step(actions[t])
        states[t+1] = motion.state()
    d = int(constants[6])
    if not np.array_equal(applied[d:], actions[:horizon-d]) or np.any(applied[:d]):
        raise RuntimeError('Logged command sequence disagrees with the selected delay')
    if not np.isfinite(states).all():
        raise ValueError(f'Nonfinite state in episode {episode_id}')
    return dict(episode_id=episode_id, seed=seed, task_id=task_id, constants=constants,
                state=states, command=actions, applied=applied, policy_guess_kg=guessed_kg,
                random_start=segment_start, random_end=segment_end)


def init_worker(cfg, run, deployment):
    global _worker
    torch.set_num_threads(1)
    ref = json.loads(Path(cfg['policy_manifest']).read_text())
    policy, metadata = load_agent(ref['checkpoint'])
    _worker = (cfg, run, deployment, policy, metadata['settings']['grip_force_limits_n'], {})


def record_pair(source, cfg, limits):
    """Replay one command sequence in fresh scenes with one independently varied factor."""
    first = source['episode_id']
    rng = np.random.default_rng([cfg['seed'],first,1467])
    columns = [3,3,0,1,4,9]  # Oversample the weak finger drive signal.
    column = rng.choice(columns)
    bounds = cfg[CONSTANTS[column] if column < 7 else 'com_m']
    low,high = np.log(bounds) if column == 0 else bounds
    mid = (low+high)/2
    settings = rng.permutation([rng.uniform(low,mid),rng.uniform(mid,high)])
    for branch, value in enumerate(settings):
        constants = source['constants'].copy()
        constants[column] = np.exp(value) if column == 0 else value
        env = make_env(TASKS[source['task_id']],cfg,limits)
        try:
            arm = PandaMotion(env,limits,cfg['finger_friction'])
            arm.set_episode(constants,source['seed'])
            states = [arm.state()]
            np.testing.assert_allclose(states[0],source['state'][0],atol=1e-6)
            applied = []
            for command in source['command']:
                applied.append(arm.step(command))
                states.append(arm.state())
            yield dict(source,episode_id=first+branch,constants=constants,state=np.stack(states),
                       applied=np.stack(applied),pair_id=first//2,changed_index=column)
        finally:
            env.close()


def collect_shard(index):
    cfg, run, deployment, policy, limits, environments = _worker
    count = 150 if deployment else cfg['episodes']
    group = cfg['shard_episodes'] if not deployment else 50
    start = index*group
    records = []
    paired = cfg.get('motion_variant')=='paired' and not deployment
    if paired and (group%2 or count%2):
        raise ValueError('Complete physical pairs must remain together')
    for i in range(start, min(count, start+group), 2 if paired else 1):
        task = TASKS[task_index(cfg, i, deployment)]
        scene = 'PickCube-v1' if task=='poke' else task
        if scene not in environments:
            env = make_env(scene, cfg, limits)
            environments[scene] = PandaMotion(env, limits, cfg['finger_friction'])
        source = record(environments[scene], policy, cfg, i, deployment)
        if paired:
            records.extend(record_pair(source,cfg,limits))
        else:
            records.append(source)
    dst = run/f'shard_{index:04}.npz'
    temp = dst.with_suffix(f'.{os.getpid()}.tmp')
    arrays = {key: np.stack([r[key] for r in records]) for key in records[0]}
    with temp.open('wb') as stream:
        np.savez(stream, **arrays)
    temp.replace(dst)
    return index


def collect(config, run, workers, deployment=False):
    cfg = json.loads(config.read_text())
    ref = json.loads(Path(cfg['policy_manifest']).read_text())
    ckpt_hash = hashlib.sha256(Path(ref['checkpoint']).read_bytes()).hexdigest()
    if ckpt_hash != ref['checkpoint_sha256']:
        raise ValueError('Frozen motion policy differs from the reference sweep')
    run.mkdir(parents=True, exist_ok=True)
    schema = dict(config=cfg, policy_sha256=ckpt_hash, deployment=deployment, tasks=TASKS,
                  constants=CONSTANTS, state=STATE, command=['dx','dy','dz','opening','strength'],
                  split='validation if (episode_id // 4) % 10 == 9; whole episodes')
    if (run/'dataset.json').exists() and json.loads((run/'dataset.json').read_text()) != json.loads(json.dumps(schema)):
        raise ValueError('Incompatible existing dataset')
    source = run/'source'
    for path in [Path(__file__), Path('robot_adaptation/simulation/context_dynamics.py')]:
        dest = source/path.name
        if dest.exists() and dest.read_bytes() != path.read_bytes():
            raise ValueError('Collection code changed. Use a new output directory.')
        source.mkdir(exist_ok=True)
        shutil.copyfile(path,dest)
    json_write(run/'dataset.json',schema)
    count, group = (150,50) if deployment else (cfg['episodes'],cfg['shard_episodes'])
    chunks = (count+group-1)//group
    missing = [i for i in range(chunks) if not (run/f'shard_{i:04}.npz').exists()]
    with mp.get_context('fork').Pool(workers, initializer=init_worker, initargs=(cfg,run,deployment)) as pool:
        for done, i in enumerate(pool.imap_unordered(collect_shard, missing), chunks-len(missing)+1):
            if done % 10 == 0 or done == chunks:
                print(f'{done}/{chunks} completed shards',flush=True)
    json_write(run/'complete.json',dict(episodes=count, transition_count=count*cfg['steps'], shards=chunks))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--config',type=Path,default=Path('configs/context_dynamics.json'))
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--deployment',action='store_true')
    args=p.parse_args()
    collect(args.config,args.output,args.workers,args.deployment)
