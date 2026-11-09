"""Record post-action proprioception and the preceding applied action."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from robot_adaptation.policy.frozen_v1 import load_agent
from robot_adaptation.collection.collect import write_json
from robot_adaptation.estimators.interaction.data import sha256
from robot_adaptation.simulation.collection_lift import WeightedCollectionEnv


def feature(state, applied):
    # qpos (including fingers), qvel, two measured compression forces, five commands.
    return np.concatenate((state[..., :18], state[..., 43:45], applied), axis=-1)


def policy_and_limits(config):
    manifest = json.loads((Path(config['source']) / 'manifest.json').read_text())
    checkpoint = Path(manifest['checkpoint'])
    assert sha256(checkpoint) == manifest['checkpoint_sha256']
    policy, trained = load_agent(checkpoint)
    return policy, trained['settings']['grip_force_limits_n']


def make_env(config, mass_g, limits):
    properties = dict(mass_kg=mass_g / 1000, static_friction=config['friction'],
                      dynamic_friction=config['friction'], finger_friction=.2, table_friction=.3)
    return WeightedCollectionEnv(properties, reported_mass_kg=config['probe_g']/1000,
                                 goal_clearance_m=config['goal_clearance_m'], grip_force_limits_n=limits)


@torch.inference_mode()
def episode(env, policy, seed, horizon, supplied_g, update=None):
    obs, _ = env.reset(seed=int(seed), options={'reconfigure': True})
    initial = obs[0].numpy().copy()
    physical = env.physics_readback()
    gravity = np.linalg.norm(physical['gravity_world_m_s2'])
    observations, inputs, success = [], [], []
    for step in range(horizon):
        obs = obs.clone()
        obs[:, 42] = supplied_g * gravity / 1000
        inputs.append(supplied_g)
        applied = policy.get_action(obs).clamp(-1, 1)
        obs, _, _, _, info = env.step(applied)
        observations.append(feature(obs[0].numpy(), applied[0].numpy()))
        success.append(bool(info['success'][0]))
        if update is not None:
            supplied_g = update(step, observations)
    return dict(x=np.asarray(observations), success=np.asarray(success), supplied_g=np.asarray(inputs),
                mass_g=1000*float(physical['mass_kg']), seed=seed, initial=initial)


def collect(config_path, out):
    c = json.loads(config_path.read_text())
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'config.json', c)
    policy, limits = policy_and_limits(c)
    torch.set_num_threads(1)
    for split in ['train','validation','calibration','test']:
        rng = np.random.default_rng(c[split+'_seed'])
        count = c[split+'_episodes'] if split != 'test' else c['sequences']*c['trials']
        folder = out/split
        folder.mkdir()
        # Stratified uniform mass draws, independent of the reset seed order.
        low, high = c['mass_range_g']
        masses = low + (high-low)*(rng.permutation(count)+rng.random(count))/count
        for i in range(count):
            mass = c['heavy_g'] if split=='test' else masses[i]
            env = make_env(c, mass, limits)
            try:
                supplied = rng.uniform(low,high) if split=='train' and i%2 else c['probe_g']
                data = episode(env, policy, c[split+'_seed']+i, c['horizon'], supplied)
                assert abs(data['mass_g']-mass)<.01
                np.savez_compressed(folder/f'{i:04d}.npz', **data)
            finally:
                env.close()
        print(split, count, flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/mass_belief.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    collect(args.config,args.output)
