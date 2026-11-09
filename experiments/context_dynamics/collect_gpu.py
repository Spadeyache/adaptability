"""Collect a pilot or full PickCube dataset on the local GPU."""
import argparse
import fcntl
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from robot_adaptation.policy.frozen_v1 import load_agent
from robot_adaptation.simulation.context_dynamics import CONSTANTS, STATE
from robot_adaptation.simulation.gpu_pick import RandomizedPickGPU
from .collect import json_write, physics
from .verify_gpu import SOURCES, inspect_shard, verify_dataset


def episode_draws(ids, seed, cfg):
    """Separate random streams for physics, guesses, motion choice, and issued action noise."""
    requested, guesses, sources = [], [], []
    noise, poke_commands = [], []
    pulse_count = (cfg['steps']+11)//12
    for i in ids.tolist():
        physics_rng, actor_rng, motion_rng = [np.random.default_rng(s)
            for s in np.random.SeedSequence([seed, i]).spawn(3)]
        requested.append(physics(physics_rng, cfg, deployment=False))
        guesses.append(np.exp(actor_rng.uniform(*np.log(cfg['mass_kg']))))
        # Balanced within each group of four. The order is independent of the physics.
        order = np.random.default_rng([seed, 93811, i//4]).permutation([0,0,1,2])
        sources.append(order[i%4])
        noise.append(motion_rng.normal(0, cfg['action_noise_sd'], (cfg['steps'],5)))
        angle = motion_rng.uniform(-np.pi, np.pi, pulse_count)
        strength = motion_rng.uniform(.14,.50,pulse_count)
        poke_commands.append(np.column_stack((strength*np.cos(angle), strength*np.sin(angle),
            motion_rng.uniform(-.09,.02,pulse_count), motion_rng.uniform(-1,1,pulse_count),
            motion_rng.uniform(-1,1,pulse_count))))
    return (np.asarray(requested), np.asarray(guesses), np.asarray(sources, np.uint8),
            np.asarray(noise,np.float32), np.asarray(poke_commands,np.float32))


def scripted_push(step, anchor, tcp):
    target = anchor.clone()
    target[:,0] += -.055 if step < 13 else min(.20, -.055+(step-12)*.014)
    target[:,2] = .09 if step < 6 else .023
    command = tcp.new_empty((len(tcp),5))
    command[:,:3] = (target-tcp)*12
    command[:,3] = -.9
    command[:,4] = .65
    return command


def random_poke(step, state, pulses):
    # Each 12-step bout approaches the cube, descends, then issues a purely
    # open-loop random push. The direction and force are redrawn for each bout.
    phase = step%12
    pulse = pulses[:,step//12]
    if phase >= 7:
        return pulse
    direction = torch.nn.functional.normalize(pulse[:,:2],dim=1)
    target = state[:,26:29].clone()
    target[:,:2] -= .050*direction
    target[:,2] += .065 if phase < 4 else .002
    command = pulse.clone()
    command[:,:3] = (target-state[:,18:21])*14
    command[:,3] = -.85
    return command


@torch.inference_mode()
def rollout(ids, seed, cfg, policy, force_limits, pilot):
    requested, guesses, source, noise, pulses = episode_draws(ids,seed,cfg)
    env = RandomizedPickGPU(requested,guesses,force_limits,cfg['finger_friction'],cfg['scene_spacing_m'])
    try:
        device = env.device
        policy = policy.to(device)
        reset_seeds = (ids+seed).tolist()
        obs, _ = env.reset(seed=reset_seeds)
        env.check_reset_physics()
        if obs.shape != (len(ids),45):
            raise ValueError(f'Unexpected policy observation shape {obs.shape}')
        torch.testing.assert_close(obs[:,42],env.reported_weight[:,0])
        source_t = torch.as_tensor(source,device=device)
        noise = torch.as_tensor(noise,device=device)
        pulses = torch.as_tensor(pulses,device=device)
        delay = torch.as_tensor(requested[:,6],device=device,dtype=torch.long)
        lanes = torch.arange(len(ids),device=device)
        horizon = cfg['steps']
        states = torch.empty((len(ids),horizon+1,39),device=device)
        commands = torch.empty((len(ids),horizon,5),device=device)
        applied = torch.empty_like(commands)
        grip_limits = np.empty((len(ids),horizon,2),np.float32)
        states[:,0] = env.state()
        anchor = states[:,0,26:29]
        for step in range(horizon):
            current = states[:,step]
            movement = policy.get_action(obs) + noise[:,step]
            push = scripted_push(step,anchor,current[:,18:21])
            poke = random_poke(step,current,pulses)
            selected = torch.where(source_t[:,None]==0,movement,
                                   torch.where(source_t[:,None]==1,push,poke))
            commands[:,step] = selected.clamp(-1,1)
            applied[:,step] = torch.where((step>=delay)[:,None],
                commands[lanes,(step-delay).clamp_min(0)],0.)
            obs, _, _, _, _ = env.step(applied[:,step])
            states[:,step+1] = env.state()
            grip_limits[:,step] = env.finger_limits()
        is_validation = (ids==9) if pilot else (ids//4)%10==9
        return dict(episode_id=ids,seed=np.asarray(reset_seeds),task_id=np.zeros(len(ids),dtype=np.uint8),
                    source_id=source,is_validation=is_validation,policy_guess_kg=guesses,
                    constants=env.realized.copy(),requested_constants=requested,
                    state=states.cpu().numpy(),command=commands.cpu().numpy(),applied=applied.cpu().numpy(),
                    finger_drive_limit_n=grip_limits)
    finally:
        env.close()


def collect(root, config, pilot, batch_size=None):
    settings = json.loads(config.read_text())
    cfg = json.loads(Path(settings['physics_config']).read_text())
    cfg['steps'] = settings['steps']
    cfg['scene_spacing_m'] = settings['scene_spacing_m']
    count = 10 if pilot else settings['episodes']
    batch = 10 if pilot else (batch_size or settings['parallel_envs'])
    seed = settings['pilot_seed'] if pilot else settings['seed']
    stage = 'pilot' if pilot else 'data'
    folder = root/stage
    folder.mkdir(parents=True,exist_ok=True)
    lock = (folder/'.collector.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ref = json.loads(Path(cfg['policy_manifest']).read_text())
    checkpoint = Path(ref['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=ref['checkpoint_sha256']:
        raise ValueError('Frozen policy differs from the historical sweep checkpoint')
    policy, metadata = load_agent(checkpoint)
    limits = metadata['settings']['grip_force_limits_n']
    cfg.update(episodes=count,seed=seed,shard_episodes=batch)
    source_files = [Path(__file__), Path(__file__).with_name('verify_gpu.py'),
                    Path(__file__).with_name('collect.py'),
                    Path('robot_adaptation/simulation/gpu_pick.py'),
                    Path('robot_adaptation/simulation/grip_controller.py'),
                    Path('robot_adaptation/simulation/training_lift.py'),
                    Path('robot_adaptation/simulation/context_dynamics.py'),
                    Path('robot_adaptation/simulation/environment.py'),
                    Path('robot_adaptation/policy/frozen_v1.py'), config, Path(settings['physics_config'])]
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    protocol = dict(config=cfg,simulator='physx_cuda',gpu=torch.cuda.get_device_name(),
        deployment=False,stage=stage,environment='PickCube-v1',tasks=['PickCube-v1'],
        sources=SOURCES,constants=CONSTANTS,state=STATE,force_limits_n=limits,
        command=['delta_x','delta_y','delta_z','opening','strength'],
        split='pilot episode 9 reserved' if pilot else 'validation if (episode_id // 4) % 10 == 9',
        policy_checkpoint=str(checkpoint),policy_sha256=ref['checkpoint_sha256'],source_sha256=hashes,
        versions={name:version(name) for name in ['mani_skill','sapien','torch','numpy']})
    if not pilot:
        pilot_info = json.loads((root/'pilot'/'complete.json').read_text())
        pilot_protocol = json.loads((root/'pilot'/'dataset.json').read_text())
        if not pilot_info['verified'] or pilot_info['episodes']!=10 or pilot_protocol['source_sha256']!=hashes:
            raise ValueError('Run and verify the ten-episode pilot with this collector before full collection')
    manifest = folder/'dataset.json'
    if manifest.exists() and json.loads(manifest.read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError('Existing run has a different protocol. Use a new output root.')
    json_write(manifest,protocol)
    (folder/'source').mkdir(exist_ok=True)
    for path in source_files:
        shutil.copyfile(path,folder/'source'/path.name)
    clock = time.monotonic()
    done_now = 0
    for index, start in enumerate(range(0,count,batch)):
        ids = np.arange(start,min(start+batch,count),dtype=np.int64)
        path = folder/f'shard_{index:04}.npz'
        if not path.exists():
            records = rollout(ids,seed,cfg,policy,limits,pilot)
            temp = path.with_suffix('.tmp')
            with temp.open('wb') as f:
                np.savez(f,**records)
            # Reopen before publishing so an interrupted run can safely resume.
            inspect_shard(temp,protocol,ids)
            temp.replace(path)
            del records
            gc.collect()
            done_now += len(ids)
        else:
            inspect_shard(path,protocol,ids)
        progress = dict(episodes=int(ids[-1]+1),transitions=int((ids[-1]+1)*cfg['steps']),
                        total=count*cfg['steps'],batch_size=batch,gpu=protocol['gpu'],
                        process_steps_per_second=round(done_now*cfg['steps']/max(1,time.monotonic()-clock),1))
        json_write(folder/'progress.json',progress)
        if index%10==0 or ids[-1]+1==count:
            print(progress,flush=True)
    verify_dataset(folder)
    lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--config',type=Path,default=Path('configs/context_pick_gpu.json'))
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--batch-size',type=int)
    a=p.parse_args()
    collect(a.root,a.config,a.pilot,a.batch_size)
