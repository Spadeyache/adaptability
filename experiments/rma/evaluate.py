"""Matched oracle/adapted sweeps, complete step traces, optional camera GIFs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from robot_adaptation.policy.rma.model import Adapter, History, split_observation
from .common import config, prepare, status, make_env, load_policy, digest


def condition(mass, friction):
    m = not .02 <= mass <= .12
    f = not .3 <= friction <= 1.
    return 'combined_ood' if m and f else 'mass_ood' if m else 'friction_ood' if f else 'in_range'


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--adapter')
    p.add_argument('--output', required=True)
    p.add_argument('--seeds', type=int, default=10, help='Paired starts per grid cell')
    p.add_argument('--smoke', action='store_true', help='One in-range cell; integration evidence only')
    p.add_argument('--validation', action='store_true', help='In-range development evaluation; leave final OOD tests untouched')
    p.add_argument('--record', action='store_true', help='Camera GIF for first seed of each cell and mode')
    a = p.parse_args()
    if a.seeds < 1:
        p.error('seeds must be positive')
    c = config(Path(a.checkpoint).parent/'manifest.json')['config']
    if a.adapter:
        provenance = config(Path(a.adapter).parent/'manifest.json')
        if provenance['policy_sha256'] != digest(a.checkpoint):
            raise ValueError('Adapter was trained against a different policy/encoder')
    out = prepare(a.output, c, 'evaluation', policy_sha256=digest(a.checkpoint),
                  adapter_sha256=digest(a.adapter) if a.adapter else None,
                  smoke=a.smoke, validation=a.validation, seeds_per_cell=a.seeds)
    policy = load_policy(a.checkpoint, 'cpu')
    adapter = Adapter().eval()
    if a.adapter:
        adapter.load_state_dict(torch.load(a.adapter, map_location='cpu', weights_only=True))
    masses = [.07] if a.smoke else c['test_mass_kg']
    frictions = [.65] if a.smoke else c['test_friction']
    if a.validation:
        masses = [m for m in masses if .02 <= m <= .12]
        frictions = [f for f in frictions if .3 <= f <= 1.]
    seed_start = c['eval_seed']+1000 if a.validation or a.smoke else c['test_seed']
    modes = ['oracle', 'adapted'] if a.adapter else ['oracle']
    env = None
    rows = []
    status(out, 'running')
    try:
        env = make_env(c)
        with (out/'episodes.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['episode','condition','mass_kg','friction','seed','mode','success','sustained_success','return','latent_rmse','trace','video'])
            writer.writeheader()
            for mass in masses:
                for friction in frictions:
                    for k in range(a.seeds):
                        initial = None
                        for mode in modes:
                            obs, _ = env.reset(seed=seed_start+k, options=dict(mass_kg=mass, friction_coefficient=friction))
                            if initial is not None and not torch.equal(initial, obs):
                                raise RuntimeError('Paired resets do not match')
                            initial = obs.clone()
                            history = History(1, c['history_length'])
                            previous = torch.zeros(1, 5)
                            trace = {key: [] for key in ['state','action','z','z_hat','success','reward','clearance_m','goal_distance_m']}
                            frames = []
                            with torch.no_grad():
                                for step in range(c['horizon']):
                                    state, physical = split_observation(obs)
                                    target = policy.encode(physical)
                                    hist = history.append(state, previous)
                                    latent = adapter(hist) if mode == 'adapted' else target
                                    action = policy.action_from_latent(state, latent).clamp(-1, 1)
                                    obs, reward, _, _, info = env.step(action)
                                    values = dict(state=state, action=action, z=target, z_hat=latent,
                                                  success=info['success'], reward=reward,
                                                  clearance_m=info['clearance_m'], goal_distance_m=info['goal_distance_m'])
                                    for key, value in values.items():
                                        trace[key].append(value[0].cpu().numpy())
                                    previous = action
                                    if a.record and k == 0 and step % 2 == 0:
                                        frames.append(env.render()[0].cpu().numpy())
                            episode = len(rows)
                            path = f'episode_{episode:05}.npz'
                            np.savez_compressed(out/path, **{key: np.asarray(v) for key,v in trace.items()})
                            video = ''
                            if frames:
                                from PIL import Image
                                video = f'episode_{episode:05}.gif'
                                images = [Image.fromarray(x.astype('uint8')) for x in frames]
                                images[0].save(out/video, save_all=True, append_images=images[1:], duration=100, loop=0)
                            row = dict(episode=episode, condition=condition(mass, friction), mass_kg=mass,
                                       friction=friction, seed=seed_start+k, mode=mode,
                                       success=int(trace['success'][-1]), sustained_success=int(np.all(trace['success'][-10:])),
                                       **{'return':float(np.sum(trace['reward']))},
                                       latent_rmse=float(np.sqrt(np.mean((np.asarray(trace['z'])-np.asarray(trace['z_hat']))**2))),
                                       trace=path, video=video)
                            rows.append(row); writer.writerow(row); f.flush()
        status(out, 'completed', episodes=len(rows))
        from .plot import report
        report(out)
    except BaseException as exc:
        status(out, 'failed', error=str(exc), episodes=len(rows))
        raise
    finally:
        if env is not None:
            env.close()
    print(out)


if __name__ == '__main__':
    main()
