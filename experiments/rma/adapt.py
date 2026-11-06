"""Train history adapter on its own rollouts; freeze privileged encoder and policy."""
import argparse
import json
import time
from pathlib import Path
import torch
from robot_adaptation.policy.rma.model import Adapter, History, split_observation
from .common import config, prepare, status, make_env, load_policy, digest


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--resume', help='Continue adapter weights with a fresh optimizer')
    p.add_argument('--seed-offset', type=int, default=0)
    a = p.parse_args()
    if a.episodes < 1:
        p.error('episodes must be positive')
    c = config(Path(a.checkpoint).parent/'manifest.json')['config']
    out = prepare(a.output, c, 'phase2', policy_checkpoint=str(Path(a.checkpoint).resolve()),
                  policy_sha256=digest(a.checkpoint), episodes=a.episodes,
                  seed_offset=a.seed_offset, initialization_sha256=digest(a.resume) if a.resume else None)
    torch.manual_seed(c['seed']+1)
    device = 'cpu' if a.cpu else 'cuda'
    policy = load_policy(a.checkpoint, device)
    adapter = Adapter().to(device)
    if a.resume:
        previous_manifest = config(Path(a.resume).parent/'manifest.json')
        if previous_manifest['policy_sha256'] != digest(a.checkpoint):
            raise ValueError('Cannot resume an adapter for a different policy')
        adapter.load_state_dict(torch.load(a.resume, map_location=device, weights_only=True))
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)
    n = 1 if a.cpu else c['num_envs']
    env = None
    started = time.time()
    status(out, 'running')
    try:
        env = make_env(c, 'physx_cpu' if a.cpu else 'physx_cuda', n)
        with (out/'learning.jsonl').open('w') as log:
            for episode in range(a.episodes):
                obs, _ = env.reset(seed=c['seed']+100000+a.seed_offset+episode)
                history = History(n, c['history_length'], device)
                previous = torch.zeros(n, 5, device=device)
                losses = []
                for step in range(c['horizon']):
                    state, physical = split_observation(obs)
                    prediction = adapter(history.append(state, previous))
                    with torch.no_grad():
                        target = policy.encode(physical)
                        action = policy.action_from_latent(state, prediction.detach()).clamp(-1, 1)
                    loss = (prediction-target).square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite adapter loss')
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
                    optimizer.step()
                    obs, _, _, _, info = env.step(action)
                    previous = action
                    losses.append(loss.item())
                row = dict(episode=episode, transitions=(episode+1)*n*c['horizon'],
                           loss=sum(losses)/len(losses), success=float(info['success'].float().mean()),
                           elapsed_seconds=time.time()-started)
                log.write(json.dumps(row)+'\n'); log.flush()
                if (episode+1) % 25 == 0:
                    torch.save(adapter.state_dict(), out/f'adapter_{episode+1}.pt')
                print(json.dumps(row), flush=True)
        torch.save(adapter.state_dict(), out/'final_adapter.pt')
        status(out, 'completed', elapsed_seconds=time.time()-started)
    except BaseException as exc:
        status(out, 'failed', error=str(exc))
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
