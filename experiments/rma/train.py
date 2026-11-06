"""Run the repository's pinned PPO with an RMA encoder/policy and existing task."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from .common import ROOT, config, prepare, status, env_kwargs, digest


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Pinned PPO integration point changed: {old}')
    return text.replace(old, new)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default='configs/rma_baseline.json')
    p.add_argument('--output', required=True)
    p.add_argument('--iterations', type=int)
    p.add_argument('--checkpoint', help='Continue RMA weights only; optimizer restarts')
    p.add_argument('--seed', type=int)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    c = config(a.config)
    if a.seed is not None:
        c['seed'] = a.seed
    if a.iterations is not None:
        c['iterations'] = a.iterations
    if c['iterations'] < 1:
        p.error('iterations must be positive')
    if a.cpu:
        c['num_envs'] = c['num_eval_envs'] = 1
    if a.checkpoint:
        previous = config(Path(a.checkpoint).parent/'manifest.json')
        for key in ('masses_kg', 'frictions', 'goal_clearance_m', 'finger_friction'):
            if previous['config'][key] != c[key]:
                raise ValueError(f'Continuation physics mismatch: {key}')
    out = prepare(a.output, c, 'phase1', initialization_sha256=digest(a.checkpoint) if a.checkpoint else None,
                  initialization='RMA weights; optimizer restarts' if a.checkpoint else 'fresh random weights')
    source = (ROOT/'robot_adaptation/policy/examples/maniskill_pick_cube/ppo_upstream.py').read_text()
    source = replace_once(source, 'class Agent(nn.Module):', 'class ReferenceAgent(nn.Module):')
    source = replace_once(source, 'if __name__ == "__main__":',
        'from robot_adaptation.policy.rma.model import Agent\nfrom robot_adaptation.simulation import training_lift\nif __name__ == "__main__":')
    source = replace_once(source, 'env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")',
                          'env_kwargs = '+repr(env_kwargs(c, 'physx_cpu' if a.cpu else 'physx_cuda')))
    source = replace_once(source, 'eval_obs, _ = eval_envs.reset(seed=args.seed)',
                          f'eval_obs, _ = eval_envs.reset(seed={c["eval_seed"]})')
    source = replace_once(source, 'torch.load(args.checkpoint)', 'torch.load(args.checkpoint, map_location=device, weights_only=True)')
    # Keep raw Gaussian PPO actions in the buffer; clip only physical execution.
    source = replace_once(source, 'eval_envs.step(agent.get_action(eval_obs, deterministic=True))',
                          'eval_envs.step(clip_action(agent.get_action(eval_obs, deterministic=True)))')
    runtime = out/'ppo_runtime.py'
    runtime.write_text(source)
    # The pinned runner prefixes run names with runs/.
    relative = os.path.relpath(out, ROOT/'runs')
    cmd = [sys.executable, '-u', str(runtime), '--exp-name', relative, '--env-id', 'GripWeightedPickCube-v2',
           '--seed', str(c['seed']), '--num-envs', str(c['num_envs']), '--num-eval-envs', str(c['num_eval_envs']),
           '--total-timesteps', str(c['iterations']*c['num_envs']*c['horizon']),
           '--num-steps', str(c['horizon']), '--num-eval-steps', str(c['horizon']),
           '--learning-rate', str(c['learning_rate']), '--num-minibatches', '5', '--update-epochs', '4',
           '--control-mode', 'pd_ee_delta_pos', '--no-partial-reset', '--no-eval-partial-reset',
           '--no-capture-video', '--finite-horizon-gae', '--eval-reconfiguration-freq', '0', '--target-kl', '0.05']
    if a.checkpoint:
        import shutil
        shutil.copy2(a.checkpoint, out/'initialization.pt')
        cmd += ['--checkpoint', str(out/'initialization.pt')]
    (out/'command.json').write_text(json.dumps(cmd, indent=2))
    if a.prepare_only:
        print(out)
        return
    started = time.time()
    status(out, 'running')
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4')
    try:
        with (out/'console.log').open('w') as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        status(out, 'completed' if result.returncode == 0 else 'failed', elapsed_seconds=time.time()-started,
               returncode=result.returncode)
        if result.returncode:
            raise RuntimeError(f'PPO failed; inspect {out}/console.log')
    except BaseException as exc:
        status(out, 'failed', error=str(exc), elapsed_seconds=time.time()-started)
        raise
    print(out)


if __name__ == '__main__':
    main()
