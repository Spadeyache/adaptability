import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import torch
from robot_adaptation.policy.rma.model import Agent

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config(path):
    return json.loads(Path(path).read_text())


def prepare(output, settings, stage, **extra):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = list((ROOT/'robot_adaptation/policy/rma').glob('*.py')) + list((ROOT/'experiments/rma').glob('*.py'))
    sources += list((ROOT/'robot_adaptation/simulation').glob('*.py'))
    sources += [ROOT/'robot_adaptation/policy/examples/maniskill_pick_cube/ppo_upstream.py']
    hashes = {}
    for src in sources:
        relative = src.relative_to(ROOT)
        target = output/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        hashes[str(relative)] = digest(src)
    manifest = dict(stage=stage, status='prepared', config=settings, source_sha256=hashes,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    method='RMA2-inspired state-based cube baseline; independent implementation',
                    reference='https://github.com/yichao-liang/rma4rma', **extra)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return output


def status(output, value, **fields):
    path = Path(output)/'manifest.json'
    data = json.loads(path.read_text())
    data.update(status=value, **fields)
    path.write_text(json.dumps(data, indent=2))


def env_kwargs(c, backend):
    return dict(obs_mode='state', sim_backend=backend, render_backend='cpu', render_mode='rgb_array',
                control_mode='pd_ee_delta_pos', condition_on_friction=True,
                friction_profile='independent_grid', masses_kg=c['masses_kg'],
                cube_friction_coefficients=c['frictions'], finger_friction_coefficient=c['finger_friction'],
                goal_clearance_m=c['goal_clearance_m'], grip_force_cost=.1)


def make_env(c, backend='physx_cpu', num_envs=1):
    import gymnasium as gym
    from robot_adaptation.simulation.training_lift import ENV_ID
    return gym.make(ENV_ID, num_envs=num_envs, **env_kwargs(c, backend))


def load_policy(path, device):
    agent = Agent().to(device)
    agent.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    agent.eval().requires_grad_(False)
    return agent
