"""Scripted collection with a train-only ten-cube gate and immutable identities."""
import argparse
import importlib.metadata
import json
import os
import time
import numpy as np
from clean.protocol import ROOT, digest, initialize, verify, write_new
from clean.native import make_env, reset_env, set_cube_properties, execute
from clean.script import plan_episode, sample_settings
from clean.schema import observe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['pilot', 'full'], required=True)
    args = parser.parse_args()
    initialize()
    data = ROOT / 'data'
    if args.stage == 'full':
        path = ROOT / 'results/pilot_audit.json'
        if not path.exists() or json.loads(path.read_text()).get('status') != 'pass':
            raise RuntimeError('Full collection requires passing saved ten-training-cube audit')
        report=json.loads(path.read_text())
        for name in ['schema.py','native.py','script.py']:
            if report['static_provenance']['source_sha256'][name]!=digest(ROOT/name):
                raise RuntimeError('Pilot gate belongs to a different generator')
        for name,expected in report['episode_sha256'].items():
            if digest(data/name)!=expected:raise RuntimeError('Pilot gate data changed')
    settings = [sample_settings(e) for e in range(10)]
    if not (data / 'episode_settings.json').exists():
        write_new(data / 'episode_settings.json', settings)
    elif json.loads((data / 'episode_settings.json').read_text()) != settings:
        raise RuntimeError('Episode settings changed; explicitly version data before recollecting')
    cubes = json.loads((data / 'cubes.json').read_text())['cubes']
    selected = cubes[:10] if args.stage == 'pilot' else cubes
    plans_dir = data / 'plans'
    plans_dir.mkdir(exist_ok=True)
    started = time.monotonic()
    env = None
    count = 0
    try:
        for cube in selected:
            for setting in settings:
                episode = setting['episode_id']
                folder = data / cube['split']
                folder.mkdir(exist_ok=True)
                dest = folder / f"cube_{cube['cube_id']:04d}_episode_{episode:02d}.npz"
                receipt = dest.with_suffix('.json')
                if dest.exists():
                    if not receipt.exists() or json.loads(receipt.read_text())['sha256'] != digest(dest):
                        raise RuntimeError(f'Existing episode failed integrity: {dest}')
                    continue
                if env is None:
                    env = make_env(setting['scene_seed'])
                else:
                    reset_env(env, setting['scene_seed'])
                set_cube_properties(env, cube)
                pose = env.cube.pose.raw_pose[0].cpu().numpy().copy()
                plan_path = plans_dir / f'episode_{episode:02d}.npz'
                if not plan_path.exists():
                    actions, phases = plan_episode(pose, setting)
                    with plan_path.open('xb') as f:
                        np.savez_compressed(f, pose=pose, action=actions, phase=np.asarray(phases))
                with np.load(plan_path) as plan:
                    if not np.array_equal(pose, plan['pose']):
                        raise RuntimeError('Matched initial pose differs')
                    actions = plan['action'].copy()
                states = [observe(env, first=True)]
                for action in actions:
                    execute(env, action)
                    states.append(observe(env))
                states = np.asarray(states)
                n = len(actions)
                temporary = dest.with_suffix('.partial')
                with temporary.open('wb') as f:
                    np.savez_compressed(f, cube_id=np.full(n, cube['cube_id'], np.int32),
                        episode_id=np.full(n, episode, np.int16), step=np.arange(n, dtype=np.int32),
                        state=states[:-1], action=actions, next_state=states[1:])
                os.replace(temporary, dest)
                event = {'cube_id': cube['cube_id'], 'episode_id': episode, 'steps': n,
                         'path': str(dest.relative_to(ROOT)), 'sha256': digest(dest),
                         'source_sha256': {name: digest(ROOT / name) for name in ['collect.py', 'native.py', 'script.py', 'schema.py']}}
                write_new(receipt, event)
                count += 1
                print(json.dumps({'stage': args.stage, 'cube_id': cube['cube_id'],
                      'episode_id': episode, 'completed_this_run': count,
                      'elapsed_seconds': round(time.monotonic() - started, 3)}), flush=True)
    except BaseException as exc:
        with (ROOT / 'results/collection_events.jsonl').open('a') as f:
            f.write(json.dumps({'stage': args.stage, 'event': 'failure', 'error': repr(exc), 'completed_this_run': count}) + '\n')
        raise
    finally:
        if env is not None:
            env.close()
    verify()
    manifest = {'stage': args.stage, 'cube_count': len(selected), 'episode_count': len(selected)*10,
                'completed_this_run': count, 'elapsed_seconds': time.monotonic()-started,
                'versions': {k: importlib.metadata.version(k) for k in ['mani_skill', 'sapien', 'mplib', 'torch', 'numpy']},
                'split_sha256': digest(data / 'split.json')}
    with (ROOT / 'results/collection_events.jsonl').open('a') as f:
        f.write(json.dumps(manifest) + '\n')
    (data / f'{args.stage}_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
