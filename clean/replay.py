"""Replay every saved candidate in a separate copy, preserving the original ledger.

Uses the original immutable cube identities/properties. By default it reuses
saved trajectories; --recollect regenerates motion in the copy with those same
identities and script settings. All probes and models are retrained either way.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
DATA_NAMES=['train','test','out_of_range','plans','episode_settings.json','pilot_manifest.json','full_manifest.json','normalizer.json','generator.json']
SOURCES=['collect.py','native.py','schema.py','script.py']


def copy_item(source,target):
    if source.is_dir():shutil.copytree(source,target)
    elif source.exists():shutil.copy2(source,target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--recollect',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    runs=[json.loads(p.read_text()) for p in sorted((ROOT/'runs').glob('iteration_*/result.json'))]
    final=json.loads((ROOT/'results/final/result.json').read_text())
    assert [r['iteration'] for r in runs]==list(range(1,len(runs)+1))
    plan=dict(candidates=len(runs),original_selected=final['iteration'],retrain_probes=True,
        reuse_saved_trajectories=not args.recollect,immutable_split_reused=True,
        regimes=[r['config'].get('grip_force_max',30.) for r in runs],
        original_final_ledger_untouched=True)
    if args.plan_only:
        print(json.dumps(plan,indent=2));return
    if args.destination is None:parser.error('--destination is required unless --plan-only is used')
    destination=args.destination.expanduser().resolve()
    if destination.exists():raise FileExistsError('Use a new destination; no existing copy is overwritten')
    if destination==ROOT or ROOT in destination.parents:
        raise ValueError('Reproduction destination must be outside the source clean directory')
    destination.mkdir(parents=True)
    root=destination/'clean';shutil.copytree(ROOT,root)
    (root/'runs').rename(root/'source_runs');(root/'results').rename(root/'source_results')
    (root/'runs').mkdir();(root/'results').mkdir()
    active=root/'source_active_regime';active.mkdir()
    for name in DATA_NAMES:
        path=root/'data'/name
        if path.exists():shutil.move(str(path),str(active/name))
    for name in SOURCES:shutil.copy2(root/name,active/name)
    env=os.environ.copy();env.update(PYTHONPATH='clean/vendor:.',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/clean-mpl')
    def command(*arguments):
        subprocess.run([sys.executable,'-m',*arguments],cwd=destination,env=env,check=True)
    regime=None;selected=final['iteration']
    for saved in runs:
        cap=saved['config'].get('grip_force_max',30.)
        if cap!=regime:
            if regime is not None:
                archive=root/f'replayed_regime_{regime:g}';archive.mkdir()
                for name in DATA_NAMES:
                    path=root/'data'/name
                    if path.exists():shutil.move(str(path),str(archive/name))
            source=root/'data/versions/grip_max_30' if cap==30. else active
            for name in DATA_NAMES:copy_item(source/name,root/'data'/name)
            for name in SOURCES:shutil.copy2(source/name,root/name)
            (root/'data/generator.json').write_text(json.dumps(dict(version=f'replay_grip_max_{cap:g}',grip_force_max=cap),indent=2)+'\n')
            if args.recollect:
                archive=root/f'saved_trajectories_{cap:g}';archive.mkdir()
                for name in DATA_NAMES:
                    if name=='generator.json':continue
                    path=root/'data'/name
                    if path.exists():shutil.move(str(path),str(archive/name))
                command('clean.collect','--stage','pilot')
            command('clean.audit','--stage','pilot')
            if args.recollect:
                if cap==30.:command('clean.collect','--stage','full')
                else:command('clean.parallel_collect','--workers','4')
            command('clean.audit','--stage','full')
            regime=cap
        number=saved['iteration'];cfg=root/'configs'/f'full_replay_{number:02d}.json'
        cfg.write_text(json.dumps(saved['config'],indent=2)+'\n')
        arguments=['clean.run','candidate','--iteration',str(number),'--config',str(cfg),
            '--hypothesis',saved['hypothesis'],'--change',saved['change'],'--device',args.device]
        if saved.get('base_iteration') is not None:arguments+=['--base-iteration',str(saved['base_iteration'])]
        command(*arguments)
        observed=json.loads((root/'runs'/f'iteration_{number:02d}'/'result.json').read_text())
        if all(observed['success'][f'S{i}'] for i in range(1,5)):
            selected=number;break
    command('clean.run','final','--iteration',str(selected),'--device',args.device)
    command('clean.report')
    command('clean.verify_results')
    print(str(root/'RESULTS.md'))


if __name__=='__main__':main()
