"""Bounded GPU training blocks with in-range development gates and saved visuals.

Gates are operational targets for this task, not the paper's published metrics.
The runner stops for review if its block budget expires; it never claims success
because the budget expired. OOD outcomes never select models or trigger training.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from .common import ROOT, config, prepare, status
from .progress import render


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='configs/rma_baseline.json')
    p.add_argument('--base-blocks', type=int, default=3)
    p.add_argument('--adapter-blocks', type=int, default=3)
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--adapter-episodes', type=int, default=1000)
    p.add_argument('--validation-seeds', type=int, default=12)
    p.add_argument('--final-seeds', type=int, default=34)
    a = p.parse_args()
    if min(a.base_blocks,a.adapter_blocks,a.iterations,a.adapter_episodes,a.validation_seeds,a.final_seeds) < 1:
        p.error('All budgets must be positive')
    c = config(a.config)
    out = prepare(a.output, c, 'pipeline', budgets=vars(a), oracle_target=.9,
                  adapted_target=.8, maximum_oracle_gap=.1)
    pinned_config = out/'config.json'
    pinned_config.write_text(json.dumps(c, indent=2))
    import os
    status(out, 'running', pid=os.getpid())

    def run(step, module, args):
        status(out, 'running', current_step=step)
        cmd = [sys.executable, '-u', '-m', module, *map(str,args)]
        with (out/(step+'.log')).open('w') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        result = proc.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        render(out)
                if result:
                    raise RuntimeError(f'{step} failed with exit {result}; see {step}.log')
            except BaseException:
                proc.terminate()
                proc.wait()
                raise
        render(out)

    def success(path, mode):
        rows = json.loads((path/'summary.json').read_text())
        return next(r['success_rate'] for r in rows if r['mode']==mode and r['condition']=='in_range')

    try:
        checkpoint = None
        base_ready = False
        for block in range(a.base_blocks):
            base = out/f'base_{block+1:02}'
            args = ['--config',pinned_config,'--output',base,'--iterations',a.iterations,'--seed',c['seed']+1000*block]
            if checkpoint:
                args += ['--checkpoint',checkpoint]
            run(base.name, 'experiments.rma.train', args)
            checkpoint = base/'final_ckpt.pt'
            val = out/f'base_validation_{block+1:02}'
            run(val.name, 'experiments.rma.evaluate', ['--checkpoint',checkpoint,'--output',val,'--validation','--seeds',a.validation_seeds])
            rate = success(val, 'oracle')
            status(out, 'running', oracle_validation_success=rate, policy_checkpoint=str(checkpoint))
            if rate >= .9:
                base_ready = True
                break
        if not base_ready:
            status(out, 'needs_review', reason='Privileged policy below 90% development success after bounded training blocks.')
            render(out)
            return
        adapter = None
        adapter_ready = False
        for block in range(a.adapter_blocks):
            dest = out/f'adapter_{block+1:02}'
            args = ['--checkpoint',checkpoint,'--output',dest,'--episodes',a.adapter_episodes,'--seed-offset',block*a.adapter_episodes]
            if adapter:
                args += ['--resume',adapter]
            run(dest.name, 'experiments.rma.adapt', args)
            adapter = dest/'final_adapter.pt'
            val = out/f'adapter_validation_{block+1:02}'
            run(val.name, 'experiments.rma.evaluate', ['--checkpoint',checkpoint,'--adapter',adapter,'--output',val,'--validation','--seeds',a.validation_seeds])
            adapted, oracle = success(val,'adapted'), success(val,'oracle')
            status(out, 'running', adapter_validation_success=adapted, adapter_checkpoint=str(adapter))
            if adapted >= .8 and oracle-adapted <= .1:
                adapter_ready = True
                break
        if not adapter_ready:
            status(out, 'needs_review', reason='Adapter below development target after bounded training blocks.')
            render(out)
            return
        run('final_evaluation', 'experiments.rma.evaluate', ['--checkpoint',checkpoint,'--adapter',adapter,
            '--output',out/'final_evaluation','--seeds',a.final_seeds,'--record'])
        status(out, 'completed', reason='Development gates met and held-out evaluation recorded; inspect results for scientific conclusions.')
        render(out)
    except BaseException as exc:
        status(out, 'failed', error=str(exc))
        render(out)
        raise


if __name__ == '__main__':
    main()
