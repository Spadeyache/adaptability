"""Supervise the requested long policy run and its untouched final evaluation.

Launch only after reviewing a completed pilot and its diagnostic evaluation.
Training and final evaluation statuses are separate; progress is written atomically.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import math
import shutil
import signal
import subprocess
import sys
import traceback

from robot_adaptation.paths import ROOT


def write_status(path, state):
    temporary=path.with_suffix('.tmp')
    state['updated_at']=datetime.now(timezone.utc).isoformat()
    temporary.write_text(json.dumps(state,indent=2)+'\n')
    temporary.replace(path)


def run_command(command, *, cwd, environment, log):
    """Keep long subprocess output on disk; terminate children on interruption."""
    print('Starting:', command, flush=True)
    with Path(log).open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            raise
    if code:
        raise RuntimeError(f'Process exited {code}; see {log}')


def required_json(path, keys):
    """An exited process is not evidence that its required artifacts exist."""
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict) or set(keys)-set(value):
        raise RuntimeError(f'Missing required fields in {path}')
    return value


def validate_guarded_pilot(pilot, manifest):
    """Verify that the correction actually ran, rather than only being enabled."""
    settings=manifest['settings']
    if settings.get('guard_policy_updates') is not True or not manifest.get('trust_region',{}).get('enabled'):
        raise ValueError('A guarded continuation requires a guarded pilot')
    rows=[json.loads(line) for line in (pilot.parent/'trust_region.jsonl').read_text().splitlines() if line.strip()]
    planned=manifest['planned_iterations']
    if planned<100:
        raise ValueError('The corrective learning pilot must cover at least 100 cycles')
    source_hash=hashlib.sha256((pilot.parent/'trust_region.py').read_bytes()).hexdigest()
    if (source_hash!=manifest.get('trust_region_sha256') or
            source_hash!=hashlib.sha256((ROOT/'robot_adaptation/policy/trust_region.py').read_bytes()).hexdigest()):
        raise ValueError('The guarded implementation changed since the reviewed pilot')
    if [row['iteration'] for row in rows]!=list(range(1,planned+1)):
        raise ValueError('Pilot guard logs do not cover every learning cycle')
    batch=settings['num_envs']*settings['num_steps']
    limit=settings['max_exact_kl']
    for row in rows:
        if (row['global_step']!=row['iteration']*batch or row['rollout_observations']!=batch
                or row['effective_actions']!=5 or row['optimization_semantics']!=manifest['optimization_semantics']):
            raise ValueError('Pilot guard evidence does not describe the full five-action rollout')
        for field in ('max_accepted_exact_kl','rollout_exact_kl'):
            value=row[field]
            if not math.isfinite(value) or value < 0 or value > limit+1e-8:
                raise ValueError('Pilot accepted an update beyond the exact KL bound')
    accepted=sum(row['accepted_updates'] for row in rows)
    if accepted<1:
        raise ValueError('Guarded pilot accepted no policy learning updates')
    return dict(iterations=len(rows),accepted_updates=accepted,
                max_accepted_exact_kl=max(row['max_accepted_exact_kl'] for row in rows),limit=limit)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-checkpoint',type=Path,required=True)
    parser.add_argument('--pilot-evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=50000)
    parser.add_argument('--eval-seed-count',type=int,default=10)
    parser.add_argument('--estimator-checkpoint',type=Path,required=True)
    parser.add_argument('--estimator-evaluation',type=Path,required=True)
    parser.add_argument('--estimator-robustness',type=Path)
    parser.add_argument('--retry-sequences',type=int,default=10)
    parser.add_argument('--require-guarded-pilot',action='store_true')
    parser.add_argument('--pilot-gate',type=Path,help='Reviewed matched-task gate for a corrective pilot')
    args=parser.parse_args(argv)
    if args.iterations<=0 or args.eval_seed_count<2 or args.retry_sequences<2:
        parser.error('Use positive training iterations and at least two final evaluation seeds')
    # The loader path must be configured by the launching shell on WSL.
    from robot_adaptation.policy.train import main as train
    pilot=args.pilot_checkpoint.resolve()
    manifest=json.loads((pilot.parent/'manifest.json').read_text())
    diagnostic=json.loads(args.pilot_evaluation.read_text())
    estimator=args.estimator_checkpoint.resolve()
    estimator_metrics=json.loads(args.estimator_evaluation.read_text())
    if estimator_metrics.get('checkpoint_sha256')!=hashlib.sha256(estimator.read_bytes()).hexdigest():
        parser.error('Estimator checkpoint must match its held-condition evaluation')
    if manifest.get('status')!='complete' or manifest.get('observations')!=46:
        parser.error('Pilot checkpoint must belong to a completed46-input training run')
    if diagnostic.get('checkpoint_sha256')!=hashlib.sha256(pilot.read_bytes()).hexdigest():
        parser.error('Pilot evaluation must match the chosen pilot checkpoint')
    if not diagnostic.get('reset_fingerprints') or any(seed>=91000 for seed in diagnostic['seeds']):
        parser.error('Use a fresh-scene diagnostic evaluation and leave910xx final seeds untouched')
    guarded_pilot=None
    if args.require_guarded_pilot:
        if args.pilot_gate is None:
            parser.error('A corrective continuation requires --pilot-gate')
        try:
            guarded_pilot=validate_guarded_pilot(pilot,manifest)
            gate=required_json(args.pilot_gate,['passed','checkpoint_sha256','evaluations'])
            if gate['passed'] is not True or gate['checkpoint_sha256']!=hashlib.sha256(pilot.read_bytes()).hexdigest():
                raise ValueError('The matched-task pilot gate did not pass for this checkpoint')
            if {'gpu_training_grid','gpu_intermediate_grid','cpu_retry'}-set(gate['evaluations']):
                raise ValueError('The corrective pilot gate omits a required task comparison')
            for evidence in gate['evaluations'].values():
                if hashlib.sha256(Path(evidence['path']).read_bytes()).hexdigest()!=evidence['sha256']:
                    raise ValueError('Reviewed pilot evaluation evidence changed')
        except (OSError,ValueError,KeyError) as error:
            parser.error(str(error))
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    status_path=output/'status.json'
    settings={**manifest['settings'],'checkpoint':str(pilot),'total_timesteps':args.iterations*manifest['settings']['num_envs']*manifest['settings']['num_steps'],
              'eval_freq':500,'run_root':str(output/'training_runs')}
    state=dict(status='preparing',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
               planned_iterations=args.iterations,settings=settings,pilot_checkpoint=str(pilot),
               pilot_evaluation=str(args.pilot_evaluation.resolve()),training_run=None,final_evaluation=None,
               retry_evaluation=None,assessment=None)
    if guarded_pilot is not None:
        state['guarded_pilot_validation']=guarded_pilot
        state['pilot_gate']=str(args.pilot_gate.resolve())
    write_status(status_path,state)
    def interrupted(number,frame):
        raise KeyboardInterrupt(f'Received signal {number}')
    handlers={number:signal.getsignal(number) for number in (signal.SIGTERM,signal.SIGHUP)}
    for number in handlers:
        signal.signal(number,interrupted)
    try:
        snapshot=output/'source'
        snapshot.mkdir()
        for folder in ('robot_adaptation','experiments'):
            shutil.copytree(ROOT/folder,snapshot/folder,
                            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (snapshot/'configs').mkdir()
        shutil.copy2(ROOT/'configs'/'adaptation_assessment.json',snapshot/'configs')
        hashes={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in snapshot.rglob('*') if p.is_file()}
        (output/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
        pinned_estimator=output/'estimator.pt'
        shutil.copy2(estimator,pinned_estimator)
        shutil.copy2(args.estimator_evaluation,output/'estimator_evaluation.json')
        if args.estimator_robustness:
            shutil.copy2(args.estimator_robustness,output/'estimator_robustness.json')
        shutil.copy2(args.pilot_evaluation,output/'pilot_evaluation.json')
        if args.pilot_gate:
            shutil.copy2(args.pilot_gate,output/'pilot_gate.json')
        conditions=[dict(mass_kg=row['target'][0],friction=row['target'][1])
                    for row in estimator_metrics['conditions']]
        (output/'retry_conditions.json').write_text(json.dumps(conditions,indent=2)+'\n')
        run=train(settings,prepare_only=True)
        if args.require_guarded_pilot:
            for filename in ('ppo_runtime.py','ppo.py','lift_task.py','hold_task.py','grip_controller.py','trust_region.py'):
                if hashlib.sha256((run/filename).read_bytes()).hexdigest()!=hashlib.sha256((pilot.parent/filename).read_bytes()).hexdigest():
                    raise RuntimeError('Training implementation differs from the reviewed pilot: '+filename)
        state.update(status='training',training_run=str(run),estimator_checkpoint=str(pinned_estimator))
        write_status(status_path,state)
        training_manifest_path=run/'manifest.json'
        training_manifest=json.loads(training_manifest_path.read_text())
        launch=json.loads((run/'launch_command.json').read_text())
        environment=dict(os.environ,**launch['environment_overrides'])
        training_manifest['status']='running'
        write_status(training_manifest_path,training_manifest)
        try:
            run_command(launch['command'],cwd=launch['cwd'],environment=environment,log=run/'console.log')
            if not (run/'final_ckpt.pt').is_file():
                raise RuntimeError('Training exited without a final checkpoint')
        except BaseException:
            training_manifest.update(status='failed',exit_code=1)
            write_status(training_manifest_path,training_manifest)
            raise
        training_manifest.update(status='complete',exit_code=0)
        write_status(training_manifest_path,training_manifest)
        state.update(status='evaluating_policy')
        write_status(status_path,state)
        # Separate interpreters rooted at the snapshot keep later workspace
        # edits from changing the final evaluation implementation.
        evaluation=output/'final_evaluation'
        run_command([sys.executable,'-m','experiments.physics_conditioning.evaluate',str(run/'final_ckpt.pt'),
                     '--output',str(evaluation),'--seed-start','91000','--seed-count',str(args.eval_seed_count)],
                    cwd=snapshot,environment=environment,log=output/'final_policy_evaluation.log')
        final_policy=required_json(evaluation/'summary.json',
                                   ['checkpoint_sha256','metrics','paired_comparisons','reset_fingerprints'])
        if final_policy['checkpoint_sha256']!=hashlib.sha256((run/'final_ckpt.pt').read_bytes()).hexdigest():
            raise RuntimeError('Final policy evaluation does not match trained checkpoint')
        state.update(status='evaluating_retries',final_evaluation=str(evaluation))
        write_status(status_path,state)
        retry=output/'retry_evaluation'
        run_command([sys.executable,'-m','experiments.interaction_adaptation','--policy',str(run/'final_ckpt.pt'),
                     '--estimator',str(pinned_estimator),'--conditions',str(output/'retry_conditions.json'),
                     '--output',str(retry),'--sequences',str(args.retry_sequences),'--attempts','3','--seed','54000'],
                    cwd=snapshot,environment=environment,log=output/'final_retry_evaluation.log')
        retry_summary=required_json(retry/'summary.json',['attempt_budget','methods'])
        if retry_summary['attempt_budget']!=3 or {'fixed','adaptive','true'}-set(retry_summary['methods']):
            raise RuntimeError('Final retry comparison is incomplete')
        state.update(status='writing_assessment',retry_evaluation=str(retry))
        write_status(status_path,state)
        assessment=output/'assessment'
        assessment_command=[sys.executable,'-m','experiments.interaction_adaptation.assess',
                     '--policy-evaluation',str(evaluation/'summary.json'),
                     '--estimator-evaluation',str(output/'estimator_evaluation.json'),
                     '--retry-evaluation',str(retry/'summary.json'),
                     '--training-manifest',str(training_manifest_path),'--output',str(assessment)]
        if args.estimator_robustness:
            assessment_command.extend(['--estimator-robustness',str(output/'estimator_robustness.json')])
        run_command(assessment_command,
                    cwd=snapshot,environment=environment,log=output/'assessment.log')
        required_json(assessment/'assessment.json',['policy','estimator','closed_loop','training'])
        if not (assessment/'assessment.md').is_file():
            raise RuntimeError('Written final assessment is missing')
        state.update(status='complete',assessment=str(assessment),completed_at=datetime.now(timezone.utc).isoformat())
        write_status(status_path,state)
    except BaseException as error:
        state.update(failed_stage=state['status'],status='failed',error=repr(error),traceback=traceback.format_exc())
        write_status(status_path,state)
        raise
    finally:
        for number,handler in handlers.items():
            signal.signal(number,handler)


if __name__=='__main__':
    main()
