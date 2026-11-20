"""Read-only provenance audit. Creates new evidence files, never overwrites."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[3]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    snapshots=out/'snapshots';snapshots.mkdir(exist_ok=False)
    commands=[
        ['git','rev-parse','HEAD'],
        ['git','status','--short'],
        ['git','log','--all','--reverse','--format=%H %aI %cI %s','-S','object_weight_n','--','*.py'],
        ['git','show','3c716ec:learning/lift_task.py'],
        ['git','show','482a4d4:learning/lift_task.py'],
        ['git','log','--all','--reverse','--format=%H %aI %cI %s','-S','estimated_mass, estimated_friction = mass, friction','--','*.py'],
        ['git','log','--all','--reverse','--format=%H %aI %cI %s','-S','def inject_parameters','--','*.py'],
        ['git','blame','-L','45,57','--','robot_adaptation/simulation/collection_lift.py'],
        ['git','blame','-L','169,183','--','experiments/interaction_adaptation/evaluate.py'],
        ['git','blame','-L','25,35','--','experiments/physics_conditioning/evaluate.py'],
        ['git','log','--all','--reverse','--format=%H %aI %cI %s','--','robot_adaptation/estimators/context_dynamics/model.py'],
    ]
    log=[]
    for cmd in commands:
        r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
        log.append(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    paths=[
        'robot_adaptation/simulation/collection_lift.py',
        'robot_adaptation/simulation/training_lift.py',
        'experiments/interaction_adaptation/evaluate.py',
        'experiments/physics_conditioning/evaluate.py',
        'robot_adaptation/estimators/context_dynamics/model.py',
        'experiments/context_dynamics/train.py',
        'docs/experiments/physics_conditioned_policy.md',
        'docs/experiments/interaction_estimate_updates.md',
        'runs/context_dynamics/pick_gpu_2m/world_model_z8/RESULTS.md',
        'runs/context_dynamics/pick_gpu_2m/world_model_z8/comparison.json',
        'runs/context_dynamics/pick_gpu_2m/world_model_z8/probe.csv',
        'BASELINE.md','PROBE.md',
        'results/probe/leak_gate.json',
        'results/probe/inventory_lift_slide.json',
    ]
    identities=[]
    for name in paths:
        src=ROOT/name;dst=snapshots/name;dst.parent.mkdir(parents=True,exist_ok=True)
        data=src.read_bytes()
        with dst.open('xb') as f:f.write(data)
        identities.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest()))
    name='runs/interaction_adaptation/initial_policy_probe/episodes/condition2_sequence0_true_attempt0.npz'
    with np.load(ROOT/name,allow_pickle=False) as z:
        mass=float(z['label__mass_kg']);weights=z['sim__policy_observation'][:,42]
        checks=[dict(path=name,mass_kg=mass,reported_mass_kg=float(z['input__reported_mass_kg']),
            observation_shape=list(z['sim__policy_observation'].shape),column=42,
            unique_weights_n=np.unique(weights).tolist(),equals_true_mg=bool(np.allclose(weights,mass*9.81)))]
    trace=ROOT/'runs/physics_conditioning/baseline_freshscene_diagnostic/traces.jsonl'
    with trace.open() as f:row=json.loads(next(f))
    labels=json.loads(trace.with_name('episodes.json').read_text())
    paired=[x for x in labels if x['seed']==row['seed'] and x['mode']==row['mode']]
    obs=np.asarray(row['observation']); supplied=np.asarray(row['supplied'])
    checks.append(dict(path=str(trace.relative_to(ROOT)),mode=row['mode'],seed=row['seed'],
        lanes=len(paired),mass_column=42,friction_column=45,
        mass_matches=bool(np.allclose(obs[:,42],[x['mass_kg']*9.81 for x in paired])),
        friction_matches=bool(np.allclose(obs[:,45],[x['friction'] for x in paired])),
        policy_supplied_equals_raw=bool(np.array_equal(obs,supplied))))
    result=dict(status='STOP_REQUIRES_USER_DECISION',training_runs=0,checks=checks,
        source_identities=identities,commands=log,
        chronology_limit='Git dates establish presence in saved history, not the exact first execution date.',
        reason='Earlier privileged-input results cannot establish label-free identification; user requires a stop before deciding their treatment.')
    with (out/'audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],checks=checks),indent=2))

if __name__=='__main__':main()
