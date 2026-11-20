"""Save the provenance follow-up and STOP review; no fitting or simulation."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output).resolve()
commands=[['git','show','482a4d4:learning-withoutCLI/lift_task.py'],
          ['git','blame','-L','244,259','--','robot_adaptation/simulation/training_lift.py']]
records=[]
for cmd in commands:
    r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    records.append(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    if r.returncode:raise RuntimeError(r.stderr)
verdicts={
    'Leakage':dict(verdict='FAIL for unrestricted archived inputs',evidence='Saved observation42 equals true mg; observation45 equals true friction. Privileged context encodes true constants.',shuffled_control='not run'),
    'Confounds':dict(verdict='NOT PASSED',evidence='Historical mass_correlated_linear collection ties the targets. Main collections have finite sample correlation; no tolerance has been approved for the literal not-correlated requirement.',actions_only='not run'),
    'Splits':dict(verdict='NOT PASSED globally',evidence='Main train/test cube IDs are disjoint within collection; smoke test cube1 overlaps full train cube1. Some legacy archives lack persistent cube splits.',normalization='not fitted'),
    'Robustness':dict(verdict='NOT TESTED',evidence='No new simple/GRU runs or seed comparisons before STOP.'),
    'Skeptical PI':dict(verdict='FAIL if earlier privileged readouts are used as identification evidence',weakest_point='Confusing a representation of supplied constants with recovery from motion.',test='Inspected train.py encode(data.constants) and model.py constants_for_encoder; inspected saved probe.csv and comparison.json. The true-label pathway is confirmed.'),
    'Physics sanity':dict(verdict='NOT TESTED',evidence='There are no new episode-to-property predictions to compare with physical expectations. Direct answer channels cannot establish physical observability.'),
}
report=dict(status='STOP_REQUIRES_USER_DECISION',reason='User STOP: earlier reported results depend on privileged inputs and their evidential use must be settled.',
    correction='The initial historical lookup used the old directory name at commit482a4d4; this addendum preserves the failed lookup and resolves the correct learning-withoutCLI path.',
    commands=records,lenses=verdicts,
    affected_results=[
        dict(path='docs/experiments/physics_conditioned_policy.md:90',effect='True-input success and input-intervention contrasts assume supplied physics. Valid conditioning results; not mass/friction inference evidence.'),
        dict(path='docs/experiments/interaction_estimate_updates.md',effect='True-input retry reference is privileged. This finding alone does not invalidate fixed/adaptive arms or estimator recovery scores.'),
        dict(path='runs/context_dynamics/pick_gpu_2m/world_model_z8/RESULTS.md',effect='Readout R2 and world-model advantage use E(true constants). Valid privileged-context results; not recovery from episode observations.'),
        dict(path='BASELINE.md',effect='B2 was explicitly Oracle. Main fixed-grip state/action source audit finds no direct label contamination of B0/B1; present finding does not establish that their scores are invalid.'),
    ],training_runs=0,changes='Added audit artifacts and stop report only. Earlier results, data, and models preserved.')
with (out/'review.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print('Saved review.json; STOP remains in force.')
