"""Run only the mandatory pretraining input guard; stop on its first failure."""
import json
from pathlib import Path
from experiments.property_probe.data import load_task,InputCorrelationError

out=Path(__file__).parent/'guard.json'
if out.exists():raise FileExistsError(out)
reports=[]
datasets=['runs/lift_slide_belief/slip_probe/paired/fixed/data',
          'runs/lift_slide_belief/slip_probe/paired/random/data',
          'runs/lift_slide_belief/three_task/data','runs/lift_slide_belief/nov13/data']
try:
    for dataset in datasets:
        for task in (0,1,2) if 'nov13' not in dataset else (0,1):
            for split in ('train','test'):
                _,_,report=load_task(dataset,split,task);reports.append(report)
except InputCorrelationError as exc:
    reports.append(exc.report)
    with out.open('x') as f:json.dump(dict(status='STOP_INPUT_CORRELATION_GUARD',reports=reports,probe_runs=0),f,indent=2)
    print(str(exc));raise SystemExit(2)
with out.open('x') as f:json.dump(dict(status='PASS',reports=reports,probe_runs=0),f,indent=2)
print('All checked task/split input guards passed; no probes run.')
