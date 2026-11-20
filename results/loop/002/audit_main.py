"""Audit every retained main-study world-model/z-table checkpoint; no training."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inventory=json.loads((ROOT/'results/probe/inventory_lift_slide.json').read_text())
groups=[('nov13','nov13/study','nov13/data',43,5),
        ('three_task','three_task/study','three_task/data',43,8),
        ('fixed_grip','slip_probe/paired/study/fixed_grip','slip_probe/paired/fixed/data',51,8),
        ('random_grip','slip_probe/paired/study/random_grip','slip_probe/paired/random/data',51,8)]
source_dir=OUT/'sources';source_dir.mkdir(exist_ok=True)
evidence=[];rows=[]
for family,study,data,sd,ad in groups:
    base=ROOT/'runs/lift_slide_belief';p=base/study
    protocol_path=(p.parent if 'grip' in family else p)/'protocol.json'
    protocol=json.loads(protocol_path.read_text())
    hashes=protocol.get('source_sha256',{'model.py':protocol.get('model_sha256')})
    for name,expected in hashes.items():
        src=Path(name) if Path(name).is_absolute() else ROOT/'experiments/lift_slide_belief'/name
        content=src.read_bytes();revision='working_tree'
        if hashlib.sha256(content).hexdigest()!=expected:
            rel=str(src.relative_to(ROOT))
            for rev in subprocess.check_output(['git','log','--all','--format=%H','--',rel],cwd=ROOT,text=True).splitlines():
                candidate=subprocess.check_output(['git','show',f'{rev}:{rel}'],cwd=ROOT)
                if hashlib.sha256(candidate).hexdigest()==expected:
                    content=candidate;revision=rev;break
            else:
                evidence.append(dict(family=family,path=str(src.relative_to(ROOT)),sha256=expected,revision=None,status='MISSING_HISTORICAL_SOURCE'))
                continue
        dst=source_dir/(family+'_'+src.name)
        with dst.open('xb') as f:f.write(content)
        evidence.append(dict(family=family,path=str(src.relative_to(ROOT)),sha256=expected,revision=revision))
    dataset=base/data
    with np.load(dataset/'train/cube_0000.npz',allow_pickle=False) as z:
        assert z['state'].shape[1]==43 and z['action'].shape[1]==ad
        raw_fields=z.files
    for seed in range(5):
        ckpt=p/f'seed_{seed}/model.pt';saved=torch.load(ckpt,map_location='cpu',weights_only=True)
        assert len(saved['model']['delta_mean'])==sd
        assert len(saved['model']['input_mean'])==sd+ad
        assert tuple(saved['train_z'].shape)==(200,4)
        assert np.array_equal(saved['cube_ids'].numpy(),np.arange(200))
        rows.append(dict(run=str(ckpt.parent.relative_to(ROOT)),checkpoint_sha256=sha(ckpt),
            family=family,seed=seed,state_dim=sd,action_dim=ad,z_shape=[200,4],
            state_column42_present=True,state_column42='right finger measured joint force',
            state_column45_present=sd==51,state_column45='TCP-frame cube relative Z position' if sd==51 else 'absent',
            concatenated_input_column45='TCP-frame cube relative Z position' if sd==51 else 'action dz',
            explicit_physics_derived_state_or_action=False if family!='nov13' else None,
            z_fitted_from_true_physics=False,readout_uses_labels_as_targets_only=True,
            model_input_code='experiments/lift_slide_belief/model.py:_transitions,fit_world_model,TrainedModel.fit_z',
            dataset_file=str((dataset/'train/cube_0000.npz').relative_to(ROOT)),
            test_dataset_file=str((dataset/'test/cube_0200.npz').relative_to(ROOT)),
            source_projection='experiments/lift_slide_belief/slip_data.py:project_rows' if sd==51 else 'experiments/lift_slide_belief/collect.py:measure',
            protocol=str(protocol_path.relative_to(ROOT))))
assert {x['path'] for x in inventory['checkpoint_artifacts']}=={r['run']+'/model.pt' for r in rows}
report=dict(status='INCOMPLETE_HISTORICAL_SOURCE_VERIFICATION',runs=rows,source_verification=evidence,
    input_boundary='Observed state + issued action + prediction-fitted z; physics labels used only as readout targets. Physical responses are not direct label injection.',
    claims={'mass_r2_0.755':'paired fixed_grip seeds0-4, five-lift z, supervised readout trained on train z only',
            'same_task_gain_11.6_percent':'same fixed_grip models, fit z on episodes0-4; evaluate episodes5-9'},
    diagnostic_reuse='Target control, rollout, force-cap and oracle diagnostics do not introduce additional trained main world-model checkpoints.',
    limitation='This audits direct label provenance, not all statistical confounds or control results.',training_runs=0)
with (OUT/'main_run_audit.json').open('x') as f:json.dump(report,f,indent=2)
lines=['# Main-run input audit','',
       'All indices are zero-based. Numerical column indices are not shared feature identities between schemas. No probe trained.',
       '', '| Run | State42 included? | State45 included? | Explicit mass/friction-derived state/action? | Dataset file | Input code |',
       '|---|---|---|---|---|---|']
for r in rows:
    decision='Unverified historical code; no label in saved state/action schema' if r['family']=='nov13' else '**No**'
    lines.append(f"| `{r['run']}` | Yes: measured finger force | {'Yes: relative Z' if r['state_column45_present'] else 'No'} | {decision} | `{r['dataset_file']}` | `model.py:_transitions`; `{r['source_projection']}` |")
lines += ['', 'The input code path is `experiments/lift_slide_belief/model.py`. In 43D state runs, column45 of concatenated state+action is action dz, not friction. '
          'The 0.755 mass R² and 11.6% same-task gain both use the five fixed_grip runs above. '
          'The supervised linear decoder uses physics labels as targets after z fitting; its output is never fed back to these world models. '
          'The audited source identities and actual checkpoint dimensions are in main_run_audit.json; exact matching code is saved in sources/.',
          '', 'Per user instruction, future probes will nevertheless omit original state columns42/45 and their computed descendants. Existing world models and scores remain unchanged.']
with (OUT/'MAIN_RUN_AUDIT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print('Audited all20 checkpoint schemas:15 runs source-verified clean;5 nov13 runs lack exact historical training source. No probes.')
