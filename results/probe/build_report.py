"""Assemble the inventory and honest not-run status after the Step 0 gate."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/probe'

def read(name):
    return json.loads((OUT/name).read_text())

def fmt(value):
    if value is None:
        return 'unknown / undefined'
    if isinstance(value,float):
        return f'{value:.6g}'
    if isinstance(value,(dict,list)):
        return json.dumps(value,ensure_ascii=False,separators=(',',':'))
    return str(value)

def cell(value):
    return fmt(value).replace('|','\\|').replace('\n',' ')

def link(name):
    return f'[{name}]({OUT/name})'

def main():
    lift=read('inventory_lift_slide.json'); hist=read('inventory_historical.json')
    other=read('inventory_other.json'); legacy=read('inventory_legacy.json'); gate=read('leak_gate.json')
    if not gate['direct_leak_found']:
        raise RuntimeError('This report is specifically for the stopped Step 0 audit; revise before reporting a pass.')
    rows=[]
    lines=['# Dataset information probe: stopped at Step 0','',
        '**No probes were trained.** Direct mass/friction encodings were found in older saved observations. '
        'The requested stop rule therefore triggered before the one-dataset, one-seed timing run. '
        'No existing model or dataset was modified, and no new simulator data was collected.','',
        'This inventory does not establish whether physical motion reveals mass or friction. '
        'It separates suitable candidate recordings from privileged inputs, derived outputs, and recordings without the required split.','',
        '## Verified stop evidence','',
        '| Saved artifact | Actual check | Consequence |','|---|---|---|',
        f"| `{gate['evidence'][0]['path']}` | `sim__policy_observation[:,42]` = {gate['evidence'][0]['observed_values_n'][0]:.9g} N; true mass = {gate['evidence'][0]['true_mass_kg']} kg. This is mass × gravity. | Direct answer inside the observation; true-mode policy also receives it. |",
        f"| `{gate['evidence'][1]['path']}` | Observation columns 42 and 45 match true mass × gravity and true friction for all {gate['evidence'][1]['batch_size']} saved lanes in the checked true-mode step. | Both answers are inside the observation; actions were generated with these inputs. |",
        '| Legacy inspection traces | Raw `state.input_vector` includes `object_weight_n`; some policy modes override that weight with a nominal value. | Saved raw state is privileged even when the policy used a nominal weight. |','',
        f"Exact values and source references: {link('leak_gate.json')}. The first two examples were independently checked. "
        'Wrong-parameter modes can also be invertible functions of truth; they are not automatically safe controls.','',
        '## Main lift, slide, and shove collections','',
        'These collections keep `true_mass` and `true_friction` outside observed state/action. '
        'No direct label injection was found in their documented state/action builders. '
        'Policy lifts use a constant mass guess; scripted speed, height, direction, force, and grip randomization do not read true cube parameters. '
        'The actions-only learned control has **not** been run, so this is a source audit, not a statistical control pass.','',
        '| Dataset | Tasks and total episodes per task | Cubes/configurations | Episodes per cube/task | Steps per episode (count histogram) | Mass–friction r |','|---|---|---|---|---|---|']
    for name,g in lift['groups'].items():
        if g['kind']!='control_transitions':
            continue
        tasks={('lift' if k=='0' else ('slide' if name=='nov13_main' else 'slow_slide') if k=='1' else 'shove'):v for k,v in g['task_episodes'].items()}
        n=len(g['unique_physical_records'])
        splits={k:len(v) for k,v in g['split_physical_keys'].items()}
        main=name in ['nov13_main','three_task_main','paired_fixed','paired_random']
        per=10 if main else 'see per-file records; no equal-count assumption'
        lines.append('| '+' | '.join(map(cell,[name,tasks,splits or n,per,g['episode_lengths'],g['mass_friction_correlation']['pearson']]))+' |')
        for task in tasks:
            rows.append(dict(dataset=name,paths=g['path_roots'],task=task))
    lines+=['', 'Counts include stored variants and shared recordings; they must not be added as independent cubes. '
        'The full collections share the same 230 cube identities. Smoke test cubes overlap the full training set; '
        'pilot and target caches do not provide the requested independent train/test population.','',
        '| Dataset | Exact saved roots | Generation / grip |','|---|---|---|']
    for name,g in lift['groups'].items():
        lines.append('| '+' | '.join(map(cell,[name,g['path_roots'],g['generation']]))+' |')
    g=lift['groups']['paired_fixed']; cs=g['split_correlations']
    lines+=['',f"For paired fixed/random and three-task data: train r = {cs['train']['pearson']:.9g}; test r = {cs['test']['pearson']:.9g}; all cubes r = {g['mass_friction_correlation']['pearson']:.9g}. "
             'These are observed sample correlations; independent sampling does not imply exactly zero sample correlation.','',
        f"All per-file fields, schemas, shapes, label-constancy checks, and split identities: {link('inventory_lift_slide.json')}. "
        'Its groups also distinguish belief banks, planner/oracle outputs, native force traces, and raw transitions.','',
        'Raw state has joint positions/velocities, gripper width, cube position/orientation/linear and angular velocities, '
        'arm effort, and finger forces. Paired datasets additionally log relative motion, finger error, actuator/contact signals, '
        'and episode diagnostics; explicitly labeled truth, IDs, and future outcome summaries are not probe features.','',
        '## Other recorded datasets','',
        'A physical setting is not automatically a persistent cube. Where no cube IDs or existing cube-disjoint test split are saved, '
        'that requirement remains unresolved; no split was invented. Null correlation means insufficient variation or missing labels.','',
        '| Saved path | Task(s) | Episodes or images | Distinct physical configurations / objects | Steps per episode | Mass–friction r |','|---|---|---|---|---|---|']
    for g in hist['groups']:
        tasks=g.get('task_names') or [g.get('task','unspecified')]
        lines.append('| '+' | '.join(map(cell,[g['path'],tasks,g.get('episodes',g.get('source_episodes',g.get('images'))),
            g.get('unique_full_physics_configurations'),g.get('steps_per_episode',g.get('source_steps_per_episode')),
            g.get('mass_friction_pearson_unique_full_configurations')]))+' |')
        for task in tasks:
            rows.append(dict(dataset=g['path'],paths=[g['path']],task=task))
    for g in other['raw_npz_families']:
        lines.append('| '+' | '.join(map(cell,[g['path'],g['task'],g['episode_count'],g['unique_physical_setting_count'],
            g['steps_histogram'],g['mass_friction_pearson_across_unique_settings']]))+' |')
        rows.append(dict(dataset=g['path'],paths=[g['path']],task=g['task']))
    for g in other['pretrained_episode_families']:
        lines.append('| '+' | '.join(map(cell,[g['path'],g['task'],g['episode_count'],'no persistent cube IDs',
            g['steps_histogram'],'friction not saved']))+' |')
        rows.append(dict(dataset=g['path'],paths=[g['path']],task=g['task']))
    for g in other['jsonl_artifacts']:
        if g['kind']!='raw_batched_policy_step_trace':
            continue
        lines.append('| '+' | '.join(map(cell,[g['path'],'physics-conditioned lift',g['episode_count'],
            f"{g['batch_size']} batch lanes; no cube IDs",g['steps_per_episode_histogram'],'see paired physical-setting inventory']))+' |')
        rows.append(dict(dataset=g['path'],paths=[g['path']],task='physics-conditioned lift'))
    lines+=['', '## Legacy inspection and deployment traces','',
        'Here the unit is a saved trace file. The full manifest lists every path, row count, action-row count, field schema, '
        'and recorded step bounds. Missing physical labels are marked unknown, rather than counted as zero cubes.','',
        '| Root | Trace files | Recovered mass/friction pairs | Saved final-step/step-count histogram | r over recovered unique pairs |','|---|---|---|---|---|']
    for g in legacy['groups']:
        lines.append('| '+' | '.join(map(cell,[g['path'],g['files'],g['unique_mass_friction_pairs'] or 'unknown',
            g['final_step_index_or_reported_steps_histogram'],g['pearson_mass_friction_over_unique_pairs']]))+' |')
        rows.append(dict(dataset=g['path'],paths=[g['path']],task=g['task']))
    lines+=['', 'Three saved inspection runs use `mass_correlated_linear` friction; their measured correlation across '
        'the five physical settings is −1 to floating-point precision. They cannot establish separate mass/friction identification.','',
        '## Field inventory, labels, generation, and exclusions','',
        f"- {link('inventory_lift_slide.json')}: every NPZ file and schema in the lift/slide family, state/action column maps, per-cube label constancy, RNG provenance, source hashes, and related result artifacts.",
        f"- {link('inventory_historical.json')}: context dynamics, interaction snippets, mass belief, RMA, and initial RGB. Each group lists saved fields, generation/grip, labels, splits, and correlations.",
        f"- {link('inventory_other.json')}: all raw failure collection, interaction adaptation, physics conditioning, pretrained policy traces, and referenced visual-review archives. Field lists and dimensions are recorded, with per-family label and policy-input checks.",
        f"- {link('inventory_legacy.json')}: every legacy JSONL/NPZ trace path and complete flattened field schemas, plus checkpoints, archived feature arrays, and aggregate logs classified separately.",
        f"- {link('file_inventory.json')}: candidate containers across the repo. This includes checkpoints, reports, and vendored library test fixtures; its file count is not a dataset count. RGB images are inventoried through the visual-mass manifest.",'',
        'Mass/friction storage is confirmed per cube for the main lift/slide collections, and per episode or physical configuration '
        'for the historical families that save labels. Some old traces lack friction labels. The raw collection schemas save mass, weight, '
        'inertia, materials, and other truth metadata under explicit label fields; none may be blanket-concatenated with observations.','',
        'Additional limitations recorded in the inventory:','',
        '- RMA oracle and true-mode mass-belief actions were generated using true parameters even when their saved state vectors remove those parameters.',
        '- Context-dynamics constants and true-physics embeddings are privileged side fields. The saved kinematic state is separate.',
        '- Visual mass is an image dataset with mass determined by object size and fixed density; it has no interaction sequence and no varied-friction target.',
        '- Fixed-friction or single-setting datasets cannot yield a meaningful friction R². Smoke datasets with one held-out cube also cannot support across-cube R².',
        '- Existing episode splits are not automatically cube-disjoint splits. Duplicated raw/prepared/archive recordings and shared shoves cannot be treated as independent evidence.',
        '- Correlation in feedback-generated actions can reflect observed physical responses. An actions-only probe would diagnose information in that channel; source inspection is needed to distinguish direct label use from ordinary feedback.','',
        '## Probe results','',
        '**All cells below are not run.** No training, validation, shuffled-label, actions-only, ablation, moving/still, or pooling score exists for this diagnostic. '
        'No GRU bar chart was generated. Zero scores would incorrectly imply measurements. The mean-reference R²=0 is a definition, '
        'not an executed control; a train-mean predictor evaluated on test data need not score zero.','',
        '| Dataset × task | Simple mass R² | Simple friction R² | GRU mass R² | GRU friction R² | Shuffled mass R² | Shuffled friction R² | Actions mass R² | Actions friction R² |',
        '|---|---|---|---|---|---|---|---|---|']
    for row in rows:
        row.update(status='not_run_global_step0_leak_gate',seeds_completed=[],
            simple=dict(mass=None,friction=None),gru=dict(mass=None,friction=None),
            shuffled=dict(mass=None,friction=None),actions_only=dict(mass=None,friction=None))
        lines.append('| '+cell(row['dataset']+' × '+row['task'])+' | '+' | '.join(['not run']*8)+' |')
    lines+=['', '## Exact commands to reproduce this inventory','',
        'These commands read existing artifacts and write only the diagnostic reports. They do not train or collect data.','',
        '```bash','cd /home/ubuntu/faliure',
        '.venv/bin/python results/probe/index_files.py',
        '.venv/bin/python results/probe/audit_lift_slide.py',
        'PYTHONPATH=. .venv/bin/python results/probe/audit_historical.py',
        '.venv/bin/python results/probe/audit_other.py',
        '.venv/bin/python results/probe/audit_legacy.py',
        '.venv/bin/python results/probe/audit_leak_gate.py',
        '.venv/bin/python results/probe/build_report.py','```','',
        'There is no probe-training rerun command or first-seed runtime: implementation/training stopped at the required gate. '
        'Continuing requires settling the dataset/feature exclusions and existing-split limitations described above. '
        'Deleting explicit labels alone does not remove information already supplied to the acquisition policy.','']
    metrics=dict(schema_version=1,status='STOPPED_BEFORE_TRAINING',stage='step0_inventory',
        reason='Direct true mass/friction or invertible label-derived values found in saved observation vectors.',
        requested_seeds=5,completed_probe_runs=0,first_dataset_first_seed_runtime_seconds=None,
        controls_status='not_run',ablation_results=None,moving_still_results=None,episode_pooling_results=None,
        plot_status='not_created_no_measured_scores',gate=gate,
        inventory_artifacts=['results/probe/'+x for x in ['file_inventory.json','inventory_lift_slide.json',
            'inventory_historical.json','inventory_other.json','inventory_legacy.json','leak_gate.json']],
        metric_rows=rows,interpretation='Inventory findings only. No conclusion yet about recovery of mass/friction from physical responses.')
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    (ROOT/'PROBE.md').write_text('\n'.join(lines))
    print(f'Saved PROBE.md and metrics.json: {len(rows)} dataset/task entries, all not run.')

if __name__=='__main__':
    main()
