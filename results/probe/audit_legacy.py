"""Read-only inventory of legacy traces. No model loading or training."""
from __future__ import annotations
import collections
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/probe'
PREFIXES = ('runs/inspect/', 'runs/learning/', 'runs/policy/', 'runs/checks/',
            'runs/diagnosis/', 'runs/deployment/', 'runs/article_inspired/',
            'runs/maniskill_example/', 'failure_discovery/results/', 'data_check/')

def fields(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for k, v in value.items():
            result.update(fields(v, f'{prefix}.{k}' if prefix else k))
    elif isinstance(value, list):
        try:
            a = np.asarray(value)
            result[prefix] = {'shape': list(a.shape), 'dtype': str(a.dtype)}
        except ValueError:
            result[prefix] = {'length': len(value), 'type': 'ragged_list'}
    else:
        result[prefix] = {'type': type(value).__name__}
    return result

def corr(pairs):
    if len(pairs) < 2:
        return None
    a = np.asarray(pairs, float)
    return float(np.corrcoef(a.T)[0, 1]) if np.all(np.std(a, axis=0) > 1e-12) else None

def main():
    index = json.loads((OUT / 'file_inventory.json').read_text())
    candidates = [x for x in index['candidate_files'] if x['path'].startswith(PREFIXES)]
    records, non_trajectories, errors, schemas = [], [], [], {}
    for item in candidates:
        rel = item['path']; p = ROOT / rel
        if '/.deps/' in rel:
            non_trajectories.append({'path': rel, 'kind': 'vendored_library_test_data'})
            continue
        if p.suffix == '.jsonl':
            try:
                with p.open() as f:
                    rows = [json.loads(line) for line in f if line.strip()]
                is_trace = bool(rows) and any('state' in r or 'observation' in r or 'obs' in r for r in rows)
                if not is_trace:
                    non_trajectories.append({'path': rel, 'kind': 'training_or_summary_log', 'rows': len(rows)})
                    continue
                schema = {}
                for r in rows:
                    schema.update(fields(r))
                key = str(len(schemas))
                for k, v in schemas.items():
                    if v == schema:
                        key = k; break
                else:
                    schemas[key] = schema
                states = [r['state'] for r in rows if isinstance(r.get('state'), dict)]
                masses = sorted({float(s['mass_kg']) for s in states if 'mass_kg' in s})
                frictions = sorted({float(s['friction']['cube_static_friction']) for s in states
                                    if isinstance(s.get('friction'), dict) and 'cube_static_friction' in s['friction']})
                direct = any('object_weight_n' in s.get('inputs', {}) for s in states)
                truth_matches = sum(np.isclose(float(s['inputs']['object_weight_n'][0]),
                                              float(s['mass_kg']) * 9.81, rtol=1e-5)
                                    for s in states if 'mass_kg' in s and 'object_weight_n' in s.get('inputs', {}))
                record = dict(path=rel, kind='episode_trace', schema_id=key,
                              logged_rows=len(rows), step_min=min([r['step'] for r in rows if 'step' in r], default=None),
                              step_max=max([r['step'] for r in rows if 'step' in r], default=None),
                              rows_with_action=sum(any(k in r for k in ['action', 'raw_action', 'applied_action']) for r in rows),
                              masses_kg=masses, frictions=frictions, cube_id=None,
                              supplied_weight_in_saved_state=direct, rows_with_weight_equal_true_mg=int(truth_matches),
                              split='No existing cube-disjoint split in trace; not invented.')
                records.append(record)
            except Exception as exc:
                errors.append({'path': rel, 'error': repr(exc)})
        elif p.suffix == '.npz':
            try:
                with np.load(p, allow_pickle=False) as z:
                    schema = {k: {'shape': list(z[k].shape), 'dtype': str(z[k].dtype)} for k in z.files}
                if p.name == 'trajectory.npz':
                    meta_path = p.with_name('episode.json')
                    meta = json.loads(meta_path.read_text())
                    actual = meta.get('actual_physics', {})
                    records.append(dict(path=rel, kind='deployment_episode', schema=schema,
                                        steps=meta.get('steps'), masses_kg=[actual['mass_kg']] if 'mass_kg' in actual else [],
                                        frictions=sorted({x['static_friction'] for x in actual.get('object_materials', [])}),
                                        generation='Saved policy with initial RGB-estimated mass input, fixed thereafter.',
                                        mass_source='episode.json actual_physics', friction_source='episode.json actual_physics.object_materials',
                                        cube_id=None, split=None,
                                        supplied_weight='Image estimate, not true label; geometry fixes mass via constant density.'))
                else:
                    non_trajectories.append(dict(path=rel, kind='derived_features_or_fitted_model', schema=schema))
            except Exception as exc:
                errors.append({'path': rel, 'error': repr(exc)})
        else:
            non_trajectories.append({'path': rel, 'kind': 'checkpoint' if p.suffix in ['.pt','.pth'] else 'metadata_or_derived_report'})
    groups = collections.defaultdict(list)
    for r in records:
        parts = r['path'].split('/')
        groups['/'.join(parts[:3])].append(r)
    summary = []
    for name, rs in sorted(groups.items()):
        pairs = sorted({(r['masses_kg'][0],r['frictions'][0]) for r in rs if len(r['masses_kg'])==1 and len(r['frictions'])==1})
        histogram = collections.Counter((r.get('steps') if r.get('steps') is not None else r.get('step_max')) for r in rs)
        per_setting = collections.Counter((r['masses_kg'][0],r['frictions'][0]) for r in rs if len(r['masses_kg'])==1 and len(r['frictions'])==1)
        summary.append(dict(path=name, task='lift/hold/carry inspection' if 'deployment' not in name else 'image-estimate lift deployment',
                            files=len(rs), unique_mass_friction_pairs=len(pairs), mass_friction_pairs=pairs,
                            identified_cube_count=None, cube_identity_note='No cube IDs; unique parameter pairs are not claimed to be unique cubes.',
                            episodes_per_setting=[{'mass':m,'friction':mu,'episodes':n} for (m,mu),n in sorted(per_setting.items())],
                            final_step_index_or_reported_steps_histogram={str(k):v for k,v in histogram.items()},
                            pearson_mass_friction_over_unique_pairs=corr(pairs),
                            supplied_weight_in_state_files=sum(r.get('supplied_weight_in_saved_state',False) for r in rs),
                            generation='Policy inspections/interventions; hold force scripts for diagnosis. See source and per-file rows.',
                            split='No existing cube-disjoint split verified in these records.'))
    report = dict(status='inventory_only_global_leak_gate_stopped', groups=summary, episodes=records,
                  schemas=schemas, other_artifacts=non_trajectories, errors=errors,
                  sources=['robot_adaptation/policy/inspect.py:65-84',
                           'robot_adaptation/runtime/deploy.py:153-179'],
                  findings=['Legacy inspection state.input_vector includes object_weight_n and state includes physics metadata.',
                            'Policy true modes use true weight; nominal policy input can differ from saved raw state vector.',
                            'Some legacy runs use mass_correlated_linear friction; pair correlations are reported, not an independence claim.',
                            'Archived gaps.npz is derived from 600 existing episodes; it is not newly collected data.',
                            'Checkpoints, model arrays, aggregate reports and vendored test fixtures are not episode datasets.'])
    (OUT/'inventory_legacy.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(groups=len(summary),episode_files=len(records),schemas=len(schemas),errors=errors),indent=2))

if __name__ == '__main__':
    main()
