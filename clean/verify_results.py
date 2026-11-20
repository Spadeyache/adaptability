"""Verify saved results and access receipts without re-evaluating any data."""
import json
import math
import statistics
from pathlib import Path
from clean.protocol import ROOT,SEEDS,PROTOCOL,digest,verify


def main():
    split=verify()
    paths=sorted((ROOT/'runs').glob('iteration_*/result.json'))
    results={json.loads(p.read_text())['iteration']:json.loads(p.read_text()) for p in paths}
    assert 1<=len(results)<=20
    expected_seeds=sorted(SEEDS)
    for number,r in results.items():
        assert sorted(run['seed'] for run in r['runs'])==expected_seeds
        if r.get('base_iteration'):
            a={'grip_force_max':30.,**results[r['base_iteration']]['config']}
            b={'grip_force_max':30.,**r['config']}
            assert len([k for k in b if a[k]!=b[k]])==1
        for run in r['runs']:
            fit=run['fit_log']
            assert fit['all_model_parameters_frozen']
            assert fit['parameter_sha256_before']==fit['parameter_sha256_after']
            assert fit['fit_episode_ids']==list(range(5))
            for info in run['training'].values():
                log=info['log']
                assert (log['updates'],log['batch_size'])==(2000,256)
                assert log['training_cube_ids']==list(range(200))
                assert log['training_episode_ids']==list(range(8))
    final=json.loads((ROOT/'results/final/result.json').read_text())
    chosen=results[final['iteration']]
    assert len(results)==20 or all(chosen['success'][f'S{i}'] for i in range(1,5))
    assert chosen['audit_sha256']==digest(ROOT/'results/full_audit.json')
    seal=json.loads((ROOT/'results/final/seal.json').read_text())
    assert seal['result_sha256']==digest(ROOT/'runs'/f"iteration_{final['iteration']:02d}"/'result.json')
    for row in seal['checkpoints'].values():
        for info in row.values():assert digest(ROOT/info['path'])==info['sha256']
    for path,sha in seal['recovery_sha256'].items():assert digest(ROOT/path)==sha
    events=[json.loads(line) for line in (ROOT/'results/final_evaluations.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events]==['begin','complete']
    assert all(e['candidate']==final['candidate'] for e in events)
    assert events[-1]['result_sha256']==digest(ROOT/'results/final/result.json')
    access=[json.loads(line) for line in (ROOT/'results/data_access.jsonl').read_text().splitlines()]
    heldout=[a for a in access if a['group']!='train']
    assert len(heldout)==4
    for a in heldout:
        assert a['timestamp']>=events[0]['timestamp']
        assert a['candidate']==final['candidate']
        assert a['episodes']==(list(range(5)) if a['purpose']=='final_fit' else list(range(5,10)))
    for group,record in final['groups'].items():
        assert sorted(r['seed'] for r in record['runs'])==expected_seeds
        assert len(record['summary'])==7
        for run in record['runs']:
            fit=run['fit_log'];assert fit['cube_ids']==split[group]
            assert fit['fit_episode_ids']==list(range(5)) and fit['all_model_parameters_frozen']
            assert fit['parameter_sha256_before']==fit['parameter_sha256_after']
            for name,metric in run['metrics'].items():
                assert len(metric['per_cube'])==30
                for key in ['one_step_nmse','rollout_error']:
                    mean=statistics.mean(cube[key] for cube in metric['per_cube'].values())
                    assert math.isclose(mean,metric[key],rel_tol=1e-10,abs_tol=1e-12)
        for model,summary in record['summary'].items():
            for metric in ['one_step_nmse','r2','rollout_error']:
                values=[r['metrics'][model][metric] for r in record['runs']]
                assert all(math.isfinite(v) for v in values)
                assert values==summary[metric]['values']
                assert math.isclose(statistics.mean(values),summary[metric]['mean'],rel_tol=1e-10,abs_tol=1e-12)
                assert math.isclose(statistics.stdev(values),summary[metric]['std'],rel_tol=1e-10,abs_tol=1e-12)
        s=record['summary'];p=s['plain']['one_step_nmse']['mean'];o=s['oracle']['one_step_nmse']['mean']
        f=s['fitted_z']['one_step_nmse'];z=s['shuffled_z']['one_step_nmse'];h=s['history']['one_step_nmse']['mean']
        assert record['success']['S1']==bool(p>o and (p-f['mean'])/(p-o)>=.5)
        assert record['success']['S2']==bool(f['mean']+f['std']<z['mean']-z['std'])
        assert record['success']['S3']==bool(f['mean']<=1.1*h)
        assert record['success']['S4']==all(record['failure_checks'].values())
    cubes=json.loads((ROOT/'data/cubes.json').read_text())['cubes']
    assert max(c['mass_kg'] for c in cubes if c['split']=='train') < min(c['mass_kg'] for c in cubes if c['split']=='out_of_range')
    report=dict(status='pass',completed_iterations=len(results),selected_iteration=final['iteration'],
        single_final_evaluation=True,heldout_access_records=len(heldout),metrics_recomputed_from_saved_scores=True,
        five_seed_sample_std_verified=True,equal_training_budget=True,frozen_weight_hashes_verified=True,
        one_change_per_iteration=True,immutable_split_verified=True,final_result_sha256=digest(ROOT/'results/final/result.json'))
    (ROOT/'results/final_integrity_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
