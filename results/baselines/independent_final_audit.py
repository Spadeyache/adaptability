"""Audit saved baseline/adaptation artifacts without training or forecasting.

Run with --allow-partial during execution; omit it for the final completeness gate.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'results/baselines'
RUNS = ROOT/'runs/baselines'
TASKS = ('lift','slow_slide','shove')
MODELS = {'B0':('B0',0), **{f'B1_K{k}':('B1',k) for k in (1,5,10,20)}, 'B2':('B2',0)}
METRICS = ('one_step_mse','one_step_r2','rollout_10_mse')
BUFFERS = ('input_mean','input_sd','delta_mean','delta_sd','oracle_mean','oracle_sd')
SCIENTIFIC_CHECKS = 0


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a,b):
    global SCIENTIFIC_CHECKS
    SCIENTIFIC_CHECKS += 1
    if a is None or b is None:
        assert a is b, (a,b)
    else:
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11)


def finite(value):
    assert value is not None and np.isfinite(value), value


def metric_check(value, expected=None):
    n,r = value['moving_rows'],value['rollout_starts']
    if expected is not None:
        assert [value[k] for k in ('selected_rows','moving_rows','rollout_starts')] == list(expected)
    assert value.get('one_step_nonfinite_rows',0)==0
    assert value.get('rollout_nonfinite_starts',0)==0
    if n:
        finite(value['one_step_mse']); finite(value['one_step_sse']); finite(value['one_step_sst'])
        close(value['one_step_mse'],value['one_step_sse']/(51*n))
        close(value['one_step_r2'],1-value['one_step_sse']/value['one_step_sst'] if value['one_step_sst']>0 else None)
    else:
        assert value['one_step_mse'] is None and value['one_step_r2'] is None
    if r:
        finite(value['rollout_10_mse'])
    else:
        assert value['rollout_10_mse'] is None


def counts_from_data(protocol,split):
    """Read only bookkeeping/moving fields and independently reconstruct cohorts."""
    result={}
    held={tuple(x) for x in split['train_validation_episodes']}
    assert len(held)==1200
    for cube in range(200):
        for task in range(3):
            assert sum(x[0]==cube and x[1]==task for x in held)==2
    declared={(x['cube_id'],x['task_code'],x['episode_id']):x['role'] for x in split['test_episodes']}
    assert len(declared)==900
    for cube,task,ep in itertools.product(range(200,230),range(3),range(10)):
        assert declared[cube,task,ep]==('context' if ep<5 else 'eval')
    for part in ('train','test'):
        by_cube={}; all_hist=np.zeros((3,180,3),dtype=np.int64)
        for path in sorted((ROOT/protocol['data_directory']/part).glob('cube_*.npz')):
            with np.load(path,allow_pickle=False) as saved:
                cube=int(saved['cube_id'][0]); task=saved['task']; episode=saved['episode_id']; step=saved['step']
                moving=saved['cube_moving']; assert np.all(saved['cube_id']==cube)
            selected=np.zeros(len(step),bool)
            hist=np.zeros((3,180,3),dtype=np.int64)
            for t,e in itertools.product(range(3),range(10)):
                which=(task==t)&(episode==e)
                n=(120,120,180)[t]
                np.testing.assert_array_equal(step[which],np.arange(n))
                selected[which]=((cube,t,e) in held) if part=='train' else e>=5
            for code in range(3):
                a=selected&(task==code); b=a&moving; c=b&(step<=(120,120,180)[code]-10)
                for column,mask in enumerate((a,b,c)):
                    hist[code,:,column]=np.bincount(step[mask],minlength=180)
            by_cube[str(cube)]=hist; all_hist+=hist
        assert set(by_cube)==set(map(str,range(200) if part=='train' else range(200,230)))
        result[part]=dict(cubes=by_cube,hist=all_hist)
    np.testing.assert_array_equal(result['test']['hist'].sum(1),[[18000,11417,10521],[18000,7734,6427],[27000,829,829]])
    assert result['test']['hist'][:,:5,1].sum()==0
    return result


def evaluation_check(value,cohort,only_cube=None):
    hist=cohort['hist'] if only_cube is None else cohort['cubes'][str(only_cube)].copy()
    if only_cube is not None:
        hist[1:]=0 # Weight adaptation scores only this cube's lifts.
    metric_check(value['overall'],hist.sum((0,1)))
    for code,task in enumerate(TASKS):
        metric_check(value['per_task'][task],hist[code].sum(0))
    close(value['overall']['one_step_sse'],sum(v['one_step_sse'] or 0 for v in value['per_task'].values()))
    total=value['overall']['rollout_starts']
    if total:
        close(value['overall']['rollout_10_mse'],sum((v['rollout_10_mse'] or 0)*v['rollout_starts'] for v in value['per_task'].values())/total)
    for scope in ('overall',*TASKS):
        h=hist.sum(0) if scope=='overall' else hist[TASKS.index(scope)]
        windows=value['early_late']['overall'] if scope=='overall' else value['early_late']['per_task'][scope]
        for name,chunk in (('early',h[:5]),('late',h[5:])):
            metric_check(windows[name],chunk.sum(0))
        profile=value['step_profile']['overall'] if scope=='overall' else value['step_profile']['per_task'][scope]
        expected=set(np.flatnonzero(h[:,0]).tolist())
        assert {entry['step'] for entry in profile}==expected
        assert len(profile)==len(expected)
        for entry in profile:
            metric_check(entry,h[entry['step']])
    if only_cube is None:
        assert set(value['per_cube'])==set(cohort['cubes'])
        for cube,child in value['per_cube'].items():
            h=cohort['cubes'][cube]
            metric_check(child['overall'],h.sum((0,1)))
            for code,task in enumerate(TASKS):metric_check(child['per_task'][task],h[code].sum(0))
    else:
        assert set(value['per_cube'])=={str(only_cube)}
    meta=value['subset_metadata']
    assert meta['rollout_horizon_transitions']==10
    assert meta['selected_rows']==int(hist[:,:,0].sum())
    assert meta['moving_starts_without_full_horizon']==int(hist[:,:,1].sum()-hist[:,:,2].sum())


def check_architecture(checkpoint,kind,k):
    assert (checkpoint['kind'],checkpoint['history_k'],checkpoint['state_dim'],checkpoint['action_dim'])==(kind,k,51,8)
    weights=checkpoint['model']
    assert not any(any(token in name.lower() for token in ('train_z','latent','z_table','embedding')) for name in checkpoint)
    assert not any(any(token in name.lower() for token in ('train_z','latent','z_table','embedding')) for name in weights)
    assert weights['network.0.weight'].shape==(256,59+{'B0':0,'B1':64,'B2':2}[kind])
    assert weights['network.6.weight'].shape==(51,256)
    if kind=='B1':
        assert weights['encoder.weight_ih_l0'].shape==(192,110)
        assert weights['encoder.weight_hh_l0'].shape==(192,64)
    else:
        assert not any(name.startswith('encoder.') for name in weights)
    assert all(torch.isfinite(tensor).all() for tensor in weights.values())
    return weights


def hierarchical_check(saved,records,getter):
    for metric in METRICS:
        values={(row['base_seed'],row['order_id']):getter(row)[metric] for row in records}
        assert len(values)==25
        assert set(values)==set(itertools.product(range(5),range(5)))
        assert all(value is not None and np.isfinite(value) for value in values.values())
        expected={str(seed):float(np.mean([values[seed,order] for order in range(5)])) for seed in range(5)}
        score=saved[metric]
        close(score['mean'],np.mean(list(expected.values())))
        close(score['sd_across_seeds'],np.std(list(expected.values()),ddof=1))
        assert score['seed_count']==5 and score['trajectory_count']==25
        assert {(v['base_seed'],v['order_id']):v['value'] for v in score['raw']}==values
        for seed in range(5):
            close(score['seed_means'][str(seed)],expected[str(seed)])
            close(score['within_seed_order_sd'][str(seed)],np.std([values[seed,o] for o in range(5)],ddof=1))


def run(allow_partial=False):
    protocol=read(RESULTS/'protocol.json'); split=read(RESULTS/'split.json'); implementation=read(RESULTS/'implementation.json')
    protocol_hash=digest(RESULTS/'protocol.json'); split_hash=digest(RESULTS/'split.json')
    normalization_metadata=read(RESULTS/'normalization.json')
    assert normalization_metadata['protocol_sha256']==protocol_hash
    assert normalization_metadata['split_sha256']==split_hash
    assert normalization_metadata['implementation_sha256']==digest(RESULTS/'implementation.json')
    assert normalization_metadata['normalization_sha256']==digest(RESULTS/'normalization.npz')
    assert normalization_metadata['source_checkpoint_sha256']==digest(ROOT/normalization_metadata['source_checkpoint'])
    assert normalization_metadata['included_labels'] is False
    with np.load(RESULTS/'normalization.npz',allow_pickle=False) as archive:
        assert set(archive.files)==set(BUFFERS[:4])
        normalization={key:archive[key].copy() for key in archive.files}
    for key,dimension in zip(BUFFERS[:4],(59,59,51,51)):
        assert normalization[key].shape==(dimension,) and normalization[key].dtype==np.float32
    assert split_hash==protocol['split_sha256']
    for path,expected in implementation['sha256'].items():assert digest(ROOT/path)==expected,path
    for path,expected in protocol['input_sha256'].items():assert digest(ROOT/path)==expected,path
    cohort=counts_from_data(protocol,split)
    observed={}; checkpoints={}; missing=[]; reference=None; oracle_reference=None
    for name,(kind,k) in MODELS.items():
        for seed in range(5):
            folder=RUNS/name/f'seed_{seed}'; path=folder/'evaluation.json'
            if not path.exists():missing.append(f'{name}/seed_{seed}');continue
            saved=read(path); key=(name,seed)
            assert (saved['model'],saved['kind'],saved['history_k'],saved['seed'])==(name,kind,k,seed)
            assert saved['protocol_sha256']==protocol_hash and saved['split_sha256']==split_hash
            assert saved['checkpoint_sha256']==digest(folder/'checkpoint.pt')
            for source,expected in implementation['sha256'].items():assert saved['source_sha256'][source]==expected
            checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            weights=check_architecture(checkpoint,kind,k)
            assert checkpoint['training']==saved['training']==read(folder/'training.json')
            training=saved['training']
            assert (training['seed'],training['train_rows'],training['validation_rows'])==(seed,672000,168000)
            assert [r['epoch'] for r in training['history']]==list(range(1,training['epochs']+1))
            close(training['best_validation_mse'],min(r['validation_mse'] for r in training['history']))
            close(training['history'][training['best_epoch']-1]['validation_mse'],training['best_validation_mse'])
            if reference is None:reference={buffer:weights[buffer].clone() for buffer in BUFFERS[:4]}
            for buffer in BUFFERS[:4]:
                assert torch.equal(weights[buffer],reference[buffer]),(name,seed,buffer)
                assert weights[buffer].numpy().tobytes()==normalization[buffer].tobytes(),(name,seed,buffer)
            if kind=='B2':
                if oracle_reference is None:oracle_reference={buffer:weights[buffer].clone() for buffer in BUFFERS[4:]}
                for buffer in BUFFERS[4:]:assert torch.equal(weights[buffer],oracle_reference[buffer])
            else:
                assert torch.equal(weights['oracle_mean'],torch.zeros(2))
                assert torch.equal(weights['oracle_sd'],torch.ones(2))
            evaluation_check(saved['evaluation'],cohort['test'])
            assert saved['evaluation']['subset_metadata']['selected_episode_ids']==[5,6,7,8,9]
            observed[key]=saved;checkpoints[key]=weights
    config=protocol['adaptation']; checked_adapt=[];missing_adapt=[]; initial_validation={}
    for seed,order_id,rate in itertools.product(range(5),range(5),config['learning_rates']):
        folder=RUNS/'adaptation'/f'seed_{seed}'/f'order_{order_id}'/f'lr_{rate:g}';path=folder/'report.json'
        if not path.exists() or read(path)['status']!='complete':
            missing_adapt.append(f'seed_{seed}/order_{order_id}/lr_{rate:g}');continue
        value=read(path);identity=value['identity'];order=config['orders'][order_id]
        assert identity['protocol_sha256']==protocol_hash and identity['split_sha256']==split_hash
        assert (identity['base_seed'],identity['order_id'],identity['learning_rate'],identity['cube_order'])==(seed,order_id,rate,order)
        assert identity['base_checkpoint_sha256']==observed['B0',seed]['checkpoint_sha256']
        assert identity['order_seed']==config['order_seeds'][order_id]
        for source,expected in identity['source_sha256'].items():assert digest(ROOT/source)==expected,source
        assert value['normalization_buffers_unchanged'] and value['original_checkpoint_unchanged']
        assert digest(value['final_checkpoint'])==value['final_checkpoint_sha256']
        adapted=torch.load(value['final_checkpoint'],map_location='cpu',weights_only=True)
        weights=check_architecture(adapted,'B0',0);base=checkpoints['B0',seed]
        for buffer in BUFFERS:assert torch.equal(weights[buffer],base[buffer]),buffer
        assert any(not torch.equal(weights[name],base[name]) for name in weights if name not in BUFFERS)
        assert len(value['current_cubes'])==30
        assert [row['cube_id'] for row in value['current_cubes']]==order
        for position,row in enumerate(value['current_cubes']):
            assert row['position']==position;fit=row['adaptation'];cube=order[position]
            assert fit['fresh_adam'] and fit['replacement'] and fit['normalization_buffers_unchanged']
            assert (fit['context_rows'],fit['updates'],fit['batch_size'],fit['learning_rate'])==(600,100,1024,rate)
            assert len(fit['batch_losses'])==100 and all(np.isfinite(fit['batch_losses']))
            seed_expected=int(np.random.SeedSequence([271828,seed,order_id,position,cube]).generate_state(1,dtype=np.uint32)[0])
            assert fit['torch_seed']==seed_expected
            evaluation_check(row['evaluation'],cohort['test'],cube)
            assert row['evaluation']['subset_metadata']['selected_episode_ids']==[5,6,7,8,9]
        assert set(value['retention'])=={'0','1','5','20'}
        for k,entry in value['retention'].items():
            assert entry['cube_id']==order[0] and entry['further_cubes']==int(k)
            assert entry['reused_current_evaluation']==(k=='0')
            evaluation_check(entry['evaluation'],cohort['test'],order[0])
        assert value['retention']['0']['evaluation']==value['current_cubes'][0]['evaluation']
        for phase in ('before','after'):evaluation_check(value['train_validation_'+phase],cohort['train'])
        if seed in initial_validation:assert value['train_validation_before']==initial_validation[seed]
        else:initial_validation[seed]=value['train_validation_before']
        current=[r['evaluation']['per_task']['lift'] for r in value['current_cubes']]
        pooled=value['current_cube_pooled'];metric_check(pooled,cohort['test']['hist'][0].sum(0))
        close(pooled['one_step_sse'],sum(r['one_step_sse'] for r in current))
        close(pooled['rollout_10_mse'],sum(r['rollout_10_mse']*r['rollout_starts'] for r in current)/pooled['rollout_starts'])
        checked_adapt.append((seed,order_id,rate))
    complete=not missing and not missing_adapt
    aggregate_path=RESULTS/'adaptation/aggregate.json'
    adaptation_complete=not missing_adapt
    adaptation_summary=None
    if adaptation_complete:
        aggregate=read(aggregate_path)
        assert aggregate['complete_design'] and aggregate['complete_primary_metrics']
        assert aggregate['completed_trajectories']==aggregate['expected_trajectories']==50
        assert not aggregate['missing_primary_metrics']
        assert aggregate['protocol_sha256']==protocol_hash and aggregate['split_sha256']==split_hash
        assert len(aggregate['trajectories'])==50
        for row in aggregate['trajectories']:
            assert digest(row['report'])==row['report_sha256']
            actual=read(row['report'])
            assert row['current_cube_pooled']==actual['current_cube_pooled']
            assert row['retention']=={k:v['evaluation']['per_task']['lift'] for k,v in actual['retention'].items()}
            for phase in ('before','after'):
                stored=row['train_validation_'+phase];full=actual['train_validation_'+phase]
                assert stored==dict(overall=full['overall'],**full['per_task'])
            for k in ('0','1','5','20'):
                for metric in METRICS:
                    close(row['retention_change_from_k0'][k][metric],row['retention'][k][metric]-row['retention']['0'][metric])
            for task in ('overall',*TASKS):
                for metric in METRICS:
                    close(row['train_validation_change'][task][metric],row['train_validation_after'][task][metric]-row['train_validation_before'][task][metric])
        for rate in config['learning_rates']:
            records=[r for r in aggregate['trajectories'] if r['learning_rate']==rate]
            group=aggregate['learning_rates'][str(rate)]
            hierarchical_check(group['current_cube_pooled'],records,lambda r:r['current_cube_pooled'])
            for key in ('retention','retention_change_from_k0'):
                for k in ('0','1','5','20'):hierarchical_check(group[key][k],records,lambda r,key=key,k=k:r[key][k])
            for key in ('train_validation_before','train_validation_after','train_validation_change'):
                for task in ('overall',*TASKS):hierarchical_check(group[key][task],records,lambda r,key=key,task=task:r[key][task])
        adaptation_summary={rate:dict(
            first_cube_k0=group['retention']['0']['one_step_mse']['mean'],
            first_cube_k20=group['retention']['20']['one_step_mse']['mean'],
            first_cube_k20_minus_k0=group['retention_change_from_k0']['20']['one_step_mse']['mean'],
            current_cube_pooled=group['current_cube_pooled']['one_step_mse']['mean'],
            train_validation_before=group['train_validation_before']['overall']['one_step_mse']['mean'],
            train_validation_after=group['train_validation_after']['overall']['one_step_mse']['mean'])
            for rate,group in aggregate['learning_rates'].items()}
        if (RESULTS/'metrics.json').exists():
            assert read(RESULTS/'metrics.json')['adaptation']==aggregate
        for extension in ('png','pdf'):
            assert (RESULTS/f'plot3_retention.{extension}').stat().st_size>0
    if complete:
        final=read(RESULTS/'metrics.json');assert final['status']=='complete'
        assert {(r['model'],r['seed']):r for r in final['runs']}==observed
        assert final['adaptation']==aggregate
        for name in MODELS:
            assert final['completed_model_seeds'][name]==list(range(5))
            for task in ('overall',*TASKS):
                rows=[observed[name,seed]['evaluation']['overall' if task=='overall' else 'per_task'] for seed in range(5)]
                if task!='overall':rows=[r[task] for r in rows]
                for metric in METRICS:
                    values=[row[metric] for row in rows];score=final['summary'][name][task][metric]
                    assert score['values']==values and score['n']==score['expected_n']==5
                    close(score['mean'],np.mean(values));close(score['sd'],np.std(values,ddof=1))
        for plot in ('plot1_lift_baselines','plot2_history_by_step','plot3_retention'):
            for extension in ('png','pdf'):assert (RESULTS/f'{plot}.{extension}').stat().st_size>0
        assert (ROOT/'BASELINE.md').exists()
    result=dict(status='passed' if complete else 'partial',complete=complete,checked_model_runs=len(observed),expected_model_runs=30,
        checked_adaptation_trajectories=len(checked_adapt),expected_adaptation_trajectories=50,
        missing_model_runs=missing,missing_adaptation_trajectories=missing_adapt,
        all_model_normalizers_bitwise_equal=True,no_latent_tables=True,model_architectures_verified=True,
        all_adapted_normalizers_bitwise_equal=True,retention_further_cube_counts=[0,1,5,20],
        test_cube_ids=list(range(200,230)),moving_rows=19980,rollout_starts=17777,
        literal_first5_moving_rows=0,scientific_scalar_checks=SCIENTIFIC_CHECKS,
        protocol_sha256=protocol_hash,split_sha256=split_hash,implementation_sha256=digest(RESULTS/'implementation.json'),
        normalization_artifact_sha256=digest(RESULTS/'normalization.npz'),normalization_artifact_bitwise_verified=True,
        adaptation_complete=adaptation_complete,adaptation_summary=adaptation_summary,
        retention_comparison_scope='Each trajectory compares the same first cube and same evaluation episodes across k. Its k0 error is not compared with the global B0 test error.',
        audit_source_sha256=digest(__file__),forecast_recomputation='None; independent B0 pilot replay is in independent_pilot_audit.json.')
    (RESULTS/'independent_final_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('status','checked_model_runs','checked_adaptation_trajectories','scientific_scalar_checks')},indent=2))
    if not complete and not allow_partial:raise SystemExit('Required artifacts are still missing; audit is partial.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    run(args.allow_partial)
