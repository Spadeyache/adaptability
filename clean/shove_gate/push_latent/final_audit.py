"""Independent arithmetic/integrity audit of saved Step2 artifacts only.

No simulator, model forward pass, adaptation optimizer, or predictive scoring is
called. The mass diagnostic is checked by algebra on saved vectors/coefficients.
"""
from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
SEEDS=[101,211,307,401,503]
VARIANTS=['plain','oracle','history','z2','z4','z8','z16']
MODELS=['plain','oracle','history','fitted_z','mean_z','wrong_z']
GROUPS=['push','moving_coast','all_steps']
PRIMARY='horizontal_velocity_delta_nmse'
METRICS=[PRIMARY,'one_step_nmse','rollout10_nmse','rollout10_final_nmse','stopping_position_error_m']
TRAIN=list(range(180));VALIDATION=list(range(180,200));TEST=list(range(260,290))
PROTOCOL_SHA='20fe5c2527ecbd061917d5f16fbf341397f1182b9cd2ec2369f2812794014ef1'

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise AssertionError(message)
def same(a,b,label,rtol=1e-10,atol=1e-12):
    require(np.allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),rtol=rtol,atol=atol),f'{label}: values differ')
def hashes(root,mapping):
    for name,expected in mapping.items():require(sha(Path(root)/name)==expected,f'Changed sealed artifact: {name}')
    return len(mapping)
def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def moment(values):
    a=np.asarray(values,dtype=float);require(len(a)==5 and np.isfinite(a).all(),'Need five finite seed values')
    return dict(seed_values=a.tolist(),mean=float(a.mean()),sample_sd=float(a.std(ddof=1)))
def verify_moment(value,expected,label):
    same(value['seed_values'],expected['seed_values'],label+' seeds')
    same(value['mean'],expected['mean'],label+' mean');same(value['sample_sd'],expected['sample_sd'],label+' SD')
def comparison(a,b):
    p=np.asarray(a['seed_values']);q=np.asarray(b['seed_values']);gap=p-q
    return dict(mean_gap=float(gap.mean()),relative_mean_improvement=float(gap.mean()/p.mean()) if p.mean()>0 else None,
        sum_seed_sds=float(p.std(ddof=1)+q.std(ddof=1)),
        beyond_seed_spread=bool(q.mean()+q.std(ddof=1)<p.mean()-p.std(ddof=1)),
        paired_seed_improvements=[float((x-y)/x) if x>0 else None for x,y in zip(p,q)],
        paired_gap_sample_sd=float(gap.std(ddof=1)))
def verify_comparison(a,b,record,label):
    expected=comparison(a,b)
    for key,value in expected.items():
        if isinstance(value,bool):require(value==record[key],label+' '+key)
        elif value is None:require(record[key] is None,label+' '+key)
        elif isinstance(value,list) and any(v is None for v in value):require(record[key]==value,label+' '+key)
        else:same(value,record[key],label+' '+key)
    return expected

def verify_aggregate(value,expected_cubes,episodes_per_cube,label):
    require(value['cubes']==len(expected_cubes),'Coverage cubes: '+label)
    require(value['episodes']==len(expected_cubes)*episodes_per_cube,'Coverage episodes: '+label)
    require(value['nonfinite_starts']==0 and value['eligible_starts']>0,'Finite coverage: '+label)
    cubes=value['per_cube'];require([c['cube_id'] for c in cubes]==expected_cubes,'Cube identities: '+label)
    require(all(c['episodes']==episodes_per_cube for c in cubes),'Per-cube episode coverage: '+label)
    same(value['mean'],np.mean([c['mean'] for c in cubes]),'Cube mean: '+label)
    if 'per_episode' in value:
        entries=value['per_episode'];require(len(entries)==len(expected_cubes)*episodes_per_cube,'Episode list: '+label)
        require(sum(e['starts'] for e in entries)==value['eligible_starts'],'Start counts: '+label)
        for c in cubes:
            chosen=[e['mean'] for e in entries if e['cube_id']==c['cube_id']]
            require(len(chosen)==episodes_per_cube,'Episode grouping: '+label)
            same(c['mean'],np.mean(chosen),'Episode mean: '+label)


def main():
    root=ROOT;results=root/'results';out=results/'final_audit.json'
    require(not out.exists(),'Final audit exists; do not overwrite evidence')
    ledger=read(results/'final_evaluation.json') if (results/'final_evaluation.json').exists() else None
    stop=read(results/'STOP.json') if (results/'STOP.json').exists() else None
    require((ledger and ledger.get('status') in ['complete','stopped','c1_failed']) or stop,'Audit waits for complete or explicit stop')
    require(sha(root/'PROTOCOL.md')==PROTOCOL_SHA,'Written preregistration changed')
    counts={}
    for label,manifest,base in [('data','data/seal.json',root/'data'),('collection_source','results/source_seal.json',root),
                               ('retired','results/retired_experiment_seal.json',root)]:
        counts[label]=hashes(base,read(root/manifest))
    require(counts['retired']==1368,'Retired snapshot coverage changed')
    pretraining=read(results/'pretraining_review.json') if (results/'pretraining_review.json').exists() else None
    if pretraining:
        require(pretraining['status']=='passed','Pretraining review failed')
        counts['pretraining_review']=hashes(root,pretraining['hashes'])
    split=read(root/'data/split.json')
    require(split['train']==TRAIN and split['validation']==VALIDATION and split['test']==TEST,'Cube split changed')
    require(split['retired_test']==list(range(200,260)),'Retired cube set changed')
    require(read(results/'full_audit.json')['status']=='passed','Collection integrity failed')
    if not ledger or ledger.get('status')!='complete':
        files=sorted((results/'test_metrics').glob('*.json'))
        result=dict(status='incomplete_protocol_stop',audit_unix_seconds=time.time(),source_sha256=sha(__file__),
            completed_comparison=False,recorded_stop=stop or ledger.get('failure'),hash_counts=counts,
            available_test_records=[str(p.relative_to(root)) for p in files],
            c1=read(results/'c1.json') if (results/'c1.json').exists() else None,
            c2=None,c3=None,no_missing_scores_inferred=True,no_predictions_training_or_adaptation_run=True)
        write_new(out,result);print(json.dumps(result,indent=2));return
    summary=read(results/'summary.json');require(ledger['summary_sha256']==sha(results/'summary.json'),'Summary seal changed')
    require(summary['primary_group']=='push' and summary['primary_metric']==PRIMARY,'Wrong primary endpoint')
    require(summary['test_cubes']==TEST and summary['query_episodes']==list(range(5,10)),'Wrong primary population')
    normalizer=read(results/'normalizer.json');scale_audit=read(results/'pretraining_scale_audit.json')
    require(normalizer['fit_cubes']==TRAIN and normalizer['fit_episodes']==list(range(8)),'Normalizer leakage')
    require(scale_audit['fit_cubes']==TRAIN and scale_audit['fit_episodes']==list(range(8)),'Independent scale leakage')
    scale=normalizer['horizontal_velocity_delta_mse_scale'];same(scale,scale_audit['scalar_push_delta_energy'],'Independent push energy')
    same(scale_audit['zero_predictor_nmse'],1.,'Zero predictor NMSE')
    require(normalizer['push_scale_coverage']['eligible_starts']==scale_audit['push_transitions'],'Push energy coverage')
    require(normalizer['push_scale_coverage']['cubes']==180 and normalizer['push_scale_coverage']['episodes']==1440,'Fitting normalization coverage')
    protocol=read(results/'model_protocol.json');require(protocol['train_cubes']==TRAIN and protocol['latent_validation_cubes']==VALIDATION and protocol['test_cubes']==TEST,'Model split mismatch')
    seal=read(results/'checkpoint_seal.json');expected={f'{v}_seed_{s}.pt' for s in SEEDS for v in VARIANTS}
    require(set(seal['checkpoints'])==expected,'Need all35 selected candidate checkpoints')
    counts['checkpoints']=hashes(root/'checkpoints',seal['checkpoints']);counts['model_manifest']=hashes(root,seal['manifest'])
    counts['test_data']=hashes(root,read(results/'test_data_seal.json'))
    adaptation=read(results/'adaptation_seal.json');counts['adaptation']=hashes(root,adaptation['files'])
    require(adaptation['checkpoint_seal_sha256']==sha(results/'checkpoint_seal.json'),'Adaptation/checkpoint seal mismatch')
    context=ledger['context']
    for key,path in [('checkpoint_seal_sha256','results/checkpoint_seal.json'),('adaptation_seal_sha256','results/adaptation_seal.json'),
                     ('test_data_seal_sha256','results/test_data_seal.json'),('data_seal_sha256','data/seal.json'),
                     ('source_sha256','evaluation.py'),('protocol_sha256','PROTOCOL.md')]:require(context[key]==sha(root/path),'Final frozen context '+key)
    require(summary['frozen_context']==context,'Summary frozen context mismatch')
    budgets={};selected_updates={};extension_count=0
    for seed in SEEDS:
        logs={};hashes_for_seed=set();updates_for_seed=set();selected_updates[str(seed)]={}
        for v in VARIANTS:
            cp=torch.load(root/'checkpoints'/f'{v}_seed_{seed}.pt',map_location='cpu',weights_only=False)
            t=read(results/'training_logs'/f'{v}_seed_{seed}.json')
            require(cp['variant']==v and cp['seed']==seed,'Checkpoint identity')
            require(cp['manifest']==seal['manifest'] and cp['protocol']==protocol and cp['normalizer']==normalizer,'Checkpoint provenance')
            require(cp['log']==t['log'],'Checkpoint/log disagreement');logs[v]=cp['log']
            updates=cp['completed_updates'];require(updates%10000==0 and 10000<=updates<=100000,'Update budget')
            require([r['update'] for r in logs[v]]==list(range(500,updates+1,500)),'Validation schedule')
            best=min(logs[v],key=lambda r:r['validation_nmse'])
            require(cp['selected_update']==best['update'] and t['selected_update']==best['update'],'Checkpoint minimum selection')
            same(cp['selected_validation_nmse'],best['validation_nmse'],'Selected validation score')
            for row in logs[v]:
                same(row['validation_nmse'],row['validation_push_horizontal_velocity_delta_nmse'],'Validation primary')
                same(row['best_validation_nmse'],min(x['validation_nmse'] for x in logs[v] if x['update']<=row['update']),'Running best')
                lr=.0001+.0009*.5*(1+math.cos(math.pi*min(row['update']-1,10000)/10000));same(row['learning_rate'],lr,'Learning rate')
            hashes_for_seed.add(cp['minibatch_sha256']);updates_for_seed.add(updates);selected_updates[str(seed)][v]=best['update']
        require(len(hashes_for_seed)==1 and len(updates_for_seed)==1,'Seven-variant paired stream/budget')
        updates=updates_for_seed.pop();budgets[str(seed)]=updates
        group=read(results/'training_logs'/f'group_seed_{seed}.json')
        require(group['completed_updates']==updates and group['minibatch_sha256'] in hashes_for_seed,'Group provenance')
        require(len(group['extension_decisions'])==updates//10000,'Extension record count')
        for endpoint,decision in zip(range(10000,updates+1,10000),group['extension_decisions']):
            extend=False;require(decision['endpoint']==endpoint,'Extension endpoint')
            for v in VARIANTS:
                before=min(r['validation_nmse'] for r in logs[v] if r['update']<=endpoint-2500)
                recent=min(r['validation_nmse'] for r in logs[v] if endpoint-2500<r['update']<=endpoint)
                expected_improvement=(before-recent)/before if before>0 else 0.;still=bool(before>0 and recent<=before*.999)
                r=decision['variants'][v];same(r['previous_best'],before,'Previous block best');same(r['final2500_best'],recent,'Final-window best')
                same(r['relative_improvement'],expected_improvement,'Extension improvement');require(r['still_improving']==still,'Extension threshold');extend|=still
            require(decision['extend']==extend,'Joint extension rule');extension_count+=1
            if endpoint<updates:require(extend,'Training continued after stopping condition')
        cap=bool(updates==100000 and group['extension_decisions'][-1]['extend'])
        require(group['convergence_not_established_at_cap']==cap,'Convergence cap flag')
        require(cap or not group['extension_decisions'][-1]['extend'],'Stopped while extension required')
    tuning=read(results/'latent_tuning.json');selected=read(results/'selected_config.json');records=tuning['records']
    grid={(d,lr,u,s) for d in [2,4,8,16] for lr in [.01,.03,.1] for u in [50,100,200,400] for s in SEEDS}
    require(len(records)==240 and {(r['z_dim'],r['learning_rate'],r['updates'],r['seed']) for r in records}==grid,'Complete validation grid')
    require(tuning['validation_cubes']==VALIDATION and tuning['support_episodes']==list(range(5)) and tuning['query_episodes']==list(range(5,10)),'Latent validation split')
    for row in records:verify_aggregate(row['query'],VALIDATION,5,'latent validation')
    tuning_index={(r['z_dim'],r['learning_rate'],r['updates'],r['seed']):r for r in records}
    local_paths=sorted((results/'latent_validation_runs').glob('*.json'))
    require(len(local_paths)==60,'Expected60 independent dimension/seed/lr fitting runs')
    for path in local_paths:
        local=read(path);require(local['weights_unchanged'],'Weights changed during validation latent fitting')
        require([r['updates'] for r in local['records']]==[50,100,200,400],'Latent snapshot grid')
        require([r['update'] for r in local['support_log']]==[50,100,200,400],'Saved latent fitting update counts')
        for row in local['records']:
            key=(row['z_dim'],row['learning_rate'],row['updates'],row['seed'])
            require(tuning_index[key]==row,'Validation aggregate differs from individual run')
            require(local['checkpoint_sha256']==seal['checkpoints'][f'z{row["z_dim"]}_seed_{row["seed"]}.pt'],'Validation fitting checkpoint changed')
    candidates=[]
    for d in [2,4,8,16]:
        for lr in [.01,.03,.1]:
            for updates in [50,100,200,400]:
                values=[r['query']['mean'] for seed in SEEDS for r in records if (r['z_dim'],r['learning_rate'],r['updates'],r['seed'])==(d,lr,updates,seed)]
                candidates.append(dict(z_dim=d,variant=f'z{d}',learning_rate=lr,updates=updates,validation_mean_nmse=float(np.mean(values)),validation_seed_sample_sd=float(np.std(values,ddof=1)),seed_values=values))
    require(len(tuning['candidates'])==48,'Candidate grid size')
    for expected,actual in zip(candidates,tuning['candidates']):
        require((expected['z_dim'],expected['learning_rate'],expected['updates'])==(actual['z_dim'],actual['learning_rate'],actual['updates']),'Candidate ordering')
        same(expected['seed_values'],actual['seed_values'],'Candidate seeds');same(expected['validation_mean_nmse'],actual['validation_mean_nmse'],'Candidate mean');same(expected['validation_seed_sample_sd'],actual['validation_seed_sample_sd'],'Candidate SD')
    best=min(candidates,key=lambda c:(c['validation_mean_nmse'],c['z_dim'],c['updates'],c['learning_rate']))
    require(selected==best and tuning['selection']==best,'Global validation-only selection/tiebreak')
    truth={int(c['cube_id']):float(c['mass_kg']) for c in read(root/'data/cubes.json')['cubes']}
    decoder_record=read(results/'mass_readouts.json');decoders=decoder_record['by_seed']
    require(decoder_record['selection_sha256']==sha(results/'selected_config.json'),'Mass readout selection changed')
    for seed in SEEDS:
        p=results/f'training_fitted_z_seed_{seed}.npz';decoder=decoders[str(seed)]
        with np.load(p,allow_pickle=False) as f:ids=f['cube_ids'].tolist();z=f['z'].astype(float)
        require(ids==TRAIN and decoder['fit_cube_ids']==TRAIN and decoder['fit_count']==180,'Mass decoder fitting cubes')
        require(z.shape==(180,selected['z_dim']),'Training refitted-z dimensions')
        require(decoder['fit_z_sha256']==sha(p),'Training refitted-z hash')
        mass=np.array([truth[c] for c in TRAIN]);design=np.column_stack([np.ones(180),z]);coefficients=np.linalg.pinv(design)@mass
        same(decoder['coefficients'],coefficients,'OLS decoder coefficients',rtol=1e-8,atol=1e-10)
        predicted=design@coefficients;same(decoder['training_mean_mass_kg'],mass.mean(),'Mean mass baseline')
        same(decoder['train_r2'],1-np.sum((predicted-mass)**2)/np.sum((mass-mass.mean())**2),'Decoder train R2')
        same(decoder['train_mae_kg'],np.abs(predicted-mass).mean(),'Decoder train MAE');same(decoder['train_rmse_kg'],np.sqrt(np.mean((predicted-mass)**2)),'Decoder train RMSE')
    names=['plain','oracle','history']+(['fitted_z','mean_z','wrong_z'] if summary['c1']['passed'] else [])
    require(set(summary['by_model'])==set(names),'Scored model family set')
    evaluated={};checked_metric_cells=0
    for name in names:
        for seed in SEEDS:
            record=read(results/'test_metrics'/f'{name}_seed_{seed}.json');evaluated[name,seed]=record
            require(record['model_name']==name and record['seed']==seed,'Test record identity')
            variant=name if name in ['plain','oracle','history'] else selected['variant']
            require(record['variant']==variant and record['checkpoint_sha256']==seal['checkpoints'][f'{variant}_seed_{seed}.pt'],'Scored checkpoint identity')
            require(record['test_cubes']==30 and record['test_episodes']==150 and record['query_episodes']==list(range(5,10)),'Test query dimensions')
            require(record['selected_update']==selected_updates[str(seed)][variant] and record['completed_updates']==budgets[str(seed)],'Scored checkpoint selection/budget')
            for group in GROUPS:
                for metric in METRICS:
                    value=record['groups'][group][metric];verify_aggregate(value,TEST,5,f'{name}/{seed}/{group}/{metric}');checked_metric_cells+=1
                    if metric==PRIMARY:
                        same(value['raw_mse_m2_s2'],value['mean']*scale,'Raw horizontal MSE');same(value['rmse_m_s'],math.sqrt(value['mean']*scale),'Horizontal RMSE')
        for group in GROUPS:
            for metric in METRICS:
                expected=moment([evaluated[name,s]['groups'][group][metric]['mean'] for s in SEEDS]);verify_moment(summary['by_model'][name][group][metric],expected,f'{name}/{group}/{metric}')
                if name!='plain':verify_comparison(summary['by_model']['plain'][group][metric],expected,summary['comparisons'][name][group][metric],f'vsPlain/{name}/{group}/{metric}')
    plain=summary['by_model']['plain']['push'][PRIMARY];oracle=summary['by_model']['oracle']['push'][PRIMARY]
    comp=verify_comparison(plain,oracle,summary['c1'],'C1');c1=bool(oracle['mean']<=.8*plain['mean'] and comp['beyond_seed_spread'])
    require(c1==summary['c1']['passed']==ledger['c1_passed'],'C1 decision')
    require(summary['c1']['coverage_complete'] and summary['c1']['at_least_twenty_percent']==(oracle['mean']<=.8*plain['mean']),'C1 components')
    c2=c3=None
    if c1:
        require(summary['selected_config']==selected,'Test adaptation selection changed')
        prefix_records={};mass_records={}
        mapping={c:TEST[(i+1)%30] for i,c in enumerate(TEST)}
        for seed in SEEDS:
            prefix=read(results/f'prefix_scores_seed_{seed}.json');prefix_records[seed]=prefix
            require(prefix['checkpoint_sha256']==seal['checkpoints'][f'{selected["variant"]}_seed_{seed}.pt'] and prefix['selected_config']==selected,'Prefix frozen config')
            require(prefix['support_counts']==list(range(1,6)) and prefix['query_episodes']==list(range(5,10)),'Prefix protocol')
            for count in range(1,6):
                verify_aggregate(prefix['scores'][str(count)],TEST,5,'Prefix query')
                require(np.asarray(prefix['z_vectors'][str(count)]).shape==(30,selected['z_dim']),'Prefix latent shape')
                require(prefix['fit_logs'][str(count)][-1]['update']==selected['updates'],'Prefix fitting budget')
            same(prefix['scores']['5']['mean'],evaluated['fitted_z',seed]['groups']['push'][PRIMARY]['mean'],'Prefix5/fitted agreement')
            actual={int(c):int(v) for c,v in evaluated['wrong_z',seed]['source_cube_mapping'].items()};require(actual==mapping,'Cyclic wrong-z assignment')
            z=np.asarray(prefix['z_vectors']['5'],dtype=float);decoder=decoders[str(seed)];pred=np.column_stack([np.ones(30),z])@np.asarray(decoder['coefficients'])
            mass=read(results/f'mass_scores_seed_{seed}.json');mass_records[seed]=mass
            require([r['cube_id'] for r in mass['predicted_mass_by_cube']]==TEST,'Mass prediction cube IDs')
            same([r['prediction_kg'] for r in mass['predicted_mass_by_cube']],pred,'Mass predictions from saved z')
            labels=np.array([truth[c] for c in TEST]);baseline=np.full(30,decoder['training_mean_mass_kg'])
            for candidate,record in [(pred,mass),(baseline,mass['baseline'])]:
                same(record['r2'],1-np.sum((candidate-labels)**2)/np.sum((labels-labels.mean())**2),'Mass R2')
                same(record['mae_kg'],np.abs(candidate-labels).mean(),'Mass MAE');same(record['rmse_kg'],np.sqrt(np.mean((candidate-labels)**2)),'Mass RMSE')
        for count in range(1,6):verify_moment(summary['prefix_curve'][str(count)],moment([prefix_records[s]['scores'][str(count)]['mean'] for s in SEEDS]),'Prefix moments')
        for metric in ['r2','mae_kg','rmse_kg']:
            verify_moment(summary['mass_readout'][metric],moment([mass_records[s][metric] for s in SEEDS]),'Mass moments')
            verify_moment(summary['mass_baseline'][metric],moment([mass_records[s]['baseline'][metric] for s in SEEDS]),'Mass baseline moments')
        fitted=summary['by_model']['fitted_z']['push'][PRIMARY];mean=summary['by_model']['mean_z']['push'][PRIMARY];wrong=summary['by_model']['wrong_z']['push'][PRIMARY]
        gap=plain['mean']-oracle['mean'];recovery=(plain['mean']-fitted['mean'])/gap;c2=bool(recovery>=.5)
        same(summary['c2']['recovered_fraction'],recovery,'C2 recovery');same(summary['c2']['plain_to_oracle_gap'],gap,'C2 oracle gap')
        same(summary['c2']['recovered_error'],plain['mean']-fitted['mean'],'C2 recovered error');require(summary['c2']['passed']==c2,'C2 decision')
        for p,o,z,r in zip(plain['seed_values'],oracle['seed_values'],fitted['seed_values'],summary['c2']['per_seed_recovered_fractions']):
            if p>o:same(r,(p-z)/(p-o),'Per-seed recovery')
            else:require(r is None,'Undefined per-seed recovery')
        a=verify_comparison(mean,fitted,summary['c3']['fitted_vs_mean_z'],'C3 mean');b=verify_comparison(wrong,fitted,summary['c3']['fitted_vs_wrong_z'],'C3 wrong')
        c3=bool(a['beyond_seed_spread'] and b['beyond_seed_spread']);require(summary['c3']['passed']==c3,'C3 decision')
    else:
        require(summary['c2'] is None and summary['c3'] is None and summary['prefix_curve'] is None and summary['mass_readout'] is None,'C1 failure StageB stop')
        require(not list(results.glob('prefix_scores_seed_*.json')) and not list(results.glob('mass_scores_seed_*.json')),'StageB artifacts despite C1 failure')
        require(summary['stopped_after_c1'] and ledger['stopped_after_c1'],'C1 stop ledger')
    require(summary['all_applicable_criteria_passed']==bool(c1 and c2 and c3)==ledger['all_applicable_criteria_passed'],'Overall criterion status')
    chronology={'protocol_mtime':(root/'PROTOCOL.md').stat().st_mtime,'collection_seal_mtime':(root/'data/seal.json').stat().st_mtime,
        'scale_audit_time':scale_audit['checked_unix_seconds'],'normalizer_mtime':(results/'normalizer.json').stat().st_mtime,
        'checkpoint_seal_mtime':(results/'checkpoint_seal.json').stat().st_mtime,'adaptation_seal_time':adaptation['completed_unix_seconds'],
        'test_started':ledger['started_unix_seconds'],'test_completed':ledger['completed_unix_seconds']}
    require(chronology['protocol_mtime']<=chronology['collection_seal_mtime']<=chronology['normalizer_mtime'],'Protocol/collection/fitting order')
    require(chronology['scale_audit_time']<=chronology['normalizer_mtime']<=chronology['checkpoint_seal_mtime']<=chronology['adaptation_seal_time']<=chronology['test_started']<=chronology['test_completed'],'Frozen evaluation chronology')
    require(pretraining is not None and pretraining['reviewed_unix_seconds']<=chronology['normalizer_mtime'],'Independent review must predate fitting')
    for p in (root/'checkpoints').glob('*.pt'):require(p.stat().st_mtime<=chronology['checkpoint_seal_mtime'],'Checkpoint after freeze')
    result=dict(status='passed',audited_unix_seconds=time.time(),source_sha256=sha(__file__),hash_counts=counts,
        metric_cells_checked=checked_metric_cells,validation_grid_records=240,candidate_configs=48,training_budgets=budgets,
        selected_updates=selected_updates,extension_decisions_checked=extension_count,selected_config=selected,
        c1_passed=c1,c2_passed=c2,c3_passed=c3,all_applicable_criteria_passed=bool(c1 and c2 and c3),
        chronology=chronology,all30fresh_cubes_and150_query_shoves_covered=True,
        no_predictions_training_or_adaptation_run=True,old_results_preserved=True,
        scalar_push_delta_energy=scale,summary_sha256=sha(results/'summary.json'))
    write_new(out,result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
