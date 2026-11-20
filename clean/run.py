"""Execute one logged validation candidate or the single sealed final evaluation."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from clean.protocol import ROOT, PROTOCOL, SEEDS, digest, verify, write_new
from clean.dataset import fit_normalizer, load_pack
from clean.models import DynamicsModel
from clean.train import train_model, fit_z
from clean.evaluate import evaluate, aggregate_seed_metrics, success_criteria, derangement
from clean.audit import static_provenance, r2_per_property, summarize

DEFAULT = dict(width=128,depth=2,z_dim=2,lr=.001,z_l2=.0001,fit_steps=200,
               fit_lr=.02,moving_weight=1.0,prediction_horizon=1,grip_force_max=30.0)
KINDS=['plain','history','oracle','two_part']
ABLATIONS=['fitted_z','mean_z','shuffled_z','another_cube_z','learned_z']


def safe(obj):
    if isinstance(obj,dict):return {str(k):safe(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [safe(v) for v in obj]
    if isinstance(obj,(np.floating,float)) and not np.isfinite(obj):return None
    if isinstance(obj,np.generic):return obj.item()
    if torch.is_tensor(obj):return safe(obj.detach().cpu().tolist())
    return obj


def append(path,event):
    with Path(path).open('a') as f:f.write(json.dumps(safe(event),allow_nan=False)+'\n')


def key_for(kind,config,seed):
    cfg={k:config[k] for k in ['width','depth','lr','moving_weight','prediction_horizon']}
    if kind=='two_part':cfg.update(z_dim=config['z_dim'],z_l2=config['z_l2'])
    signature=dict(kind=kind,config=cfg,seed=seed,protocol=digest(ROOT/'data/protocol.json'),
        normalizer=digest(ROOT/'data/normalizer.json'),audit=digest(ROOT/'results/full_audit.json'),
        sources={name:digest(ROOT/name) for name in ['models.py','train.py','dataset.py']})
    return hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:20]


def load_model(path,device):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    cfg=checkpoint['config']
    model=DynamicsModel(checkpoint['kind'],48,9,width=cfg['width'],depth=cfg['depth'],z_dim=cfg['z_dim']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    return model,checkpoint


def require_audit():
    verify()
    report=json.loads((ROOT/'results/full_audit.json').read_text())
    if report['status']!='pass':raise RuntimeError('Full data audit did not pass')
    if report['split_sha256']!=digest(ROOT/'data/split.json'):raise RuntimeError('Split changed')
    static=static_provenance()
    if not static['pass']:raise RuntimeError(str(static['violations']))
    for name in ['schema.py','script.py','native.py']:
        if report['static_provenance']['source_sha256'][name]!=digest(ROOT/name):
            raise RuntimeError('Audited data generator changed: '+name)
    for name,expected in report['episode_sha256'].items():
        if digest(ROOT/'data'/name)!=expected:
            raise RuntimeError('Audited training episode changed: '+name)
    return report,static


def z_probe(z,cube_ids,seed):
    """Fixed ridge probe of fresh fitted z, evaluated across training-cube folds."""
    ids=cube_ids.detach().cpu().numpy()
    zz=z.detach().cpu().numpy().astype(np.float64)
    props=json.loads((ROOT/'data/cubes.json').read_text())['cubes']
    labels=np.asarray([[props[i]['mass_kg'],props[i]['friction']] for i in ids])
    order=np.random.default_rng(seed).permutation(len(ids));n=max(2,int(np.ceil(.2*len(ids))))
    te,tr=order[:n],order[n:]
    xm=zz[tr].mean(0);xs=np.maximum(zz[tr].std(0),1e-8)
    ym=labels[tr].mean(0);ys=np.maximum(labels[tr].std(0),1e-8)
    x=np.c_[(zz-xm)/xs,np.ones(len(zz))]
    y=(labels-ym)/ys
    reg=np.eye(x.shape[1])*.001;reg[-1,-1]=0
    w=np.linalg.solve(x[tr].T@x[tr]+reg,x[tr].T@y[tr])
    r2=r2_per_property(labels[te],(x[te]@w)*ys+ym)
    return dict(seed=seed,train_cube_ids=ids[tr].tolist(),heldout_cube_ids=ids[te].tolist(),
        mass_r2=float(r2[0]),friction_r2=float(r2[1]))


def score_two_part(model,fitpack,scorepack,norm,config,seed,outdir,include_learned):
    fitted=fit_z(model,fitpack,config,seed)
    ids=fitted['cube_ids'];z=fitted['z']
    table=model.mean_z()[None,:].expand(260,-1).clone()
    table[ids]=z
    shuffled=table.clone()
    permutation=torch.as_tensor(derangement(seed,len(ids)),device=z.device,dtype=torch.long)
    shuffled[ids]=z[permutation]
    mean=model.mean_z()[None,:].expand(260,-1).clone()
    other=mean.clone()
    other[ids]=model.embedding.weight.detach()[(ids+1)%200]
    scores={}
    for name,values in [('fitted_z',table),('mean_z',mean),('shuffled_z',shuffled),('another_cube_z',other)]:
        scores[name]=evaluate(model,scorepack,norm,z_by_cube=values)
    if include_learned:scores['learned_z']=evaluate(model,scorepack,norm)
    torch.save({'cube_ids':ids.cpu(),'z':z.detach().cpu(),'fit_log':fitted['log'],
                'shuffle_indices':permutation.cpu()},outdir/f'z_seed_{seed}_{scorepack.group}.pt')
    return scores,fitted


def candidate(number,config,hypothesis,change,device,base_iteration=None):
    audit,static=require_audit()
    if (ROOT/'results/final_evaluations.jsonl').exists():
        raise RuntimeError('Final evaluation has begun; no further tuning is permitted')
    if number<1 or number>20:raise ValueError('20-iteration budget')
    generator_path=ROOT/'data/generator.json'
    if generator_path.exists():
        generator=json.loads(generator_path.read_text())
        if config.get('grip_force_max',30.0)!=generator['grip_force_max']:
            raise RuntimeError('Candidate generator metadata does not match active data')
    base_iteration=(number-1) if base_iteration is None and number>1 else base_iteration
    if base_iteration is not None and not 1<=base_iteration<number:
        raise ValueError('Candidate base must be a completed earlier iteration')
    norm=fit_normalizer()
    source_files=['models.py','train.py','dataset.py','evaluate.py','audit.py','protocol.py','run.py']
    candidate_seal=dict(code={name:digest(ROOT/name) for name in source_files},
        audit=digest(ROOT/'results/full_audit.json'),normalizer=digest(ROOT/'data/normalizer.json'),
        split=digest(ROOT/'data/split_seal.json'),config=config,base_iteration=base_iteration)
    out=ROOT/'runs'/f'iteration_{number:02d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Candidate already evaluated')
    if (out/'seal.json').exists():
        if json.loads((out/'seal.json').read_text())!=candidate_seal:
            raise RuntimeError('Interrupted candidate code/data/config changed; cannot mix seed runs')
    else:write_new(out/'seal.json',candidate_seal)
    if not (out/'config.json').exists():
        if number>1:
            previous={**DEFAULT,**json.loads((ROOT/'runs'/f'iteration_{base_iteration:02d}'/'config.json').read_text())['config']}
            differences=[k for k in config if config[k]!=previous[k]]
            if len(differences)!=1:raise ValueError(f'Change exactly one thing: {differences}')
        write_new(out/'config.json',dict(iteration=number,config=config,hypothesis=hypothesis,change=change,base_iteration=base_iteration))
        with (ROOT/'LOOP_LOG.md').open('a') as f:
            f.write(f'\n## Iteration {number}\n\nHypothesis: {hypothesis}\n\nChange: {change}\n\nBase: {base_iteration}. Status: running; validation only.\n')
    else:
        saved=json.loads((out/'config.json').read_text())
        if saved['config']!=config:raise RuntimeError('Interrupted candidate configuration changed')
    start=time.monotonic()
    trainpack=load_pack('train',range(8),norm,device,purpose='train')
    fitpack=load_pack('train',range(5),norm,device,purpose='validation_fit')
    valpack=load_pack('train',[8,9],norm,device,purpose='validation')
    cache=ROOT/'runs/checkpoints';cache.mkdir(parents=True,exist_ok=True)
    runs=[]
    for seed in SEEDS:
        seed_result=out/f'seed_{seed}.json'
        if seed_result.exists():
            runs.append(json.loads(seed_result.read_text()));continue
        record=dict(seed=seed,metrics={},checkpoints={},training={},fit_log={})
        for kind in KINDS:
            key=key_for(kind,config,seed);path=cache/f'{kind}_{key}.pt'
            if path.exists():
                model,checkpoint=load_model(path,device);log=checkpoint['log'];reused=True
            else:
                model,log=train_model(kind,trainpack,norm,config,seed,path);reused=False
            record['checkpoints'][kind]=str(path.relative_to(ROOT))
            record['training'][kind]=dict(log=log,reused_identical_checkpoint=reused,sha256=digest(path))
            if kind!='two_part':record['metrics'][kind]=evaluate(model,valpack,norm)
            else:
                scores,fitted=score_two_part(model,fitpack,valpack,norm,config,seed,out,True)
                record['metrics'].update(scores);record['fit_log']=fitted['log']
                record['z_property_probe']=z_probe(fitted['z'],fitted['cube_ids'],seed)
            print(json.dumps({'iteration':number,'seed':seed,'kind':kind,'elapsed_seconds':time.monotonic()-start,
                'metrics':{k:{m:v for m,v in value.items() if m in ['one_step_nmse','r2','rollout_error']} for k,value in record['metrics'].items()}}),flush=True)
            del model
        write_new(seed_result,safe(record));runs.append(record)
    names=list(runs[0]['metrics'])
    summary={name:aggregate_seed_metrics([r['metrics'][name] for r in runs]) for name in names}
    f4=(summary['fitted_z']['one_step_nmse']['mean']+summary['fitted_z']['one_step_nmse']['std'] < summary['shuffled_z']['one_step_nmse']['mean']-summary['shuffled_z']['one_step_nmse']['std'] and summary['fitted_z']['one_step_nmse']['mean'] < summary['mean_z']['one_step_nmse']['mean'])
    f5=dict(pass_check=True,comparisons=[],interpretation='Conditional dimension-growth diagnostic; familiar-cube validation proxy, not unseen-cube proof.')
    for prev in sorted((ROOT/'runs').glob('iteration_*/result.json')):
        old=json.loads(prev.read_text())
        if old['config']['z_dim']<config['z_dim']:
            old_config={**DEFAULT,**old['config']}
            changed=[k for k in config if old_config[k]!=config[k]]
            if changed==['z_dim']:
                previous_learned=old['summary']['learned_z']['one_step_nmse']['mean']
                previous_fitted=old['summary']['fitted_z']['one_step_nmse']['mean']
                current_learned=summary['learned_z']['one_step_nmse']['mean']
                current_fitted=summary['fitted_z']['one_step_nmse']['mean']
                if any(v is None or not np.isfinite(v) for v in [previous_learned,previous_fitted,current_learned,current_fitted]):
                    f5['comparisons'].append(dict(iteration=old['iteration'],undefined=True))
                    f5['pass_check']=False
                    continue
                better=current_learned<previous_learned
                worse=current_fitted>previous_fitted
                f5['comparisons'].append(dict(iteration=old['iteration'],learned_improved=better,fitted_worsened=worse))
                f5['pass_check']=f5['pass_check'] and not(better and worse)
    accesses=[json.loads(line) for line in (ROOT/'results/data_access.jsonl').read_text().splitlines()]
    checks={'F1':audit['f1']['pass'] and static['pass'],'F2':audit['f2_pass'],
        'F3':all(a['group']=='train' for a in accesses),'F4':bool(f4),'F5':f5['pass_check'],
        'F6':all(k in summary for k in ['plain','oracle','mean_z']),
        'F7':len(runs)==5 and all(len(summary[k]['one_step_nmse']['values'])==5 and not summary[k]['nonfinite_metrics'] for k in summary),
        'F8':audit['f8_pass'] and all('z_property_probe' in r for r in runs),'F9':True}
    criteria=success_criteria(summary['plain'],summary['history'],summary['oracle'],summary['fitted_z'],summary['shuffled_z'],failure_checks=checks)
    gain=summary['mean_z']['one_step_nmse']['mean']-summary['fitted_z']['one_step_nmse']['mean']
    paired_gains=[];paired_recovery=[]
    for run in runs:
        m=run['metrics'];p=m['plain']['one_step_nmse'];o=m['oracle']['one_step_nmse'];f=m['fitted_z']['one_step_nmse']
        paired_gains.append(m['mean_z']['one_step_nmse']-f)
        paired_recovery.append((p-f)/(p-o) if p>o else np.nan)
    result=dict(iteration=number,config=config,hypothesis=hypothesis,change=change,base_iteration=base_iteration,summary=summary,
        failure_checks=checks,f5=f5,success=criteria,mean_z_absolute_gain=gain,
        paired_mean_z_gain=summarize(paired_gains),paired_gap_recovery=summarize(paired_recovery),
        z_recovery={p:summarize([r['z_property_probe'][p] for r in runs]) for p in ['mass_r2','friction_r2']},
        signal_probe=audit['probes']['state_action']['r2'],audit_reference=audit['saved_run'],
        audit_sha256=digest(ROOT/'results/full_audit.json'),runs=runs,elapsed_seconds=time.monotonic()-start)
    write_new(out/'result.json',safe(result))
    append(ROOT/'results/validation_evaluations.jsonl',dict(iteration=number,result=str((out/'result.json').relative_to(ROOT)),sha256=digest(out/'result.json')))
    with (ROOT/'LOOP_LOG.md').open('a') as f:
        f.write('\nResults: '+json.dumps(safe(dict(summary={k:v['one_step_nmse'] for k,v in summary.items()},checks=checks,success=criteria,mean_z_absolute_gain=gain)))+'\n')
    print(json.dumps(safe({'iteration':number,'summary':summary,'checks':checks,'success':criteria})),flush=True)
    return result


def final(number,device):
    audit,_=require_audit();verify()
    source=ROOT/'runs'/f'iteration_{number:02d}'/'result.json'
    chosen=json.loads(source.read_text());config=chosen['config']
    if chosen['audit_sha256']!=digest(ROOT/'results/full_audit.json'):
        raise RuntimeError('Selected final candidate was trained on a different data version')
    early_success=all(chosen['success'].get(f'S{i}') is True for i in range(1,5))
    completed=list((ROOT/'runs').glob('iteration_*/result.json'))
    if not early_success and len(completed)!=PROTOCOL['iteration_budget']:
        raise RuntimeError('Final evaluation requires validation success or all20 completed iterations')
    ledger=ROOT/'results/final_evaluations.jsonl'
    if ledger.exists():raise RuntimeError('Final evaluation was already invoked; do not retry or retune')
    from clean.recovery import prepare_recovery,score_recovery
    norm=fit_normalizer()
    prepare_recovery(source.parent,norm,device)
    candidate_id=f'iteration_{number:02d}_{digest(source)[:12]}'
    out=ROOT/'results/final';out.mkdir(parents=True,exist_ok=True)
    seal=dict(candidate=candidate_id,iteration=number,config=config,result_sha256=digest(source),
              checkpoints={str(r['seed']):{k:dict(path=p,sha256=digest(ROOT/p)) for k,p in r['checkpoints'].items()} for r in chosen['runs']},
              recovery_sha256={str(p.relative_to(ROOT)):digest(p) for p in (source.parent/'recovery').glob('*') if p.is_file()},
              code_sha256={p.name:digest(p) for p in ROOT.glob('*.py')})
    write_new(out/'seal.json',seal)
    append(ledger,dict(event='begin',candidate=candidate_id,seal_sha256=digest(out/'seal.json'),timestamp=time.time()))
    groups={}
    for group in ['test','out_of_range']:
        fitpack=load_pack(group,range(5),norm,device,purpose='final_fit',final_candidate=candidate_id)
        scorepack=load_pack(group,range(5,10),norm,device,purpose='final_score',final_candidate=candidate_id)
        runs=[]
        for run in chosen['runs']:
            seed=run['seed'];record=dict(seed=seed,metrics={})
            for path,expected in seal['recovery_sha256'].items():
                if digest(ROOT/path)!=expected:raise RuntimeError('Sealed recovery diagnostic changed')
            for kind in KINDS:
                path=ROOT/run['checkpoints'][kind]
                if digest(path)!=run['training'][kind]['sha256']:raise RuntimeError('Checkpoint changed')
                model,_=load_model(path,device)
                if kind!='two_part':record['metrics'][kind]=evaluate(model,scorepack,norm)
                else:
                    scores,fitted=score_two_part(model,fitpack,scorepack,norm,config,seed,out,False)
                    record['metrics'].update(scores);record['fit_log']=fitted['log']
                    record['property_recovery']=score_recovery(source.parent,seed,fitpack,fitted,norm)
                del model
            write_new(out/f'{group}_seed_{seed}.json',safe(record));runs.append(record)
            print(json.dumps(safe(dict(group=group,seed=seed,metrics={k:{m:v[m] for m in ['one_step_nmse','r2','rollout_error']} for k,v in record['metrics'].items()}))),flush=True)
        summary={name:aggregate_seed_metrics([r['metrics'][name] for r in runs]) for name in runs[0]['metrics']}
        final_checks=dict(chosen['failure_checks'])
        final_checks['F7']=bool(final_checks['F7'] and len(runs)==5 and all(not value['nonfinite_metrics'] for value in summary.values()))
        final_checks['F3']=bool(final_checks['F3'] and all(r['fit_log']['all_model_parameters_frozen'] and r['fit_log']['parameter_sha256_before']==r['fit_log']['parameter_sha256_after'] for r in runs))
        criteria=success_criteria(summary['plain'],summary['history'],summary['oracle'],summary['fitted_z'],summary['shuffled_z'],failure_checks=final_checks)
        gains=[];recovery=[]
        for run in runs:
            m=run['metrics'];p=m['plain']['one_step_nmse'];o=m['oracle']['one_step_nmse'];f=m['fitted_z']['one_step_nmse']
            gains.append(m['mean_z']['one_step_nmse']-f)
            recovery.append((p-f)/(p-o) if p>o else np.nan)
        property_recovery={method:{prop:summarize([r['property_recovery'][method][prop] for r in runs])
            for prop in ['mass_kg','friction']} for method in ['gru','z']}
        groups[group]=dict(summary=summary,success=criteria,runs=runs,property_recovery=property_recovery,failure_checks=final_checks,
            paired_mean_z_gain=summarize(gains),paired_gap_recovery=summarize(recovery))
        del fitpack,scorepack
    result=dict(candidate=candidate_id,iteration=number,config=config,groups=groups,validation=chosen['summary'],
                failure_checks=groups['test']['failure_checks'],signal_probe=chosen['signal_probe'],z_recovery=chosen['z_recovery'])
    write_new(out/'result.json',safe(result))
    append(ledger,dict(event='complete',candidate=candidate_id,result_sha256=digest(out/'result.json'),timestamp=time.time()))
    print(json.dumps(safe({g:groups[g]['success'] for g in groups})),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['candidate','final'])
    parser.add_argument('--iteration',type=int,required=True);parser.add_argument('--config')
    parser.add_argument('--base-iteration',type=int)
    parser.add_argument('--hypothesis',default='Initial capacity and a small object latent may explain object-specific dynamics.')
    parser.add_argument('--change',default='Initial locked reference candidate, d=2.')
    parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(2)
    if args.mode=='candidate':
        cfg=copy.deepcopy(DEFAULT)
        if args.config:cfg.update(json.loads(Path(args.config).read_text()))
        candidate(args.iteration,cfg,args.hypothesis,args.change,args.device,args.base_iteration)
    else:final(args.iteration,args.device)


if __name__=='__main__':main()
