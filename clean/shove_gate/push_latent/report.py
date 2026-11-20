"""Report saved Step 2 runs only. Never trains, adapts, or predicts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SEEDS=[101,211,307,401,503]
VARIANTS=['plain','oracle','history','z2','z4','z8','z16']
MODEL_ORDER=['plain','history','oracle','fitted_z','mean_z','wrong_z']
LABELS={'plain':'Plain','oracle':'Oracle','history':'History (GRU 64)',
        'fitted_z':'Two-part: fitted z (5 shoves)','mean_z':'Two-part: mean z',
        'wrong_z':"Two-part: another cube’s z"}
COLORS={'plain':'#256b9b','oracle':'#b35418','history':'#428a52','fitted_z':'#7452a2',
        'mean_z':'#87733c','wrong_z':'#b34a75','z2':'#7452a2','z4':'#87733c','z8':'#b34a75','z16':'#3299a6'}
GROUPS=['push','moving_coast','all_steps']
GROUP_LABELS={'push':'Push','moving_coast':'Moving coast','all_steps':'All retained steps'}
PRIMARY='horizontal_velocity_delta_nmse'
METRICS=[PRIMARY,'one_step_nmse','rollout10_nmse','rollout10_final_nmse','stopping_position_error_m']
METRIC_LABELS={PRIMARY:'Horizontal velocity-change NMSE','one_step_nmse':'Full-state one-step NMSE',
              'rollout10_nmse':'Mean ten-step rollout NMSE','rollout10_final_nmse':'Tenth-step-only NMSE',
              'stopping_position_error_m':'Stopping-position error (m)'}

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fmt(value,digits=6): return 'not evaluated' if value is None else f'{float(value):.{digits}g}'
def moment(row): return 'not evaluated' if row is None else f'{fmt(row["mean"])} ± {fmt(row["sample_sd"],3)}'
def pct(value): return 'not evaluated' if value is None else f'{100*value:+.2f}%'
def link(root,name,label): return f'[{label}]({root/name})'
def require(condition,message):
    if not condition: raise RuntimeError(message)
def close(a,b): return np.isclose(a,b,rtol=1e-11,atol=1e-13)
def save(fig,path):
    for extension in ['png','svg']: fig.savefig(path.with_suffix('.'+extension),dpi=180,bbox_inches='tight')
    plt.close(fig)

def inputs(root):
    results=root/'results';ledger=read(results/'final_evaluation.json') if (results/'final_evaluation.json').exists() else None
    stop=read(results/'STOP.json') if (results/'STOP.json').exists() else None
    require((ledger and ledger.get('status') in ['complete','c1_failed','stopped']) or stop,
            'Reporting waits for a completed or explicitly stopped evaluation')
    summary=read(results/'summary.json') if (results/'summary.json').exists() else None
    if ledger and ledger.get('status')=='complete':
        require(summary is not None and ledger['summary_sha256']==sha(results/'summary.json'),'Final summary ledger mismatch')
        seal=read(results/'checkpoint_seal.json')
        require(len(seal['checkpoints'])==35,'Expected all 35 frozen candidate checkpoints')
        for name,expected in seal['checkpoints'].items():
            require(sha(root/'checkpoints'/name)==expected,'Frozen checkpoint bytes changed')
        require(summary['primary_metric']==PRIMARY and summary['primary_group']=='push','Wrong primary endpoint')
        require(summary['test_cubes']==list(range(260,290)) and summary['query_episodes']==list(range(5,10)), 'Wrong fresh query population')
        for name,groups in summary['by_model'].items():
            for group in GROUPS:
                for metric in METRICS:
                    row=groups[group][metric];values=[]
                    for seed in SEEDS:
                        p=results/'test_metrics'/f'{name}_seed_{seed}.json';record=read(p)
                        require(record['checkpoint_sha256']==seal['checkpoints'][f'{record["variant"]}_seed_{seed}.pt'], 'Scored checkpoint differs from frozen checkpoint')
                        value=record['groups'][group][metric]
                        require(value['cubes']==30 and value['episodes']==150,'Incomplete query coverage')
                        values.append(value['mean'])
                    require(values==row['seed_values'],'Summary seed values disagree')
                    require(close(np.mean(values),row['mean']) and close(np.std(values,ddof=1),row['sample_sd']), 'Summary moments disagree')
        p=summary['by_model']['plain']['push'][PRIMARY];o=summary['by_model']['oracle']['push'][PRIMARY]
        c1=(o['mean']<=.8*p['mean'] and o['mean']+o['sample_sd']<p['mean']-p['sample_sd'])
        require(c1==summary['c1']['passed'],'C1 arithmetic disagrees')
        if c1:
            z=summary['by_model']['fitted_z']['push'][PRIMARY];mean=summary['by_model']['mean_z']['push'][PRIMARY];wrong=summary['by_model']['wrong_z']['push'][PRIMARY]
            recovery=(p['mean']-z['mean'])/(p['mean']-o['mean'])
            require(close(recovery,summary['c2']['recovered_fraction']) and (recovery>=.5)==summary['c2']['passed'],'C2 arithmetic disagrees')
            c3=(z['mean']+z['sample_sd']<mean['mean']-mean['sample_sd'] and z['mean']+z['sample_sd']<wrong['mean']-wrong['sample_sd'])
            require(c3==summary['c3']['passed'],'C3 arithmetic disagrees')
        else:
            require(summary['c2'] is None and summary['c3'] is None and summary['prefix_curve'] is None, 'C1 failure did not stop Stage B')
    values={}
    for key,name in [('full','full_audit.json'),('timing','collection_timing.json'),('normalizer','normalizer.json'),
                     ('tuning','latent_tuning.json'),('selected','selected_config.json'),('model_protocol','model_protocol.json')]:
        path=results/name
        values[key]=read(path) if path.exists() else None
    training={};groups={}
    for seed in SEEDS:
        group=results/'training_logs'/f'group_seed_{seed}.json'
        if group.exists():groups[seed]=read(group)
        for variant in VARIANTS:
            p=results/'training_logs'/f'{variant}_seed_{seed}.json'
            if p.exists():
                r=read(p);best=min(r['log'],key=lambda row:row['validation_nmse'])
                require(best['update']==r['selected_update'] and close(best['validation_nmse'],r['selected_validation_nmse']), 'Selected checkpoint is not validation minimum')
                training[variant,seed]=r
    paths={root/'PROTOCOL.md',root/'DATA_REPORT.md'}
    for name in ['full_audit.json','collection_timing.json','normalizer.json','model_protocol.json','checkpoint_seal.json',
                 'test_data_seal.json','adaptation_seal.json','selected_config.json','latent_tuning.json','mass_readouts.json',
                 'summary.json','c1.json','final_evaluation.json','STOP.json','action_probe.json','final_audit.json',
                 'pretraining_review.json','pretraining_scale_audit.json','pretest_audit.json']:
        if (results/name).exists():paths.add(results/name)
    for folder in ['training_logs','test_metrics','latent_validation_runs']:
        paths.update((results/folder).glob('*.json'))
    for pattern in ['prefix_scores_seed_*.json','mass_scores_seed_*.json']:
        paths.update(results.glob(pattern))
    return ledger,stop,summary,values,training,groups,sorted(paths)

def plot_primary(root,summary):
    names=[n for n in MODEL_ORDER if n in summary['by_model']]
    fig,ax=plt.subplots(figsize=(10,5.7));offsets=np.linspace(-.08,.08,5)
    for x,name in enumerate(names):
        row=summary['by_model'][name]['push'][PRIMARY]
        ax.scatter(x+offsets,row['seed_values'],color=COLORS[name],s=32,zorder=3)
        ax.errorbar(x,row['mean'],yerr=row['sample_sd'],fmt='D',color='black',capsize=5,zorder=4)
    ax.set_xticks(range(len(names)),[LABELS[n].replace('Two-part: ','') for n in names],rotation=12,ha='right')
    ax.set_ylabel('Push horizontal velocity-change NMSE');ax.grid(axis='y',alpha=.2)
    ax.set_ylim(bottom=0);ax.set_title('Fresh test cubes: shoves 5–9, push steps')
    fig.text(.5,.01,'Points: five saved paired seeds. Black diamonds: mean ± sample SD. Lower is better.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));save(fig,root/'results/figures/primary_comparison')

def plot_training(root,training,groups):
    if not training:return
    fig,axes=plt.subplots(3,2,figsize=(13,11));axes=axes.ravel()
    for ax,seed in zip(axes,SEEDS):
        for v in VARIANTS:
            if (v,seed) not in training:continue
            row=training[v,seed];log=row['log']
            ax.plot([r['update'] for r in log],[r['validation_nmse'] for r in log],color=COLORS[v],lw=1,label=LABELS.get(v,v))
            ax.scatter(row['selected_update'],row['selected_validation_nmse'],color=COLORS[v],marker='*',s=70,edgecolor='black',linewidth=.4,zorder=3)
        if seed in groups:
            for d in groups[seed]['extension_decisions'][:-1]:ax.axvline(d['endpoint'],ls=':',color='#aaa',lw=.7)
        budget=f"{groups[seed]['completed_updates']:,}" if seed in groups else 'incomplete'
        ax.set_title(f'Seed {seed}: {budget} updates / variant');ax.set_yscale('log')
        ax.set_xlabel('Training updates');ax.set_ylabel('Checkpoint-validation push NMSE (log)');ax.grid(alpha=.2)
    axes[-1].axis('off');handles,labels=axes[0].get_legend_handles_labels();axes[-1].legend(handles,labels,loc='upper left',fontsize=10,ncol=2)
    axes[-1].text(.03,.5,'Every saved 500-update validation check.\nStars mark selected checkpoints.\nDotted lines mark extension blocks.\n\nAll seven variants share each seed’s budget\nand minibatch stream.\n\nValidation shoves 8–9 of fitting cubes only.\nThe 20 latent-validation cubes do not select\ncheckpoints or training budgets.',va='top',fontsize=11,linespacing=1.5)
    fig.suptitle('Checkpoint validation: push horizontal velocity change',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.96));save(fig,root/'results/figures/validation_curves')

def plot_tuning(root,tuning,selected):
    if not tuning:return
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for ax,dim in zip(axes.ravel(),[2,4,8,16]):
        for lr,color in zip([.01,.03,.1],['#256b9b','#b35418','#428a52']):
            rows=sorted([r for r in tuning['candidates'] if r['z_dim']==dim and r['learning_rate']==lr],key=lambda r:r['updates'])
            ax.errorbar([r['updates'] for r in rows],[r['validation_mean_nmse'] for r in rows],
                        yerr=[r['validation_seed_sample_sd'] for r in rows],color=color,label=f'lr {lr:g}',marker='o',capsize=3)
        if selected and selected['z_dim']==dim:ax.scatter(selected['updates'],selected['validation_mean_nmse'],marker='*',s=200,color='black',zorder=5)
        ax.set_title(f'z dimension {dim}');ax.set_xlabel('z-fitting updates');ax.set_ylabel('Validation query push NMSE')
        ax.set_xticks([50,100,200,400]);ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Latent-validation cubes only: all preregistered fitting choices',fontsize=14)
    fig.text(.5,.015,'Mean ± sample SD across five seeds; each score averages 20 validation cubes. Black star: one global selected choice.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95));save(fig,root/'results/figures/latent_validation_grid')

def plot_prefix(root,summary):
    if summary.get('prefix_curve') is None:return
    fig,ax=plt.subplots(figsize=(8,5.7));rows=[summary['prefix_curve'][str(k)] for k in range(1,6)]
    ax.errorbar(range(1,6),[r['mean'] for r in rows],yerr=[r['sample_sd'] for r in rows],color=COLORS['fitted_z'],marker='o',capsize=4,label='Fitted z')
    for name in ['plain','history','oracle','mean_z','wrong_z']:
        row=summary['by_model'][name]['push'][PRIMARY]
        ax.axhline(row['mean'],color=COLORS[name],ls='--',lw=1,label=LABELS[name])
    ax.set_xticks(range(1,6));ax.set_xlabel('Completed support shoves used to fit z')
    ax.set_ylabel('Query push horizontal velocity-change NMSE');ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax.set_title('Same fresh query shoves 5–9 for every support prefix')
    fig.text(.5,.015,'Each prefix restarts at training-mean z with the same frozen fitting settings. Error bars: seed SD.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));save(fig,root/'results/figures/support_prefix')

def generate(root=ROOT):
    root=Path(root).resolve();ledger,stop,summary,v,training,groups,paths=inputs(root)
    (root/'results/figures').mkdir(exist_ok=True)
    plot_training(root,training,groups);plot_tuning(root,v['tuning'],v['selected'])
    complete=bool(ledger and ledger.get('status')=='complete' and summary is not None)
    lines=['# Step 2: push confirmation and object-latent adaptation','']
    if not complete:
        reason=(stop or {}).get('reason',(ledger or {}).get('failure',{}).get('reason','No completed frozen summary'))
        lines += ['**STOPPED before a complete final comparison.** '+reason,'',
                  'No missing score is inferred. Any saved partial results remain evidence only; they do not establish a five-seed gate. The protocol requires stopping on this failure, with no automatic repair, retuning, resampling, or replacement run.','']
        if (root/'results/c1.json').exists():
            c1=read(root/'results/c1.json');lines += [f'C1 had been recorded as **{"PASS" if c1["passed"] else "FAIL"}** before this stop. C2/C3 are not completed by this report.','']
    else:
        plot_primary(root,summary);plot_prefix(root,summary)
        c1=summary['c1'];c2=summary['c2'];c3=summary['c3']
        outcomes=[f'C1 **{"PASS" if c1["passed"] else "FAIL"}**']
        outcomes += [f'C{i} **{"not evaluated" if c is None else "PASS" if c["passed"] else "FAIL"}**' for i,c in [(2,c2),(3,c3)]]
        overall=('All preregistered success criteria are met.' if summary['all_applicable_criteria_passed'] else
                 'The overall preregistered success criteria are not met: '+('C1 failed.' if not c1['passed'] else 'C2 failed.' if not c2['passed'] else 'C3 failed.'))
        lines += [', '.join(outcomes)+'.','', '**'+overall+'**','',
                  'Primary population: push steps of shoves 5–9 on the 30 fresh cubes 260–289. Mean ± sample SD is across five paired seeds. Lower error is better.','',
                  '| Model / control | Push horizontal-delta NMSE | Improvement vs Plain | Beyond seed spread? |',
                  '|---|---:|---:|---|']
        for name in MODEL_ORDER:
            if name not in summary['by_model']:
                lines.append(f'| {LABELS[name]} | Not evaluated after C1 failure | — | — |');continue
            row=summary['by_model'][name]['push'][PRIMARY];comparison=summary['comparisons'].get(name,{}).get('push',{}).get(PRIMARY)
            lines.append(f'| {LABELS[name]} | {moment(row)} | {"reference" if comparison is None else pct(comparison["relative_mean_improvement"])} | {"—" if comparison is None else "yes" if comparison["beyond_seed_spread"] else "no"} |')
        lines += ['', '| Model / control | Raw horizontal Δv MSE ((m/s)²) | Pooled-coordinate Δv RMSE (m/s) |',
                  '|---|---:|---:|']
        scale=v['normalizer']['horizontal_velocity_delta_mse_scale']
        for name in MODEL_ORDER:
            if name in summary['by_model']:
                raw_mse=summary['by_model'][name]['push'][PRIMARY]['mean']*scale
                lines.append(f'| {LABELS[name]} | {fmt(raw_mse,8)} | {fmt(np.sqrt(raw_mse),8)} |')
        lines += ['', 'Raw MSE equals the saved five-seed mean NMSE times the frozen fitting push-energy scale. '
                  'The pooled-coordinate RMSE is the square root of that MSE, pooling x/y coordinates and all five seeds after the stated cube/episode averaging.','']
        lines += ['',f'![Primary comparison]({root}/results/figures/primary_comparison.png)','',
            f'C1 requires ≥20% Oracle improvement and `Oracle mean+SD < Plain mean−SD`. Measured improvement is **{pct(c1["relative_mean_improvement"])}**, '
            f'with mean gap **{fmt(c1["mean_gap"])}** versus summed seed SD **{fmt(c1["sum_seed_sds"])}**.','']
        if not c1['passed']:
            lines += ['C1 failed. Test z fitting, prefix scoring, mean/wrong-z controls, and the test mass readout were not performed. History is reported alongside the two baselines.','']
        else:
            lines += [f'C2 requires fitted z to recover at least 50% of the Plain-to-Oracle gap. It recovers **{100*c2["recovered_fraction"]:.2f}%**.','',
                      f'History recovers **{100*(summary["by_model"]["plain"]["push"][PRIMARY]["mean"]-summary["by_model"]["history"]["push"][PRIMARY]["mean"])/c2["plain_to_oracle_gap"]:.2f}%** of the same gap as a descriptive comparison; History does not decide C2 or C3.','',
                      '| C3 comparison | Mean gap favoring fitted z | Sum of seed SDs | Beyond spread? |','|---|---:|---:|---|']
            for key,label in [('fitted_vs_mean_z','Mean z → fitted z'),('fitted_vs_wrong_z',"Another cube’s z → fitted z")]:
                row=c3[key];lines.append(f'| {label} | {fmt(row["mean_gap"])} | {fmt(row["sum_seed_sds"])} | {"yes" if row["beyond_seed_spread"] else "no"} |')
            h=summary['by_model']['history']['push'][PRIMARY];z=summary['by_model']['fitted_z']['push'][PRIMARY]
            history_gap=h['mean']-z['mean'];history_spread=h['sample_sd']+z['sample_sd']
            lines += ['',f'For fitted z versus History, the mean gap favoring fitted z is **{fmt(history_gap)}**, compared with combined seed SD **{fmt(history_spread)}**. '
                      + ('The gap exceeds that spread.' if history_gap>history_spread else 'The gap does not exceed that spread.')]
            lines += ['', 'The mean-z and wrong-z spread comparisons must both pass C3. No other metric or support prefix can rescue a failed criterion.','',
                      '| Support shoves | Query push NMSE |','|---:|---:|']
            for k in range(1,6):lines.append(f'| {k} | {moment(summary["prefix_curve"][str(k)])} |')
            lines += ['',f'![Support prefix curve]({root}/results/figures/support_prefix.png)','',
                'Each prefix independently restarts at training-mean z and uses identical selected fitting settings. All query scores use shoves 5–9. The wrong-z control uses the fixed cyclic next-cube mapping, never a similarity or performance match.','',
                '## Affine mass readout','',
                'An affine least-squares map from support-refitted z to true mass was fitted on 180 weight-training cubes only, then frozen. The 20 latent-validation cubes did not fit or select it. These scores do not decide any criterion.','',
                '| Diagnostic | R² | MAE (kg) | RMSE (kg) |','|---|---:|---:|---:|']
            for key,label in [('mass_readout','Fitted-z affine decoder'),('mass_baseline','Training-mean mass')]:
                row=summary[key];lines.append(f'| {label} | {moment(row["r2"])} | {moment(row["mae_kg"])} | {moment(row["rmse_kg"])} |')
            lines += ['']
    lines += ['## Protocol, data, and separation','',
        link(root,'PROTOCOL.md','The written protocol and reasons')+' preceded all Step 2 collection, fitting, tuning, and scoring. The prior 56% push benefit was secondary; fresh cubes independently confirm or reject it as a predeclared primary endpoint. The previous test sets/results remain untouched.','',
        'IDs 0–179 fit weights on shoves 0–7; shoves 8–9 select checkpoints and budgets. IDs 180–199 are reserved solely for latent dimension and fitting choices. Fresh IDs 260–289 use seed 2026111519; IDs 200–259 are retired. Original training traces are byte-identical.','']
    if v['full']:
        f=v['full']['fresh_test_summary'];lines += [f'Fresh collection: **{f["launches"]}/{f["episodes"]} launches**, **{f["moving_coast_fraction"]:.2%}** moving coast after trimming, maximum coast **{f["maximum_coast_distance_m"]:.6f} m**; **{f["transitions"]:,}** retained and **{f["trimmed_transitions"]:,}** stationary transitions trimmed. All episodes reached rest, no moving transition was trimmed, and no sample was rejected or replaced. The integrity audit preserved **{v["full"]["retired_artifacts_unchanged"]:,}** retired artifacts.','']
    if v['normalizer']:
        scale=v['normalizer']['horizontal_velocity_delta_mse_scale'];lines += [f'Primary normalization is fitting-only push change energy **{scale:.12g} (m/s)²**, averaged equally by cube, episode, and push step. A zero-change predictor scores 1 on that fitting subset. All primary/query scores use this same scalar.','']
    lines += ['State/action inputs exclude IDs, phase, lengths, masks, stop times, termination indicators, and future observations. True properties enter only Oracle and the separate mass-readout diagnostic. History uses ten strictly preceding transitions from the current episode, with fixed initial-state/zero-action/zero-delta padding and no mask. Recursive History uses predicted transitions after the forecast start.','',
        '## Training and validation-only choices','',
        'All seven variants retain the full 15 normalized state-change training loss and three 256-unit ReLU hidden layers. History adds GRU 64. The four two-part candidates use zero-initialized per-training-cube embeddings of dimension 2,4,8,16 with 1e-4 L2. No mass/friction label enters a two-part model.','',
        'Every seed shares minibatches and update budgets across all seven variants. Checkpoints minimize fitting-cube held-out-shove push NMSE. The predeclared final 2,500-update improvement rule extends every variant by 10,000 updates if any variant improves its previous best by ≥0.1%, with a 100,000-update resource cap.','']
    if groups:
        lines += ['| Seed | Updates per variant | Still improving at cap? |','|---:|---:|---|']
        for seed in SEEDS:
            if seed in groups:lines.append(f'| {seed} | {groups[seed]["completed_updates"]:,} | {"yes" if groups[seed]["convergence_not_established_at_cap"] else "no"} |')
        lines += ['']
    if training:
        lines += [f'![All checkpoint-validation curves]({root}/results/figures/validation_curves.png)','',
                  '| Seed | Variant | Selected update | Selected validation push NMSE |','|---:|---|---:|---:|']
        for seed in SEEDS:
            for variant in VARIANTS:
                if (variant,seed) in training:
                    r=training[variant,seed];lines.append(f'| {seed} | {variant} | {r["selected_update"]:,} | {fmt(r["selected_validation_nmse"])} |')
        lines += ['']
    if v['selected']:
        c=v['selected'];lines += [f'The 20 latent-validation cubes selected one global configuration: **z dimension {c["z_dim"]}, learning rate {c["learning_rate"]:g}, {c["updates"]} updates**. Its validation query mean NMSE is **{fmt(c["validation_mean_nmse"])}**. Selection used support shoves 0–4 and query shoves 5–9 across all five seeds; no test result or true physical property selected it.','',
            f'![Latent validation grid]({root}/results/figures/latent_validation_grid.png)','']
    if complete:
        lines += ['## Secondary prediction metrics','',
            'These scores decide no criterion. All groups use query shoves 5–9. Old state metrics retain full 15-coordinate normalization. Ten-step forecasts are recursive; stopping forecasts continue to planned release + 1.5 s, never an observed stop time or trimmed length. No clipping, physics correction, or observed-state reset is used.','',
            '| Group | Metric | Model | Mean ± seed SD |','|---|---|---|---:|']
        for group in GROUPS:
            for metric in METRICS:
                for name in MODEL_ORDER:
                    if name in summary['by_model']:lines.append(f'| {GROUP_LABELS[group]} | {METRIC_LABELS[metric]} | {LABELS[name]} | {moment(summary["by_model"][name][group][metric])} |')
        lines += ['', 'Individual stopping-position errors are retained below, including any large finite recursive failures; these are prediction errors, not distances traveled by real simulated cubes.','',
            '| Seed | Model | Push stopping error (m) | Moving-coast stopping error (m) | All-step stopping error (m) |','|---:|---|---:|---:|---:|']
        for i,seed in enumerate(SEEDS):
            for name in MODEL_ORDER:
                if name in summary['by_model']:
                    vals=[summary['by_model'][name][g]['stopping_position_error_m']['seed_values'][i] for g in GROUPS]
                    lines.append(f'| {seed} | {LABELS[name]} | '+ ' | '.join(fmt(x,9) for x in vals)+' |')
        large=sorted([(value,name,seed,group) for name in summary['by_model'] for group in GROUPS
                      for seed,value in zip(SEEDS,summary['by_model'][name][group]['stopping_position_error_m']['seed_values'])
                      if value>1.],reverse=True)
        if large:
            examples='; '.join(f'{LABELS[name]}, seed {seed}, {GROUP_LABELS[group].lower()}: {value:.6f} m'
                               for value,name,seed,group in large[:3])
            lines += ['', '**Large finite stopping-prediction errors remain included.** The largest seed/group means are '+examples+'. These recursive failures do not change C1–C3, and no affected seed was removed or replaced.','']
        lines += ['', '## Seed-level primary scores','', '| Seed | '+' | '.join(LABELS[n] for n in MODEL_ORDER if n in summary['by_model'])+' |',
                  '|---:|'+''.join('---:|' for n in MODEL_ORDER if n in summary['by_model'])]
        for i,seed in enumerate(SEEDS):lines.append(f'| {seed} | '+' | '.join(fmt(summary['by_model'][n]['push'][PRIMARY]['seed_values'][i],9) for n in MODEL_ORDER if n in summary['by_model'])+' |')
        lines += ['']
    probe=root/'results/action_probe.json'
    if probe.exists():
        d=read(probe);lines += ['## Actions-only independence diagnostic','',
            'Fresh commands were scored only after the frozen final-evaluation ledger opened. Fitting/tuning used only the 180 weight-training cubes, with an internal 162/18 split; the 20 latent-validation cubes were excluded.','',
            '| Probe | Mass R² ± seed SD | Combined-friction R² ± seed SD |','|---|---:|---:|']
        for name,row in d['generator']['aggregate'].items():lines.append(f'| {name} | {row["mass_kg"]["mean_r2"]:.6f} ± {row["mass_kg"]["seed_std_r2"]:.3g} | {row["combined_friction"]["mean_r2"]:.6f} ± {row["combined_friction"]["seed_std_r2"]:.3g} |')
        lines += ['']
    else:lines += ['The optional fresh actions-only probe has not been scored in this report; generator separation and data integrity were audited without predictive test tuning.','']
    lines += ['## Saved evidence','']
    for name,label in [('PROTOCOL.md','Written protocol and reasons'),('DATA_REPORT.md','Collection report'),('results/summary.json','Frozen final comparisons'),
        ('results/final_evaluation.json','Final evaluation ledger'),('results/full_audit.json','Data integrity audit'),('results/final_audit.json','Independent final audit'),
        ('results/pretraining_review.json','Source review and contract checks before fitting'),
        ('results/pretest_audit.json','Frozen-input audit before fresh predictive test scoring'),
        ('results/checkpoint_seal.json','All candidate checkpoint hashes'),('results/adaptation_seal.json','Frozen adaptation and mass-readout settings'),
        ('results/latent_tuning.json','Complete latent validation grid'),('results/retired_experiment_seal.json','Preserved retired artifacts'),('results/STOP.json','Recorded protocol stop')]:
        if (root/name).exists():lines.append('- '+link(root,name,label))
    lines += ['', 'Every plotted point is computed from saved run records. Error bars show sample SD across optimization seeds, not confidence intervals over all possible cubes. No report operation fits a model or reruns a prediction.','',
        'Rebuild after a completed or explicitly stopped evaluation:','', '```bash',
        'PYTHONPATH=clean/vendor:. MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove_gate.push_latent.report','```','']
    manifest={str(p.relative_to(root)):sha(p) for p in paths};manifest['report.py']=sha(Path(__file__))
    (root/'results/report_inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines += [link(root,'results/report_inputs.json','Report input SHA-256 manifest'),'']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    status='Stopped before complete comparison' if not complete else ('C1 failed; no test z fitting' if not summary['c1']['passed'] else f'C1 passed; C2 {"passed" if summary["c2"]["passed"] else "failed"}; C3 {"passed" if summary["c3"]["passed"] else "failed"}')
    (root/'README.md').write_text('# Step 2: push confirmation and object-latent adaptation\n\n**'+status+'.** See [RESULTS.md](RESULTS.md) for the saved scores, validation and prefix curves, mass readout, secondary forecasts, and limitations.\n\nThe [protocol and reasons](PROTOCOL.md) preceded all Step 2 runs. Original training traces remain byte-identical; IDs 200–259 are retired and unchanged. Fresh test IDs 260–289 use seed 2026111519.\n\nIDs 0–179 shoves 0–7 fit weights and normalization; shoves 8–9 choose checkpoints and budgets. IDs 180–199 only select latent dimension and z-fitting settings. Test support is shoves 0–4; query is shoves 5–9.\n\nNo test result changes a checkpoint, fitting choice, gate, or sample. A protocol STOP is final for this run. Report generation reads saved outputs only; it never simulates, trains, adapts, or scores.\n\nRebuild the report from `/home/ubuntu/faliure`:\n\n```bash\nPYTHONPATH=clean/vendor:. MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove_gate.push_latent.report\n```\n')
    print(json.dumps(dict(report=str(root/'RESULTS.md'),complete=complete,status=status,input_hashes=len(manifest))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    generate(parser.parse_args().root)
