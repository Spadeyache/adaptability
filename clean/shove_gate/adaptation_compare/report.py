"""Render Step 3 reports and scientific figures from saved artifacts only.

This module does not import simulation, models, adaptation, or evaluation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SEEDS=[101,211,307,401,503]
METRICS=['push_velocity_nmse','coast_velocity_nmse','full_state_nmse','composite']
LABELS=['Push velocity NMSE','Moving-coast velocity NMSE','Full-state NMSE','Composite']
RATES=['1e-05','1e-04']
SNAPSHOTS=[1,5,6,10,15,16,20,25,30]
COLORS={'A':'#7552a8','1e-05':'#2878a8','1e-04':'#b45317','history':'#398850','oracle':'#bb4c77','plain':'#777777'}

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fmt(value):return f'{value:.6g}'
def moment(value):return f"{fmt(value['mean'])} ± {fmt(value['sample_sd'])}"
def verdict(value):return 'PASS' if value else 'FAIL'
def link(path,label):return f'[{label}]({Path(path).resolve()})'
def write(path,text):Path(path).write_text(text.rstrip()+'\n')
def moments(values):
    a=np.asarray(values,dtype=float)
    if a.shape!=(5,) or not np.isfinite(a).all():raise ValueError('Need five finite saved seed scores')
    return dict(seed_values=a.tolist(),mean=float(a.mean()),sample_sd=float(a.std(ddof=1)))

def score_table(rows):
    lines=['| Model / observation | '+' | '.join(LABELS)+' |','|---|'+'---:|'*len(METRICS)]
    for name,values in rows:lines.append('| '+name+' | '+' | '.join(moment(values[m]) for m in METRICS)+' |')
    return lines

def validation_lines(selected,scales):
    a=selected['A'];b=selected['B']
    lines=['# Step 3: validation-only adaptation settings','',
        f"A selects **z dimension {a['z_dim']}, prior penalty {a['penalty']:g}**, Adam learning rate {a['learning_rate']:g}, {a['updates']} updates.",
        f"B selects **learning rate {b['learning_rate']:g}**, {b['updates']} full-support Adam updates. Both declared rates remain in the test comparison.",'',
        'Only cubes 180–199 selected these settings: support shoves 0–4, query shoves 5–9, mean across all 20 cubes and five seeds. No new test cube or retired test observation enters selection.','']
    rows=[('A selected',a['validation'])]+[(f"B lr {r['learning_rate']:g}"+(' (selected)' if r['learning_rate']==b['learning_rate'] else ''),r['validation']) for r in selected['b_rates']]
    lines+=score_table(rows)
    lines+=['','All three components have their own fitting-only zero-change energy scale. The composite is their unweighted average; lower is better. Scales use IDs 0–179, shoves 0–7, while frozen network input/output scalers are unchanged.','',
        '| Scale | Value |','|---|---:|',f"| Push horizontal Δv energy ((m/s)²) | {scales['push_velocity_energy']:.12g} |",
        f"| Moving-coast horizontal Δv energy ((m/s)²) | {scales['coast_velocity_energy']:.12g} |",
        f"| Full normalized-state Δ energy | {scales['full_state_energy']:.12g} |",'',
        'A minimizes full-state support loss plus the declared penalty toward training-mean z. B minimizes the identical full-state data term while updating all Plain weights. Every fit uses all retained support steps, with equal episode weighting, and exactly 50 updates.','',
        'A exact ties prefer smaller dimension, then larger penalty. B exact ties prefer smaller learning rate. No hyperparameter grid is extended after these choices.','',
        '| z dimension | Prior penalty | Validation composite ± seed SD |','|---:|---:|---:|']
    for c in selected['a_candidates']:lines.append(f"| {c['z_dim']} | {c['penalty']:g} | {moment(c['validation']['composite'])} |")
    lines+=['',link(ROOT/'PROTOCOL.md','Protocol and reasons'),'']
    return lines

def save_figure(fig,name):
    directory=ROOT/'results/figures';directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/(name+'.png'),dpi=170,bbox_inches='tight',pad_inches=.25)
    fig.savefig(directory/(name+'.svg'),bbox_inches='tight',pad_inches=.25);plt.close(fig)

def validation_plot(selected):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,metric,label in zip(axes.flat,METRICS,LABELS):
        for dim,color in zip([4,8,16],['#2878a8','#398850','#7552a8']):
            rows=sorted([c for c in selected['a_candidates'] if c['z_dim']==dim],key=lambda c:c['penalty'])
            ax.errorbar([r['penalty'] for r in rows],[r['validation'][metric]['mean'] for r in rows],
                yerr=[r['validation'][metric]['sample_sd'] for r in rows],marker='o',capsize=3,color=color,label=f'z dimension {dim}')
        chosen=selected['A'];ax.scatter(chosen['penalty'],chosen['validation'][metric]['mean'],s=160,marker='*',color='black',zorder=5)
        ax.set(xscale='log',xlabel='Prior penalty λ',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Adaptation-validation cubes 180–199 only: full-state z fitting\nError bars: sample SD across five seeds; black star: selected setting')
    save_figure(fig,'validation_penalty')

def render_validation():
    selected=read(ROOT/'results/selected_config.json');scales=read(ROOT/'results/metric_scales.json')
    lines=validation_lines(selected,scales)
    validation_plot(selected)
    lines+=['![Validation penalty grid]('+str(ROOT/'results/figures/validation_penalty.png')+')']
    write(ROOT/'VALIDATION_SELECTION.md','\n'.join(lines))

def draw_lines(ax,x,series,label,color):
    means=np.array([v['mean'] for v in series]);sds=np.array([v['sample_sd'] for v in series])
    ax.plot(x,means,'o-',label=label,color=color,markersize=4)
    ax.fill_between(x,means-sds,means+sds,color=color,alpha=.12)

def plots(summary):
    curves=summary['curves'];primary=summary['criteria']['primary_lr']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,metric,label in zip(axes.flat,METRICS,LABELS):
        draw_lines(ax,list(range(1,6)),[curves['A'][str(k)][metric] for k in range(1,6)],'A: fitted z',COLORS['A'])
        for rate in RATES:draw_lines(ax,list(range(1,6)),[curves['B'][rate][str(k)][metric] for k in range(1,6)],f'B lr {rate}'+(' (selected)' if rate==primary else ''),COLORS[rate])
        for name in ['history','oracle','plain']:
            ax.axhline(summary['baselines'][name]['query'][metric]['mean'],color=COLORS[name],ls='--',alpha=.8,label=name.title())
        ax.set(xlabel='Support shoves',ylabel=label,xticks=range(1,6));ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Fresh cubes 290–319, fixed query shoves 5–9\nIndependent adaptation resets; bands are five-seed sample SD')
    save_figure(fig,'adaptation_speed')
    for population,name,title in [('first_cube','first_cube_return','Return to first cube 290'),('training_validation','training_validation_forgetting','Training-cube validation: IDs 0–179, shoves 8–9')]:
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        ns=SNAPSHOTS if population=='first_cube' else [0]+SNAPSHOTS
        x=[n-1 for n in ns] if population=='first_cube' else ns
        for ax,metric,label in zip(axes.flat,METRICS,LABELS):
            draw_lines(ax,x,[summary['sequence']['A'][str(n)][population][metric] for n in ns],'A: stored z',COLORS['A'])
            for rate in RATES:draw_lines(ax,x,[summary['sequence']['B'][rate][str(n)][population][metric] for n in ns],f'B lr {rate}'+(' (selected)' if rate==primary else ''),COLORS[rate])
            for baseline in ['history','oracle','plain']:
                ax.axhline(summary['baselines'][baseline][population][metric]['mean'],color=COLORS[baseline],ls='--',alpha=.8,label=baseline.title())
            ax.set(xlabel='Other cubes visited' if population=='first_cube' else 'Total cubes visited',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=7)
            if population=='first_cube':
                for n in [5,15,29]:ax.axvline(n,color='gray',ls=':',lw=.8)
        fig.suptitle(title+'\nB carries weights; A reloads z. Bands: five-seed SD.')
        save_figure(fig,name)

def detailed_csvs(summary):
    results=ROOT/'results';curves=summary['curves']
    with (results/'prefix_scores.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','learning_rate','support_shoves','metric','mean','seed_sd',*['seed_'+str(s) for s in SEEDS]])
        for name,rate,curve in [('A','0.01',curves['A'])]+[('B',r,curves['B'][r]) for r in RATES]:
            for k,values in curve.items():
                for metric,v in values.items():writer.writerow([name,rate,k,metric,v['mean'],v['sample_sd'],*v['seed_values']])
    with (results/'sequence_scores.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','learning_rate','total_visits','population','metric','mean','seed_sd',*['seed_'+str(s) for s in SEEDS]])
        for name,rate,series in [('A','0.01',summary['sequence']['A'])]+[('B',r,summary['sequence']['B'][r]) for r in RATES]:
            for n,pops in series.items():
                for pop,values in pops.items():
                    for metric,v in values.items():writer.writerow([name,rate,n,pop,metric,v['mean'],v['sample_sd'],*v['seed_values']])
    with (results/'current_cube_sequence_scores.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','learning_rate','total_visits','cube_id','metric','mean','seed_sd',*['seed_'+str(s) for s in SEEDS]])
        for method,rate,stem in [('A','0.01','A')]+[('B',r,'B_lr'+r) for r in RATES]:
            records=[read(results/'sequence'/f'{stem}_seed{s}.json') for s in SEEDS]
            for i,cube in enumerate(range(290,320)):
                for metric in METRICS:
                    value=moments([r['visits'][i]['query']['scores'][metric] for r in records])
                    writer.writerow([method,rate,i+1,cube,metric,value['mean'],value['sample_sd'],*value['seed_values']])

def diagnostic_lines():
    names=[('A: fitted z','prefix/A_seed{}_k5.json','query')]+[(f'B lr {r}',f'prefix/B_lr{r}_seed{{}}_k5.json',None) for r in RATES]
    names += [(n.title(),f'baselines/{n}_seed{{}}.json','query') for n in ['history','oracle','plain']]
    keys=['all_coast_velocity_nmse','raw_push_velocity_mse','raw_coast_velocity_mse','legacy_full_state_mse']
    lines=['','## Additional saved one-step diagnostics','',
        'These five-support-shove query scores do not replace the three declared components. All-coast velocity includes the stationary tail and uses the moving-coast energy denominator. Raw velocity MSE has units (m/s)²; legacy full-state MSE precedes division by the full-state energy.','',
        '| Model | All-coast velocity NMSE | Raw push Δv MSE | Raw moving-coast Δv MSE | Legacy full-state MSE |','|---|---:|---:|---:|---:|']
    for label,pattern,subkey in names:
        records=[read(ROOT/'results'/pattern.format(s)) for s in SEEDS]
        scores=[r[subkey]['scores'] if subkey else r['scores'] for r in records]
        lines.append('| '+label+' | '+' | '.join(moment(moments([s[k] for s in scores])) for k in keys)+' |')
    return lines

def complete_report(summary):
    selected=summary['selected_config'];criteria=summary['criteria'];primary=criteria['primary_lr'];curves=summary['curves']
    lines=['# Step 3: adaptation, forgetting, and return','',
        f"**P1 {verdict(criteria['p1']['passed'])}; P2 {verdict(criteria['p2']['passed'])}. Overall {verdict(summary['all_criteria_passed'])}.**",'',
        f'The primary comparison uses B learning rate **{primary}**, chosen on validation cubes only. Both rates are reported; the alternate rate cannot replace the selected comparison.','',
        'Scores are on fresh cubes 290–319, query shoves 5–9. Mean ± sample SD is across five paired seeds; lower is better. Coast means moving-coast steps. The full-state score uses all retained steps. Composite is the equal average of these three separately normalized errors.','']
    rows=[('A: fitted z, five shoves',curves['A']['5'])]+[(f'B lr {r}, five shoves'+(' (selected)' if r==primary else ''),curves['B'][r]['5']) for r in RATES]
    rows += [(n.title(),summary['baselines'][n]['query']) for n in ['history','oracle','plain']]
    lines+=score_table(rows)
    lines+=['','Oracle receives true mass and combined friction. Its frozen Step 2 checkpoint was selected for push velocity change; its actual scores on coast and full state must be assessed separately rather than assumed to lead every component.']
    lines+=['','## Adaptation speed','',
        f"P2: **{verdict(criteria['p2']['passed'])}**; {criteria['p2']['interpretation']}. Earliest qualifying A prefix: **{criteria['p2']['earliest_prefix'] if criteria['p2']['earliest_prefix'] is not None else 'none'}**.",
        'Each prefix independently restarts from training-mean z or the frozen Plain checkpoint and takes exactly 50 updates. All prefixes use the same query shoves. A worse mean is not counted as a match merely because error bars overlap.','']
    rows=[]
    for k in range(1,6):
        rows.append((f'A, {k} support shoves',curves['A'][str(k)]))
        rows.extend((f'B lr {r}, {k} support shoves',curves['B'][r][str(k)]) for r in RATES)
    lines+=score_table(rows)
    lines+=['','![Adaptation speed]('+str(ROOT/'results/figures/adaptation_speed.png')+')','',
        '| B rate | Score | P2 | Earliest A prefix reaching B at five shoves |','|---|---|---|---:|']
    for rate in RATES:
        for metric,label in zip(METRICS,LABELS):
            p=criteria['by_lr'][rate][metric]['p2'];lines.append(f"| {rate} | {label} | {verdict(p['passed'])} | {p['earliest_prefix'] if p['earliest_prefix'] is not None else 'none'} |")
    lines+=['','## Forgetting and return','',
        'The ascending sequence is 290,…,319. B retains all weights between cubes but starts a fresh Adam optimizer at each cube. A stores independent z vectors and reloads them without optimization. A unchanged return error is a structural consequence of frozen shared weights and preserved vectors, not learned resistance to interference.','',
        'Total visits 6,16,30 correspond to returns after 5,15,29 other cubes. P1 requires A non-rise for every seed and B rise beyond seed spread at all three returns: return mean−SD > initial mean+SD. The initial score is after adapting to the first cube.','',
        '| B rate | Score | Other cubes | A non-rise | B rise beyond spread | P1 at return | B paired change mean ± SD |','|---|---|---:|---|---|---|---:|']
    for rate in RATES:
        for metric,label in zip(METRICS,LABELS):
            for other,row in criteria['by_lr'][rate][metric]['p1']['by_other_cubes'].items():
                lines.append(f"| {rate} | {label} | {other} | {verdict(row['a_nonrise'])} | {verdict(row['b_rise_beyond_seed_spread'])} | {verdict(row['passed'])} | {moment(row['b_paired_change'])} |")
    failures=[other for other,row in criteria['p1']['by_other_cubes'].items() if not row['b_rise_beyond_seed_spread']]
    if failures:lines+=['',f"For the selected B rate, the claimed rise beyond seed spread is absent after **{', '.join(failures)} other cubes**. A retained its first-cube prediction through unchanged frozen weights and stored z."]
    lines+=['','| B rate | Score | P1 across all three returns |','|---|---|---|']
    for rate in RATES:
        for metric,label in zip(METRICS,LABELS):lines.append(f"| {rate} | {label} | {verdict(criteria['by_lr'][rate][metric]['p1']['passed'])} |")
    for population,title,figure in [('first_cube','First-cube query scores','first_cube_return'),('training_validation','Training-cube validation scores','training_validation_forgetting')]:
        lines+=['',f'### {title}','']
        lines+=score_table([(name.title()+' (constant reference)',summary['baselines'][name][population]) for name in ['history','oracle','plain']])
        lines+=['','The reference models remain frozen and receive no across-cube adaptation.','']
        rows=[];ns=SNAPSHOTS if population=='first_cube' else [0]+SNAPSHOTS
        for n in ns:
            rows.append((f'A, total visits {n}',summary['sequence']['A'][str(n)][population]))
            rows.extend((f'B lr {r}, total visits {n}',summary['sequence']['B'][r][str(n)][population]) for r in RATES)
        lines+=score_table(rows);lines+=['','!['+title+']('+str(ROOT/'results/figures'/f'{figure}.png')+')']
    lines+=diagnostic_lines()
    data=read(ROOT/'results/full_audit.json');fresh=data['fresh_test_summary'];selected_a=selected['A']
    lines+=['','## Validation choices and data integrity','',
        f"Selected A: dimension **{selected_a['z_dim']}**, penalty **{selected_a['penalty']:g}**, learning rate **0.01**, **50 updates**. Selected B rate: **{primary}**, **50 updates**. The validation-only grid and its component scores are in {link(ROOT/'VALIDATION_SELECTION.md','the selection report')}.",
        'The written protocol preceded all runs. Settings were frozen and announced before fresh properties, commands, or traces were generated. Original 200 cubes and all traces remain byte-identical; 35 Step 2 checkpoints remain frozen. IDs 200–289 are retired from Step 3.',
        f"Fresh collection: **{fresh['launches']}/{fresh['episodes']} launches**, **{fresh['moving_coast_fraction']:.2%}** moving coast after trimming, maximum coast distance **{fresh['maximum_coast_distance_m']:.6f} m**, **{fresh['transitions']:,}** retained and **{fresh['trimmed_transitions']:,}** trimmed transitions. No sample was rejected or replaced.",
        'No phase, length, mask, ID, time, termination feature, or physical label enters A, B, or History. Adaptation uses only support shoves; query scores never update weights or z. True mass and combined friction enter Oracle only.',
        'No world-model retraining or checkpoint selection occurred. Temporary B copies were adapted as specified. Every plotted point comes from saved run records; report generation performs no predictions.','',
        '## Saved evidence','']
    for name,label in [('PROTOCOL.md','Protocol and reasons'),('results/summary.json','All frozen comparisons and criterion arithmetic'),('results/final_evaluation.json','Final evaluation ledger'),('results/final_audit.json','Independent final audit'),('results/full_audit.json','Collection audit'),('results/selection_seal.json','Frozen validation selection'),('results/settings_announced.json','Announcement before test generation'),('results/retired_experiment_seal.json','Preserved earlier evidence'),('results/prefix_scores.csv','All prefix means and individual seeds'),('results/sequence_scores.csv','All sequence and return scores and individual seeds'),('results/current_cube_sequence_scores.csv','Each currently visited cube in the continual sequence')]:
        lines.append('- '+link(ROOT/name,label))
    lines+=['','Raw velocity MSE, the original full-state normalized MSE, and all-coast velocity scores are retained in the per-run JSON records under results/prefix, results/baselines, and results/sequence. Bands show optimization-seed spread, not confidence intervals over possible test cubes or visitation orders. P1 concerns the single designated first cube 290 in one fixed ascending order; it does not establish forgetting across all possible first cubes or orders.']
    write(ROOT/'RESULTS.md','\n'.join(lines))
    write(ROOT/'README.md',f"# Step 3: adaptation comparison\n\nP1 **{verdict(criteria['p1']['passed'])}**, P2 **{verdict(criteria['p2']['passed'])}**; overall **{verdict(summary['all_criteria_passed'])}**.\n\nSee [RESULTS.md](RESULTS.md) for all component scores, adaptation curves, forgetting and return checks. [PROTOCOL.md](PROTOCOL.md) preceded all runs. [VALIDATION_SELECTION.md](VALIDATION_SELECTION.md) records choices made before new test generation.\n\nOriginal training data and Step 2 checkpoints remain unchanged. No further experiment is implied.")

def manifest():
    paths=[ROOT/'PROTOCOL.md',ROOT/'report.py']
    paths += [p for p in (ROOT/'results').rglob('*.json') if p.name!='report_inputs.json']
    value={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    (ROOT/'results/report_inputs.json').write_text(json.dumps(value,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validation-only',action='store_true');args=parser.parse_args()
    if not (ROOT/'results/selected_config.json').exists():
        stop=read(ROOT/'results/STOP.json') if (ROOT/'results/STOP.json').exists() else None
        if stop is None:raise RuntimeError('Validation selection is not complete')
        write(ROOT/'RESULTS.md','# Step 3: stopped before validation selection\n\nNo fresh-test or P1/P2 result is available.\n\n'+json.dumps(stop,indent=2))
        manifest();return
    if args.validation_only or not (ROOT/'VALIDATION_SELECTION.md').exists():render_validation()
    if args.validation_only:
        print('Validation selection report rendered from saved records only.');return
    ledger=read(ROOT/'results/final_evaluation.json') if (ROOT/'results/final_evaluation.json').exists() else {}
    if ledger.get('status')!='complete':
        stop=read(ROOT/'results/STOP.json') if (ROOT/'results/STOP.json').exists() else None
        if stop is None:raise RuntimeError('Final report waits for completed evaluation or explicit STOP')
        write(ROOT/'RESULTS.md','# Step 3: stopped\n\nThe experiment stopped before completing the comparison. P1/P2 are not inferred from incomplete results.\n\n'+json.dumps(stop,indent=2)+'\n\n'+link(ROOT/'VALIDATION_SELECTION.md','Completed validation-only results'))
        manifest();return
    audit=read(ROOT/'results/final_audit.json')
    if audit.get('status')!='passed':raise RuntimeError('Independent final audit must pass before complete report')
    summary=read(ROOT/'results/summary.json');plots(summary);detailed_csvs(summary);complete_report(summary);manifest()
    print(json.dumps({'report':str(ROOT/'RESULTS.md'),'P1':summary['criteria']['p1']['passed'],'P2':summary['criteria']['p2']['passed']}))

if __name__=='__main__':main()
