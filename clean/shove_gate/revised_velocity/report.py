"""Report completed velocity-gate artifacts only; never simulate, fit, or score."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clean.shove_gate.report_direct import read,sha,fmt,mean_sd,improvement,plot_panel,link,COLORS

ROOT=Path(__file__).resolve().parent
GROUPS=('push','moving_coast','all_steps')
GROUP_LABELS={'push':'Push','moving_coast':'Moving coast','all_steps':'All retained steps'}
PRIMARY='horizontal_velocity_delta_nmse'
METRICS=(PRIMARY,'one_step_nmse','rollout10_nmse','rollout10_final_nmse','stopping_position_error_m')
LABELS={PRIMARY:'Horizontal Δvelocity NMSE','one_step_nmse':'Full-state one-step NMSE',
        'rollout10_nmse':'Ten-step rollout NMSE','rollout10_final_nmse':'Tenth-step-only NMSE',
        'stopping_position_error_m':'Stopping-position error (m)'}


def checked_inputs(root):
    names=dict(summary='results/gate_summary.json',ledger='results/final_evaluation.json',
        full='results/full_audit.json',probe='results/action_probe.json',timing='results/collection_timing.json',
        physics='results/physics_floor.json',test_physics='results/test_physics_floor.json',
        checkpoint_seal='results/checkpoint_seal.json',test_data_seal='results/test_data_seal.json',
        normalizer='results/normalizer.json',model_protocol='results/model_protocol.json')
    paths={k:root/v for k,v in names.items()}; values={k:read(p) for k,p in paths.items()}
    s=values['summary']; ledger=values['ledger']
    if ledger['status']!='complete' or ledger['summary_sha256']!=sha(paths['summary']):
        raise RuntimeError('Requires completed, byte-matched final evaluation')
    if ledger['primary_gate_passed']!=s['primary_gate_passed'] or s['primary_metric']!=PRIMARY:
        raise RuntimeError('Gate decision or primary metric mismatch')
    if values['full']['status']!='passed' or s['test_physics_floor_sha256']!=sha(paths['test_physics']):
        raise RuntimeError('Failed collection audit or changed physics reference')
    if values['physics']['protocol_sha256']!=sha(root/'REVISED_PROTOCOL.md'):
        raise RuntimeError('Written revision protocol changed')
    paths['written_protocol']=root/'REVISED_PROTOCOL.md';paths['physics_interpretation']=root/'PHYSICS_FLOOR_REPORT.md'
    records={}; training={}; pairs={}
    for seed in s['seeds']:
        pairpath=root/'results/training_logs'/f'pair_seed_{seed}.json';pairs[seed]=read(pairpath);paths[f'pair{seed}']=pairpath
        for kind in ['plain','oracle']:
            name=f'{kind}_seed_{seed}'; p=root/'results/test_metrics'/f'{name}.json'
            r=read(p);records[kind,seed]=r;paths['test_'+name]=p
            expected=values['checkpoint_seal']['checkpoints'][name+'.pt']
            if r['kind']!=kind or r['seed']!=seed or r['checkpoint_sha256']!=expected or sha(root/'checkpoints'/f'{name}.pt')!=expected:
                raise RuntimeError('Scored checkpoint identity/hash mismatch')
            p=root/'results/training_logs'/f'{name}.json';t=read(p);training[kind,seed]=t;paths['train_'+name]=p
            if t['completed_updates']!=pairs[seed]['completed_updates'] or t['minibatch_sha256']!=pairs[seed]['minibatch_sha256']:
                raise RuntimeError('Paired training budget/minibatch mismatch')
            best=min(t['log'],key=lambda row:row['validation_nmse'])
            if best['update']!=t['selected_update'] or best['validation_nmse']!=t['selected_validation_nmse']:
                raise RuntimeError('Checkpoint did not select minimum primary validation score')
    for group in GROUPS:
        for metric in METRICS:
            for kind in ['plain','oracle']:
                vals=[records[kind,seed]['groups'][group][metric]['mean'] for seed in s['seeds']]
                item=s['comparisons'][group][metric][kind]
                if vals!=item['seed_values']:raise RuntimeError('Summary seed scores differ')
                if all(v is not None for v in vals):
                    if not np.isclose(np.mean(vals),item['mean'],rtol=1e-12,atol=1e-15) or not np.isclose(np.std(vals,ddof=1),item['sample_sd'],rtol=1e-12,atol=1e-15):
                        raise RuntimeError('Summary moments differ')
                elif item['mean'] is not None:raise RuntimeError('Nonfinite seed silently dropped')
    if (root/'results/final_audit.json').exists():paths['final_audit']=root/'results/final_audit.json'
    return values,records,training,pairs,paths


def save_figure(fig,path):
    for extension in ['png','svg']:fig.savefig(path.with_suffix('.'+extension),dpi=180,bbox_inches='tight')
    plt.close(fig)


def make_figures(root,values,training,pairs):
    output=root/'results/figures';output.mkdir(exist_ok=True)
    s=values['summary'];seeds=s['seeds'];primary=s['comparisons']['moving_coast'][PRIMARY]
    fig,ax=plt.subplots(figsize=(7.5,5.5))
    plot_panel(ax,primary,seeds,'Moving-coast horizontal velocity change','NMSE of Δ(vx, vy)')
    for key,color,label,style in [('requested_unclipped','#298251','Requested physics reference','--'),
                                   ('stop_clipped_diagnostic','#7356a2','Stop-clipped diagnostic',':')]:
        ax.axhline(s['physics_comparison'][key]['physics_nmse'],color=color,ls=style,lw=1.4,label=label)
    ax.legend(fontsize=8,loc='best')
    verdict='PASS' if s['primary_gate_passed'] else 'FAIL'
    fig.suptitle(f'Fresh-test velocity gate: {verdict}',fontsize=16)
    fig.text(.5,.015,f'Five paired seeds; black diamonds: mean ± sample SD. '
             f'Oracle improvement {improvement(primary["relative_mean_improvement"])}.\n'
             'Reference lines use the same fresh cubes and training-only normalization.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.95));save_figure(fig,output/'velocity_gate')
    shown=(PRIMARY,'one_step_nmse','rollout10_nmse','stopping_position_error_m')
    fig,axes=plt.subplots(3,4,figsize=(16.3,10.1))
    for row,group in enumerate(GROUPS):
        for col,metric in enumerate(shown):
            ax=axes[row,col];plot_panel(ax,s['comparisons'][group][metric],seeds,GROUP_LABELS[group],LABELS[metric])
            if group=='moving_coast' and metric==PRIMARY:
                ax.set_facecolor('#fff5e9');ax.set_title('Moving coast — gate',fontsize=11)
            if metric=='stopping_position_error_m':
                ax.set_yscale('symlog',linthresh=1.)
                ax.set_ylabel('Stopping error (m; symmetric log)',fontsize=9)
                comparison=s['comparisons'][group][metric]
                finite=[x for k in ['plain','oracle'] for x in comparison[k]['seed_values'] if x is not None]
                lows=[comparison[k]['mean']-comparison[k]['sample_sd'] for k in ['plain','oracle'] if comparison[k]['mean'] is not None]
                highs=[comparison[k]['mean']+comparison[k]['sample_sd'] for k in ['plain','oracle'] if comparison[k]['mean'] is not None]
                if finite:
                    ax.set_ylim(min(-1.,min(lows)*3),max(1.,max(finite+highs)*3))
    fig.suptitle('Fresh test cubes: primary and secondary prediction scores',fontsize=16)
    fig.text(.5,.012,'Actual paired runs; black diamonds: mean ± sample SD. Lower is better; axes differ by panel.\n'
             'Stopping axes use symmetric log (linear within ±1 m), retaining every seed and the full SD bars.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.965));save_figure(fig,output/'all_metrics')
    fig,axes=plt.subplots(3,2,figsize=(12.6,10.7));axes=axes.ravel()
    for ax,seed in zip(axes,seeds):
        for kind in ['plain','oracle']:
            t=training[kind,seed];rows=t['log']
            xs=[r['update'] for r in rows];ys=[r['validation_horizontal_velocity_delta_nmse'] for r in rows]
            ax.plot(xs,ys,color=COLORS[kind],lw=1.25,label=kind.title())
            ax.scatter(t['selected_update'],t['selected_validation_nmse'],marker='*',s=120,
                       color=COLORS[kind],edgecolor='black',linewidth=.5,zorder=4)
        for decision in pairs[seed]['extension_decisions'][:-1]:ax.axvline(decision['endpoint'],color='#aaa',ls=':',lw=.7)
        ax.set_title(f'Seed {seed}: {pairs[seed]["completed_updates"]:,} updates/model',fontsize=11)
        ax.set_yscale('log');ax.set_xlabel('Training updates');ax.set_ylabel('Validation Δvelocity NMSE (log)')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[-1].axis('off');axes[-1].text(.03,.92,
        'Every saved validation check (500-update spacing).\n\n'
        'Stars mark selected checkpoints.\nDotted lines mark completed extension blocks.\n\n'
        'Minimum 10,000 updates per paired seed.\nExtend both models when either improves ≥0.1%\n'
        'over its previous best in the block’s final 2,500 updates.\n\n'
        'Maximum 100,000 updates; no test-based extension.',va='top',fontsize=11,linespacing=1.5)
    fig.suptitle('Validation curves: moving-coast horizontal velocity change',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.96));save_figure(fig,output/'validation_curves')
    return output


def generate(root=ROOT):
    root=Path(root).resolve();v,records,training,pairs,paths=checked_inputs(root)
    s=v['summary'];primary=s['comparisons']['moving_coast'][PRIMARY];verdict='PASS' if s['primary_gate_passed'] else 'FAIL'
    figures=make_figures(root,v,training,pairs)
    floor=v['physics'];alltrain=floor['splits']['all_training'];scale=floor['normalization']['scalar_horizontal_delta_mse']
    near=alltrain['metrics']['requested_unclipped']['nmse']<=.01
    lines=['# Revised horizontal-velocity prediction gate','',
        f'**Gate: {verdict}.** On 30 fresh test cubes, Oracle’s moving-coast horizontal-velocity-change '
        f'NMSE is **{mean_sd(primary["oracle"])}**, versus **{mean_sd(primary["plain"])}** for Plain '
        f'across five paired seeds. Relative improvement: **{improvement(primary["relative_mean_improvement"])}**.','',
        f'The ≥20% reduction condition is **{"met" if s["at_least_twenty_percent"] else "not met"}**; '
        f'the seed-spread condition is **{"met" if s["beyond_seed_spread"] else "not met"}**. '
        f'The mean gap is {fmt(primary["mean_gap"])} versus a sum of seed SDs of '
        f'{fmt(primary["plain"]["sample_sd"]+primary["oracle"]["sample_sd"] if primary["defined"] else None)}. '
        'The gate requires `Oracle mean + SD < Plain mean − SD`.','',
        f'![Fresh-test primary gate]({figures/"velocity_gate.png"})','',
        '## Written revision and retired test set','',
        'The '+link(root,'REVISED_PROTOCOL.md','revised protocol and reasons')+' were saved before physics '
        'scoring, fresh collection, and revised training. The previous 15-coordinate average could obscure '
        'a horizontal friction signal; this motivated a new metric and does not establish the cause of the '
        'earlier failure. The new primary score isolates horizontal velocity change and uses a fitting-only '
        'change-energy scale. Checkpoints now minimize this validation score, with a predeclared extension rule.','',
        'The old test results informed this revision. IDs 200–229 are retired; their data, checkpoints and '
        'reported results remain unchanged in the parent experiment. New test IDs 230–259 use seed '
        '**2026111517**, the same generator, and no outcome-based rejection. The 200 training cubes and '
        'their original traces are reused byte-for-byte.','',
        '## Physics sanity reference measured before learning','',
        'The requested prediction is `−combined_friction × 9.81 × 0.005 × vxy/|vxy|`, with zero at exactly '
        'zero speed. It is not clipped. Near-zero was defined before measurement as NMSE ≤0.01. '
        f'The all-training reference **{"meets" if near else "does not meet"}** that threshold.','',
        '| Training subset | Requested NMSE | Requested RMSE (m/s) | Stop-clipped diagnostic NMSE | Moving-coast steps |',
        '|---|---:|---:|---:|---:|']
    if primary['defined'] and primary['mean_gap']<0:
        summed_sd=primary['plain']['sample_sd']+primary['oracle']['sample_sd']
        worse_every_seed=all(o>p for p,o in zip(primary['plain']['seed_values'],primary['oracle']['seed_values']))
        explanation=(f'**Oracle is worse by {-100*primary["relative_mean_improvement"]:.2f}%** on the new primary metric'
                     + (' in all five seeds. ' if worse_every_seed else '. '))
        if -primary['mean_gap']>summed_sd:
            explanation+=f'The gap in Plain’s favor ({-primary["mean_gap"]:.7f}) exceeds the combined seed SD ({summed_sd:.7f}); the clear separation favors Plain.'
        lines[6:6]=[explanation,'']
    for key,label in [('fitting','Fitting episodes 0–7'),('validation','Validation episodes 8–9'),('all_training','All training')]:
        metrics=floor['splits'][key]['metrics'];r=metrics['requested_unclipped'];c=metrics['stop_clipped_diagnostic']
        lines.append(f'| {label} | {r["nmse"]:.8f} | {r["rmse_m_s"]:.8f} | {c["nmse"]:.8f} | {r["eligible_starts"]:,} |')
    d=alltrain['diagnostics'];unclip=alltrain['metrics']['requested_unclipped']['nmse'];clip=alltrain['metrics']['stop_clipped_diagnostic']['nmse']
    lines += ['',f'The full-step decrement would cross zero in {d["ideal_step_would_cross_zero"]["eligible_starts"]:,} '
        f'of {alltrain["metrics"]["requested_unclipped"]["eligible_starts"]:,} starts. The recorded cube stops '
        f'instead of reversing. Clipping the decrement to available speed reduces reference error by '
        f'{100*(1-clip/unclip):.2f}%; outside these stopping crossings, NMSE is '
        f'{d["ideal_step_would_not_cross_zero"]["unclipped_nmse"]:.8f}. At speeds ≥0.1 m/s, measured '
        f'deceleration divided by combined friction × gravity is '
        f'{d["speed_ge_0p1m_s"]["observed_deceleration_over_mu_g"]:.8f}. '
        'These measurements closely match simple sliding friction; the dominant deviation is the final '
        'partial stopping step. Small remaining contact-simulation residuals have no identified solver cause.','',
        f'The frozen normalization scalar is **{scale:.12g} (m/s)²**, the equal-cube/episode average '
        'horizontal delta energy on fitting moving-coast steps. A zero-change predictor has NMSE 1 on '
        'that subset. Validation-only sanity error is reported separately even when it lies above 0.01. '
        'The analytic reference is a diagnostic baseline, not a proven irreducible error floor.','',
        link(root,'PHYSICS_FLOOR_REPORT.md','Detailed pretraining physics interpretation and diagnostics'),'',
        '## Fresh test collection and action independence','',
        '| Measurement | Reused 200 training cubes | Fresh 30 test cubes |','|---|---:|---:|']
    train=v['full']['training_summary'];fresh=v['full']['fresh_test_summary']
    for label,key,formatter in [('Episodes','episodes',lambda x:f'{x:,}'),('Launch rate','launch_fraction',lambda x:f'{x:.2%}'),
        ('Moving fraction of retained coast','moving_coast_fraction',lambda x:f'{x:.2%}'),
        ('Maximum coast distance','maximum_coast_distance_m',lambda x:f'{x:.6f} m'),
        ('Retained transitions','transitions',lambda x:f'{x:,}'),('Trimmed stationary transitions','trimmed_transitions',lambda x:f'{x:,}')]:
        lines.append(f'| {label} | {formatter(train[key])} | {formatter(fresh[key])} |')
    lines += ['',f'Fresh collection took {v["timing"]["this_invocation_seconds"]:.2f} s wall time. '
        'Force remains uniform 4–7 N at the COM, direction/start position/yaw random, and duration uniform '
        'over 60,65,…,100 ms. Actions are exactly zero in coast. Every episode reached confirmed rest. '
        'The fixed 1.5 s raw coast, 100 ms stop confirmation and five-step trimmed tail are unchanged. '
        'No moving step was trimmed, and no sampled episode was discarded.','',
        '| Actions-only probe, planned commands | Mass R² ± seed SD | Combined friction R² ± seed SD |',
        '|---|---:|---:|']
    for family,item in v['probe']['generator']['aggregate'].items():
        m=item['mass_kg'];f=item['combined_friction']
        lines.append(f'| {family.replace("_"," ")} | {m["mean_r2"]:.5f} ± {m["seed_std_r2"]:.5f} | {f["mean_r2"]:.5f} ± {f["seed_std_r2"]:.5f} |')
    lines += ['', 'Probe hyperparameters use training-cube validation only. Commands describe the fixed '
        'first 100 ms force schedule and independently commanded duration. No observed length, padding '
        'mask, stop time, ID, phase or termination signal enters either dynamics model. Per-cube true '
        'mass and friction remain in `cubes.json`; only Oracle receives mass and combined friction.','',
        '## Fresh held-out prediction scores','',
        'All scores weight starts within episode, episodes within cube, then cubes equally. Positive '
        'improvement favors Oracle. “Beyond spread” means the positive Plain-minus-Oracle mean gap '
        'exceeds the sum of the two sample SDs. Five-seed SD measures optimization variation.','',
        '| Group | Metric | Plain mean ± SD | Oracle mean ± SD | Gap | Improvement | Beyond spread? |',
        '|---|---|---:|---:|---:|---:|---|']
    for group in GROUPS:
        for metric in METRICS:
            r=s['comparisons'][group][metric];label=GROUP_LABELS[group]
            if group=='moving_coast' and metric==PRIMARY:label='**Moving coast — gate**'
            lines.append(f'| {label} | {LABELS[metric]} | {mean_sd(r["plain"])} | {mean_sd(r["oracle"])} | '
                         f'{fmt(r["mean_gap"])} | {improvement(r["relative_mean_improvement"])} | {"yes" if r["gap_exceeds_sum_of_seed_sds"] else "no"} |')
    lines += ['',f'![All groups and metrics]({figures/"all_metrics.png"})','',
        'The new NMSE divides horizontal-change MSE by the fixed fitting scale. Old full-state metrics '
        'retain their original 15-coordinate normalization. Ten-step forecasts recursively use predicted '
        'states; the primary ten-step secondary metric averages all ten horizons, with tenth-only '
        'error also shown. Push windows stay in push; moving-coast windows start moving and remain in coast. '
        'Stopping-position forecasts run to planned release +1.5 s, never to an observed stop or trimmed '
        'length. No clipping, physics correction, or observed-state resets are applied.','',
        '| Seed | Plain primary NMSE | Oracle primary NMSE | Paired Oracle improvement |','|---:|---:|---:|---:|']
    for i,seed in enumerate(s['seeds']):
        lines.append(f'| {seed} | {fmt(primary["plain"]["seed_values"][i],8)} | {fmt(primary["oracle"]["seed_values"][i],8)} | {improvement(primary["paired_seed_improvements"][i]) if primary["defined"] else "undefined"} |')
    lines += ['', '**Long-horizon instability remains in the reported results.** Finite rollout values can '
        'still be catastrophic. The per-seed stopping-position errors below are retained in the five-seed '
        'means and SDs; no outlier was dropped, clipped, or replaced. These are prediction errors, not '
        'physical distances traveled by the cubes.','',
        '| Seed | Plain moving-coast stop error (m) | Oracle moving-coast stop error (m) | Oracle push stop error (m) | Oracle all-step stop error (m) |',
        '|---:|---:|---:|---:|---:|']
    for i,seed in enumerate(s['seeds']):
        coast=s['comparisons']['moving_coast']['stopping_position_error_m']
        push=s['comparisons']['push']['stopping_position_error_m']
        allsteps=s['comparisons']['all_steps']['stopping_position_error_m']
        lines.append(f'| {seed} | {fmt(coast["plain"]["seed_values"][i],9)} | {fmt(coast["oracle"]["seed_values"][i],9)} | '
                     f'{fmt(push["oracle"]["seed_values"][i],9)} | {fmt(allsteps["oracle"]["seed_values"][i],9)} |')
    oracle_stops=s['comparisons']['moving_coast']['stopping_position_error_m']['oracle']['seed_values']
    if all(value is not None for value in oracle_stops):
        maximum_index=int(np.argmax(oracle_stops));other=[x for i,x in enumerate(oracle_stops) if i!=maximum_index]
        if oracle_stops[maximum_index]>100*max(other):
            lines += ['',f'Oracle seed **{s["seeds"][maximum_index]}** has moving-coast stopping error '
                f'**{oracle_stops[maximum_index]:,.3f} m**, while the other Oracle seeds range '
                f'{min(other):.4f}–{max(other):.4f} m. This unstable run drives the five-seed Oracle '
                f'mean to **{s["comparisons"]["moving_coast"]["stopping_position_error_m"]["oracle"]["mean"]:,.3f} m**.']
    lines += ['', '## Oracle relative to physics on the same fresh cubes','',
        '| Reference | Physics NMSE | Oracle NMSE | Oracle excess NMSE | Oracle / physics error | Physics RMSE (m/s) | Oracle RMSE (m/s) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,label in [('requested_unclipped','Requested unclipped formula'),('stop_clipped_diagnostic','Stop-clipped diagnostic')]:
        r=s['physics_comparison'][key]
        lines.append(f'| {label} | {fmt(r["physics_nmse"],8)} | {fmt(r["oracle_mean_nmse"],8)} | {fmt(r["oracle_excess_nmse"],8)} | '
                     f'{fmt(r["oracle_to_physics_error_ratio"],6)} | {fmt(r["physics_rmse_m_s"],8)} | {fmt(r["oracle_rmse_from_mean_mse_m_s"],8)} |')
    lines += ['', 'Oracle RMSE is the square root of its five-seed mean MSE. The unclipped formula is the '
        'requested comparison; the stop-clipped variant is a separately labeled diagnostic. Both use '
        'the same fresh moving-coast starts and the fitting-only normalization.','',
        '## Validation curves and adaptive training','',
        'Both models retain three 256-unit ReLU layers and the original full-state normalized-change '
        'training loss. Episodes 0–7 fit; episodes 8–9 select checkpoints using moving-coast horizontal '
        'velocity-change NMSE. Adam, batch512 and paired minibatch streams are unchanged. Learning rate '
        'cosine-decays 0.001→0.0001 over the first 10,000 updates, then remains at 0.0001.','',
        'Every 10,000 updates, the protocol compares the best validation score in the last 2,500 updates '
        'with the previous best. At least 0.1% improvement by either family extends both by another '
        '10,000, up to 100,000. Checkpoint selection is independent of the final update budget.','',
        f'![Validation curves]({figures/"validation_curves.png"})','',
        '| Seed | Updates/model | Plain selected update | Oracle selected update | Plain selected validation NMSE | Oracle selected validation NMSE | Still improving at cap? |',
        '|---:|---:|---:|---:|---:|---:|---|']
    for seed in s['seeds']:
        p=training['plain',seed];o=training['oracle',seed];pair=pairs[seed]
        lines.append(f'| {seed} | {pair["completed_updates"]:,} | {p["selected_update"]:,} | {o["selected_update"]:,} | '
                     f'{fmt(p["selected_validation_nmse"],8)} | {fmt(o["selected_validation_nmse"],8)} | {"yes" if pair["convergence_not_established_at_cap"] else "no"} |')
    lines += ['', '| Seed | Block end | Plain final-window improvement | Oracle final-window improvement | Extend both? |','|---:|---:|---:|---:|---|']
    for seed in s['seeds']:
        for decision in pairs[seed]['extension_decisions']:
            lines.append(f'| {seed} | {decision["endpoint"]:,} | {improvement(decision["families"]["plain"]["relative_improvement"])} | '
                         f'{improvement(decision["families"]["oracle"]["relative_improvement"])} | {"yes" if decision["extend"] else "no"} |')
    capped=[seed for seed in s['seeds'] if pairs[seed]['convergence_not_established_at_cap']]
    if capped:lines += ['', '**Resource cap reached while still improving:** seeds '+', '.join(map(str,capped))+'. Validation convergence was not established for these pairs.']
    nonfinite=sum(r['groups'][g][m]['nonfinite_starts'] for r in records.values() for g in GROUPS for m in METRICS)
    lines += ['',f'Summed pair training wall time: **{sum(p["elapsed_seconds"] for p in pairs.values()):.2f} s**; '
        f'summed model test-prediction time: **{sum(r["elapsed_seconds"] for r in records.values()):.2f} s**. '
        'These exclude collection, startup, preparation, audits, and reporting.','',
        ('No scored metric/start/seed entries were nonfinite.' if nonfinite==0 else
         f'{nonfinite:,} metric/start/seed entries were nonfinite; corresponding aggregates remain undefined rather than dropping them.'),
        'All ten checkpoints were frozen before the sole predictive test evaluation. Test results did '
        'not select checkpoints, extend training, or change the gate.','',
        ('The revised gate passed. This supports improved one-step horizontal velocity-change prediction '
         'with true parameters under this simulator and protocol. It does not establish a learned belief '
         'model or reliable long-horizon control.' if s['primary_gate_passed'] else
         'The revised gate failed. Work stops at this comparison; no test-based tuning or belief-model training follows.'),
        '', '## Saved evidence','']
    for name,label in [('REVISED_PROTOCOL.md','Protocol and reasons written before running'),('PHYSICS_FLOOR_REPORT.md','Pretraining physics interpretation'),
        ('results/physics_floor.json','Saved training physics measurements'),('results/test_physics_floor.json','Fresh-test physics measurements'),
        ('results/gate_summary.json','All five-seed comparisons and gate'),('results/final_evaluation.json','Completed final-evaluation ledger'),
        ('results/full_audit.json','Collection audit'),('results/action_probe.json','Commands-only probe'),
        ('results/checkpoint_seal.json','Frozen checkpoint/source manifest'),('results/test_data_seal.json','Fresh held-out byte hashes')]:
        lines.append('- '+link(root,name,label))
    if 'final_audit' in paths:lines.append('- '+link(root,'results/final_audit.json','Independent final results audit'))
    lines += ['', 'All figures come from saved measurements and validation logs. No fabricated runs or '
        'smoothed validation samples are used. PNG/SVG files are in `results/figures/`.','',
        'Rebuild this report only after final evaluation completes:','', '```bash',
        'PYTHONPATH=clean/vendor:. MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove_gate.revised_velocity.report','```','']
    output=root/'RESULTS.md';output.write_text('\n'.join(lines)+'\n')
    manifest={str(p.relative_to(root)):sha(p) for p in paths.values()};manifest['report.py']=sha(__file__)
    (root/'results/report_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(report=str(output),figures=str(figures),gate=verdict)))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    generate(parser.parse_args().root)
