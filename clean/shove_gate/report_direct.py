"""Build the direct-force gate report and figures from completed saved results.

This module never collects physics, trains models, or evaluates predictions.
It refuses to run before the single final evaluation has completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

ROOT=Path(__file__).resolve().parent
GROUPS=('push','moving_coast','all_steps')
GROUP_LABELS={'push':'Push','moving_coast':'Moving coast','all_steps':'All retained steps'}
METRICS=('one_step_nmse','rollout10_nmse','stopping_position_error_m')
METRIC_LABELS={'one_step_nmse':'One-step NMSE','rollout10_nmse':'Ten-step rollout NMSE',
               'stopping_position_error_m':'Stopping-position error (m)'}
COLORS={'plain':'#256b9b','oracle':'#b35418'}


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(value,digits=5):return 'undefined' if value is None else f'{float(value):.{digits}g}'


def mean_sd(record):
    return 'undefined' if record['mean'] is None else f"{fmt(record['mean'])} ± {fmt(record['sample_sd'],3)}"


def improvement(value):return 'undefined' if value is None else f'{100*float(value):+.2f}%'


def link(root,name,label=None):return f'[{label or name}]({root/name})'


def checked_inputs(root):
    names={
        'summary':'results/gate_summary.json','ledger':'results/final_evaluation.json',
        'pilot':'results/pilot_audit.json','full':'results/full_audit.json','pilot_gate':'results/pilot_gate.json',
        'probe':'results/action_probe.json','metrics':'results/metrics.json',
        'protocol':'data/protocol.json','model_protocol':'results/model_protocol.json',
        'checkpoint_seal':'results/checkpoint_seal.json','test_data_seal':'results/test_data_seal.json',
        'timing':'results/timing.json','pilot_timing':'results/pilot_timing.json',
    }
    paths={key:root/name for key,name in names.items()}
    values={key:read(path) for key,path in paths.items()}
    if values['ledger'].get('status')!='complete':
        raise RuntimeError('The final evaluation must be complete before reporting')
    if values['ledger']['summary_sha256']!=sha(paths['summary']):
        raise RuntimeError('Final summary does not match completed evaluation ledger')
    if not values['pilot_gate']['passed'] or values['pilot']['status']!='passed' or values['full']['status']!='passed':
        raise RuntimeError('Pilot authorization or collected-data integrity did not pass')
    if values['summary']['primary_gate_passed']!=values['ledger']['primary_gate_passed']:
        raise RuntimeError('Gate decision disagrees with final ledger')
    records={};training={}
    for seed in values['summary']['seeds']:
        for kind in ['plain','oracle']:
            name=f'{kind}_seed_{seed}'
            path=root/'results/test_metrics'/f'{name}.json'
            result=read(path);records[(kind,seed)]=result
            if result['kind']!=kind or result['seed']!=seed:
                raise RuntimeError('Saved test result identity mismatch')
            expected=values['checkpoint_seal']['checkpoints'][name+'.pt']
            if result['checkpoint_sha256']!=expected or sha(root/'checkpoints'/f'{name}.pt')!=expected:
                raise RuntimeError('Scored checkpoint differs from frozen checkpoint')
            paths['test_'+name]=path
            training_path=root/'results/training_logs'/f'{name}.json'
            training[(kind,seed)]=read(training_path);paths['training_'+name]=training_path
    # Independently recompute summary moments from the saved seed-level scores.
    for group in GROUPS:
        for metric in METRICS:
            for kind in ['plain','oracle']:
                seed_values=[records[(kind,s)]['groups'][group][metric]['mean'] for s in values['summary']['seeds']]
                item=values['summary']['comparisons'][group][metric][kind]
                if seed_values!=item['seed_values']:raise RuntimeError('Seed scores disagree with summary')
                if all(v is not None for v in seed_values):
                    if not np.isclose(np.mean(seed_values),item['mean'],rtol=1e-12,atol=1e-15):raise RuntimeError('Mean mismatch')
                    if not np.isclose(np.std(seed_values,ddof=1),item['sample_sd'],rtol=1e-12,atol=1e-15):raise RuntimeError('Seed spread mismatch')
                elif item['mean'] is not None:raise RuntimeError('Undefined seed was silently dropped')
    return values,records,training,paths


def plot_panel(ax,comparison,seeds,title,ylabel):
    offsets=np.linspace(-.075,.075,len(seeds))
    collected=[]
    for i,_seed in enumerate(seeds):
        pair=[comparison[k]['seed_values'][i] for k in ['plain','oracle']]
        if all(v is not None and np.isfinite(v) for v in pair):
            ax.plot(np.array([0,1])+offsets[i],pair,color='#b8b8b8',lw=.9,zorder=1)
        for x,(kind,value) in enumerate(zip(['plain','oracle'],pair)):
            if value is not None and np.isfinite(value):
                ax.scatter(x+offsets[i],value,s=28,color=COLORS[kind],alpha=.85,zorder=2)
                collected.append(value)
    undefined=[]
    for x,kind in enumerate(['plain','oracle']):
        row=comparison[kind]
        if row['mean'] is not None:
            ax.errorbar(x,row['mean'],yerr=row['sample_sd'],fmt='D',color='black',
                        ms=5,capsize=5,elinewidth=1.2,zorder=3)
        else:
            undefined.append(kind.title())
    if undefined:
        ax.text(.5,.97,'Undefined aggregate: '+', '.join(undefined),transform=ax.transAxes,
                ha='center',va='top',fontsize=8,color='#a02b23')
    ax.set_xticks([0,1],['Plain','Oracle']);ax.set_xlim(-.35,1.35)
    ax.set_title(title,fontsize=11);ax.set_ylabel(ylabel,fontsize=9)
    ax.grid(axis='y',alpha=.2)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    if collected and min(collected)>=0:
        bottom=min(0.,min(comparison[k]['mean']-comparison[k]['sample_sd']
                          for k in ['plain','oracle'] if comparison[k]['mean'] is not None)) if not undefined else 0.
        ax.set_ylim(bottom=bottom)


def make_figures(root,summary):
    output=root/'results/figures';output.mkdir(parents=True,exist_ok=True)
    seeds=summary['seeds']
    fig,axes=plt.subplots(3,3,figsize=(12.4,10.1))
    for row,group in enumerate(GROUPS):
        for col,metric in enumerate(METRICS):
            comparison=summary['comparisons'][group][metric]
            plot_panel(axes[row,col],comparison,seeds,GROUP_LABELS[group],METRIC_LABELS[metric])
            if group=='moving_coast' and metric=='one_step_nmse':
                axes[row,col].set_facecolor('#fff5e9')
                axes[row,col].set_title('Moving coast — primary gate',fontsize=11)
    fig.suptitle('Direct-force shoves: held-out cube predictions',fontsize=16,y=.995)
    fig.text(.5,.012,'Colored points: five saved seeds. Lines pair the same seed. Black diamonds: mean ± sample SD.\n'
             'Error bars show seed spread, not confidence intervals. Lower is better; axes differ by panel.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.965))
    for extension in ['png','svg']:fig.savefig(output/f'gate_metrics.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,5.2))
    primary=summary['comparisons']['moving_coast']['one_step_nmse']
    plot_panel(ax,primary,seeds,'Moving-coast one-step prediction','Normalized MSE')
    verdict='PASS' if summary['primary_gate_passed'] else 'FAIL'
    fig.suptitle(f'Oracle gate: {verdict}',fontsize=16)
    fig.text(.5,.025,'Paired seeds and mean ± sample SD; no test-driven tuning.\n'
             f'Mean improvement: {improvement(primary["relative_mean_improvement"])}. '
             f'Beyond seed spread: {"yes" if primary["gap_exceeds_sum_of_seed_sds"] else "no"}.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.95))
    for extension in ['png','svg']:fig.savefig(output/f'moving_coast_gate.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    return output


def probe_rows(lines,aggregate,scope):
    for family in sorted(aggregate):
        mass=aggregate[family]['mass_kg'];mu=aggregate[family]['combined_friction']
        lines.append(f'| {scope} | {family.replace("_"," ")} | '
                     f'{mass["mean_r2"]:.4f} ± {mass["seed_std_r2"]:.4f} | '
                     f'{mu["mean_r2"]:.4f} ± {mu["seed_std_r2"]:.4f} |')


def generate(root=ROOT):
    root=Path(root).resolve();values,records,training,paths=checked_inputs(root)
    summary,pilot,full,probe,protocol=(values[k] for k in ['summary','pilot','full','probe','protocol'])
    figure_dir=make_figures(root,summary)
    primary=summary['comparisons']['moving_coast']['one_step_nmse']
    result='PASS' if summary['primary_gate_passed'] else 'FAIL'
    excessive=[r for r in values['metrics'] if r['stopping_distance_m']>1.]
    excessive_train=sum(r['cube_id']<200 for r in excessive)
    excessive_test=len(excessive)-excessive_train
    lines=[
        '# Direct-force Plain–Oracle gate', '',
        f'**Primary gate: {result}.** Oracle’s relative improvement in moving-coast one-step error is '
        f'**{improvement(primary["relative_mean_improvement"])}** relative to Plain. '
        f'The five-seed mean ± sample SD is **{mean_sd(primary["plain"])}** for Plain and '
        f'**{mean_sd(primary["oracle"])}** for Oracle.', '',
        f'The required ≥20% reduction is **{"met" if summary["at_least_twenty_percent"] else "not met"}**; '
        f'the requirement `Oracle mean + SD < Plain mean − SD` is '
        f'**{"met" if summary["beyond_seed_spread"] else "not met"}**. '
        'The criterion was frozen before fitting and the held-out evaluation. '
        'No belief model was trained.', '',
        f'The moving-coast one-step mean gap, Plain minus Oracle, is **{fmt(primary["mean_gap"])}**. '
        f'The sum of the two seed SDs is **{fmt(primary["plain"]["sample_sd"]+primary["oracle"]["sample_sd"] if primary["defined"] else None)}**. '
        'The test checks both the relative improvement and this stricter spread separation.', '',
        f'![Moving-coast prediction gate]({figure_dir/"moving_coast_gate.png"})', '',
        '## Pilot and full collection', '',
        'The pusher was removed. A native horizontal external force is applied at the cube center of mass, '
        'with zero external torque, followed by exactly zero applied force during coast. This is an idealized '
        'force actuator on an infinite plane. It does not establish Panda-controller performance.', '',
        'The same cube identities remain independently sampled over 0.18–0.50 kg and material friction '
        '0.4–1.0; the floor material is 0.3. Per-cube mass, cube friction, and configured combined friction '
        'remain only in `data/cubes.json`. Each shove uses independent force magnitude 4–7 N and world '
        'direction, with random start XY and yaw. Duration is uniform on the 5 ms simulation grid '
        '**60, 65, …, 100 ms**, using a separate property-independent random stream.', '',
        '| Measurement | Same ten-cube pilot | Full 230-cube collection |',
        '|---|---:|---:|',
        f'| Episodes | {pilot["episodes"]:,} | {full["episodes"]:,} |',
        f'| Launch fraction, speed >2 cm/s at release | {pilot["launch_fraction"]:.2%} | {full["launch_fraction"]:.2%} |',
        f'| Moving fraction of retained coast steps | {pilot["moving_coast_fraction"]:.2%} | {full["moving_coast_fraction"]:.2%} |',
        f'| Maximum coast distance | {pilot["maximum_coast_distance_m"]:.4f} m | {full["maximum_coast_distance_m"]:.4f} m |',
        f'| Retained model transitions | {pilot["transitions"]:,} | {full["transitions"]:,} |',
        f'| Stationary transitions trimmed | {pilot["trimmed_transitions"]:,} | {full["trimmed_transitions"]:,} |',
        f'| Maximum time to confirmed stop onset | {pilot["maximum_time_to_stop_s"]:.3f} s | {full["maximum_time_to_stop_s"]:.3f} s |', '',
        'The pilot met the user’s continuation condition: at least 90% launches and maximum coast distance '
        'below 1 m. Collection therefore proceeded under that explicit conditional authorization. '
        f'**The full sample included {len(excessive)} episodes above 1 m '
        f'({excessive_train} training, {excessive_test} held-out), reaching '
        f'{full["maximum_coast_distance_m"]:.4f} m.** The pilot distance bound did not hold for every '
        'sampled object/action combination; no such episodes were discarded.', '',
        'All collected episodes reached confirmed rest. Raw coast recordings last 1.5 s, fixed before '
        'collection to allow stop confirmation across the allowed parameter range. Model episodes retain '
        'five 5 ms steps after stop onset, rounded up to a model boundary. Confirmation requires 100 ms '
        'continuously below 1 mm/s linear speed and 0.05 rad/s angular speed. No moving transitions were trimmed.', '',
        '## Held-out prediction results', '',
        'Each model was evaluated on the same 30 held-out cubes and their ten shoves. The table reports '
        'mean ± sample SD across five paired seeds. Within a seed, starts are averaged within episode, '
        'then episodes within cube, then cubes equally. Positive relative improvement favors Oracle. '
        '“Beyond spread” means the positive mean gap, Plain minus Oracle, exceeds the sum of the two seed SDs, equivalently '
        '`Oracle mean + SD < Plain mean − SD`.', '',
        '| Group | Metric | Plain mean ± SD | Oracle mean ± SD | Mean gap: Plain − Oracle | Relative improvement | Beyond spread? |',
        '|---|---|---:|---:|---:|---:|---|',
    ]
    for group in GROUPS:
        for metric in METRICS:
            row=summary['comparisons'][group][metric]
            label=GROUP_LABELS[group]
            if group=='moving_coast' and metric=='one_step_nmse':label='**Moving coast — gate**'
            lines.append(f'| {label} | {METRIC_LABELS[metric]} | {mean_sd(row["plain"])} | '
                         f'{mean_sd(row["oracle"])} | {fmt(row["mean_gap"])} | {improvement(row["relative_mean_improvement"])} | '
                         f'{"yes" if row["gap_exceeds_sum_of_seed_sds"] else "no"} |')
    lines += ['',f'![All three prediction groups and metrics]({figure_dir/"gate_metrics.png"})','',
              'One-step error is normalized MSE over all 15 cube-state coordinates. Ten-step rollout '
              'error averages normalized state MSE across all ten recursively predicted steps; the '
              'tenth-step-only diagnostic is also saved in the JSON. Push windows remain in push; '
              'moving-coast starts must be moving and their window remains in coast. All-step windows '
              'can cross release. Full raw traces provide targets beyond the trimmed endpoint.', '',
              'Stopping-position error is Euclidean XY error in meters. From every eligible retained '
              'start, predictions continue to the **preplanned release time plus 1.5 s**, then compare '
              'with the recorded confirmed-rest position. Neither the actual stop time nor the trimmed '
              'file length determines the prediction horizon. Rollouts use their own intermediate '
              'states and the preplanned applied-force sequence.', '']
    ten=summary['comparisons']['moving_coast']['rollout10_nmse']
    stopping=summary['comparisons']['moving_coast']['stopping_position_error_m']
    lines += ['**The primary gate is a one-step test.** For moving-coast starts, the independently reported '
              f'ten-step improvement is {improvement(ten["relative_mean_improvement"])}, and the stopping-position '
              f'improvement is {improvement(stopping["relative_mean_improvement"])}. '
              f'Oracle’s stopping-position error is {mean_sd(stopping["oracle"])} m. '
              'The stopping-position improvement also falls within the combined seed spread. '
              'No clipping, rotation projection, physics correction, or observed-state reset is applied during rollouts.', '']
    divergent=[]
    for group in GROUPS:
        for metric in METRICS:
            for kind in ['plain','oracle']:
                count=sum(records[(kind,s)]['groups'][group][metric]['nonfinite_starts'] for s in summary['seeds'])
                if count:divergent.append(f'{kind}/{group}/{metric}: {count:,} start–seed pairs')
    if divergent:
        lines += ['Nonfinite forecasts were not dropped from finite-only means. Corresponding aggregate '
                  'metrics remain undefined:', '']+['- '+v for v in divergent]+['']
    else:
        lines += ['No scored forecasts were nonfinite. Finite errors remain reported at their measured '
                  'size, including any large recursive errors.', '']
    lines += ['## Actions-only probes and trimming', '',
              'The actual pilot probe holds one cube out at a time and pools its out-of-fold predictions '
              'across all ten cubes. The fixed first 100 ms applied-force schedule includes the independently '
              'commanded pulse duration; it does not include observed stopping duration or trimmed length. '
              'A separate 200-train/30-test generator audit uses only preplanned commands, without dynamics.', '',
              '| Probe scope/input | Family | Mass R² ± seed SD | Combined-friction R² ± seed SD |',
              '|---|---|---:|---:|']
    probe_rows(lines,probe['pilot']['feature_sets']['commands_only']['aggregate'],'Pilot command schedule')
    probe_rows(lines,probe['generator']['aggregate'],'Planned commands, 200/30 cubes')
    baseline=probe['pilot']['fold_mean_baseline_r2']
    lines += ['',f'The ten-cube leave-one-out fold-mean baseline is R² {baseline["mass_kg"]:.4f} '
              f'for mass and {baseline["combined_friction"]:.4f} for combined friction. '
              'Pilot raw R² values are reported literally and are not called near zero. The wider '
              'planned-command audit has small negative R² and shows no useful predictive signal. '
              'Cube friction and combined friction are affine targets, so their R² values agree.', '',
              '| Forbidden side-channel diagnostic | Family | Mass R² ± seed SD | Combined-friction R² ± seed SD |',
              '|---|---|---:|---:|']
    for feature_set,label in [('retained_length_only','Trimmed length only'),('commands_plus_retained_length','Commands + trimmed length')]:
        probe_rows(lines,probe['pilot']['feature_sets'][feature_set]['aggregate'],label)
    lines += ['', '**Trimmed length reveals physical outcomes.** The positive mass-prediction scores '
              'above demonstrate why length, termination indicators, padding masks, stop times and IDs '
              'are forbidden model features. Plain and Oracle are per-transition MLPs receiving only '
              'the explicit state/action whitelist, with true mass and combined friction added only '
              'for Oracle. Scoring-group masks and observation counts are bookkeeping, not network inputs. '
              'The length diagnostic is not evidence that the randomized force generator used object labels.', '',
              '## Training, frozen evaluation, and limits', '',
              'Both families have three 256-unit ReLU hidden layers and predict normalized state change. '
              'Training uses episodes 0–7 of the 200 training cubes; episodes 8–9 select checkpoints. '
              'State/action statistics use only the training subset, and context statistics use training '
              'cube labels only. All models receive 10,000 Adam updates, batch size 512, with cosine '
              'learning rate 0.001→0.0001. Sampling first chooses an episode uniformly, then a retained '
              'transition uniformly. Paired families have identical minibatch-stream hashes.', '',
              'Validation checks every 500 updates select the lowest equal-cube/episode all-step one-step '
              'NMSE while still completing the entire training budget. All ten selected checkpoints and '
              'the held-out byte hashes were frozen before model scoring. The final evaluation ledger '
              'contains one completed invocation; test results did not choose a checkpoint or change the gate.', '',
              '| Seed | Plain selected update | Oracle selected update | Plain training time (s) | Oracle training time (s) |',
              '|---:|---:|---:|---:|---:|']
    for seed in summary['seeds']:
        p,o=training[('plain',seed)],training[('oracle',seed)]
        if p['minibatch_sha256']!=o['minibatch_sha256']:raise RuntimeError('Paired minibatch hash mismatch')
        lines.append(f'| {seed} | {p["selected_update"]} | {o["selected_update"]} | '
                     f'{p["elapsed_seconds"]:.2f} | {o["elapsed_seconds"]:.2f} |')
    total_train=sum(r['elapsed_seconds'] for r in training.values())
    total_eval=sum(r['elapsed_seconds'] for r in records.values())
    lines += ['',f'Summed recorded model-training time: **{total_train:.2f} s**. '
              f'Summed recorded test prediction time: **{total_eval:.2f} s**. '
              'These sums exclude collection, import/startup, data preparation, audits and report generation.', '',
              'Oracle receives true parameters. This experiment tests whether those parameters improve '
              'prediction in the direct-force simulator; estimating them from experience remains untested. '
              'The setup includes exact simulator state, a fixed floor material, and a finite sampled '
              'object/action distribution. Five-seed SD measures optimization variation, not hardware '
              'noise or a confidence interval over all possible cubes.', '',
              ('The frozen prediction gate passed. No belief model was started.' if summary['primary_gate_passed'] else
               'The frozen prediction gate failed. Work stops at this reported comparison; no test-driven tuning or belief model was started.'), '',
              '## Saved evidence', '']
    for name,label in [('results/gate_summary.json','Five-seed comparisons and gate decision'),
                       ('results/final_evaluation.json','Completed final-evaluation ledger'),
                       ('results/checkpoint_seal.json','Frozen checkpoint and source manifest'),
                       ('results/test_data_seal.json','Held-out trace byte hashes'),
                       ('results/model_protocol.json','Frozen training/evaluation protocol'),
                       ('results/pilot_audit.json','Pilot physics/data audit'),
                       ('results/full_audit.json','Full collection audit'),
                       ('results/final_audit.json','Independent final results and artifact audit'),
                       ('results/action_probe.json','Actions-only and length probe results'),
                       ('README_DIRECT.md','Direct-force experiment setup and reproduction')]:
        lines.append('- '+link(root,name,label))
    lines += ['', 'Every figure is generated only from the saved held-out seed metrics. Paired points '
              'represent actual saved runs; no synthetic samples, smoothing, or confidence intervals '
              'are introduced. PNG and SVG versions are saved under `results/figures/`.', '',
              'Rebuild this report after the completed evaluation:', '', '```bash',
              'cd /home/ubuntu/faliure',
              'PYTHONPATH=clean/vendor:. MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove_gate.report_direct',
              '```', '', 'Input SHA-256 hashes:', '']
    lines += [f'- `{path.relative_to(root)}`: `{sha(path)}`' for path in paths.values()]
    output=root/'RESULTS_DIRECT.md';output.write_text('\n'.join(lines)+'\n')
    manifest={str(path.relative_to(root)):sha(path) for path in paths.values()}
    manifest['report_direct.py']=sha(Path(__file__))
    (root/'results/report_direct_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(report=str(output),figures=str(figure_dir),primary_gate=result)))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();generate(args.root)
