"""Publication-sized figures and plain-language metrics from saved outcomes."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from robot_adaptation.collection.collect import write_json


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def report(run):
    c = json.loads((run/'config.json').read_text())
    fit = json.loads((run/'fit.json').read_text())
    rows = read(run/'trials.csv')
    arms = list(dict.fromkeys(r['arm'] for r in rows))
    n, k = c['sequences'], c['trials']
    draws = np.random.default_rng(c['seed']+1).integers(n,size=(10000,n))

    def matrix(arm, key):
        entries = sorted((r for r in rows if r['arm']==arm),key=lambda r:(int(r['sequence']),int(r['trial'])))
        return np.array([float(r[key]) for r in entries]).reshape(n,k)

    def stats(values):
        av = np.mean(values,axis=0)
        lo,hi = np.quantile(values[draws].mean(1),[.025,.975],axis=0)
        return np.stack([av,lo,hi])

    def line(ax, x, ys, label, color):
        mean,lo,hi = stats(ys)
        ax.plot(x,mean,marker='o' if len(x)==k else None,label=label,color=color)
        ax.fill_between(x,lo,hi,color=color,alpha=.14)

    colors = ['#275990','#b24b29','#498272']
    fig,(left,right) = plt.subplots(1,2,figsize=(10,4.4), layout='constrained')
    numbers = np.arange(1,k+1)
    metrics = {}
    for arm,color in zip(arms,colors):
        err = matrix(arm,'abs_error_g')
        line(left,numbers,err,arm,color)
        metrics[arm]={'mae_g_mean_lo_hi': stats(err).T.tolist()}
        if arm!=arms[0]:
            variance = matrix(arm,'variance_g2')
            line(right,numbers,variance,arm,color)
            metrics[arm]['variance_g2_mean_lo_hi'] = stats(variance).T.tolist()
            metrics[arm]['nominal_95_coverage'] = np.mean(err<=1.96*np.sqrt(variance),axis=0).tolist()
    for ax in [left,right]:
        ax.set(xlabel='Trial with the same cube',xticks=numbers,ylim=(0,None))
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    left.set_ylabel('Mean absolute mass error (g)')
    right.set_ylabel('Mean posterior variance (g²)')
    fig.suptitle(f'{c["heavy_g"]} g cube · {n} paired five-trial sequences · 95% seed bootstrap intervals')
    fig.savefig(run/'mass_across_trials.png',dpi=180)
    fig.savefig(run/'mass_across_trials.pdf')
    plt.close(fig)
    # All five episodes stay together when resampling a sequence.
    steps = read(run/'steps.csv')
    h = c['horizon']
    errors = np.array([abs(float(row['raw_g'])-float(row['truth_g'])) for row in steps]).reshape(n,k,h)
    fig,ax=plt.subplots(figsize=(7.2,3.8),layout='constrained')
    line(ax,np.arange(1,h+1),errors.mean(1),'Raw estimator',colors[0])
    ax.axhline(c['settle_tolerance_g'],linestyle='--',color='black',label='Settling tolerance')
    ax.set(xlabel='Control steps since reset',ylabel='Mean absolute error (g)',title='Mass estimation during the lift')
    ax.legend(); ax.grid(alpha=.18)
    fig.savefig(run/'within_trial.png',dpi=180); plt.close(fig)
    settled = []
    for curve in errors.reshape(-1,h):
        bad = np.flatnonzero(curve>c['settle_tolerance_g'])
        first = int(bad[-1]+2) if len(bad) else 1
        if h-first+1>=c['settle_consecutive_steps']:
            settled.append(first)
    metrics['settling'] = dict(settled=len(settled),episodes=n*k,median_step=float(np.median(settled)) if settled else None)
    diffs = {}
    for arm in [arms[0],arms[2]]:
        delta = matrix(arms[1],'abs_error_g')-matrix(arm,'abs_error_g')
        diffs[arm]=stats(delta).T.tolist()
    metrics['carried_minus_comparator_mae_g'] = diffs
    carried_error = matrix(arms[1],'abs_error_g')
    metrics['carried_mae_trial_five_minus_one_g'] = stats(carried_error[:,-1] - carried_error[:,0]).tolist()
    side = read(run/'side_check.csv') if (run/'side_check.csv').exists() else []
    outcomes = {mode:sum(int(row['success']) for row in side if row['mode']==mode) for mode in ['true','belief']} if side else {}
    metrics['side_success_counts']=outcomes
    metrics['fixed_probe_successes'] = sum(int(row['fixed_probe_success']) for row in rows if row['arm']==arms[0])
    if side:
        side_scores = {mode: np.array([int(row['success']) for row in side if row['mode']==mode]).reshape(n,k)
                       for mode in ['belief','true']}
        metrics['side_belief_minus_true_success_pp'] = stats(100*(side_scores['belief']-side_scores['true']).mean(1)).tolist()
    write_json(run/'summary.json',metrics)
    def interval(row):
        return f'{row[0]:.1f} [{row[1]:.1f}, {row[2]:.1f}]'
    lines=['# Mass belief across five lifts','',
           f'{n} independent sequences of five fresh starts at {c["heavy_g"]} g. Object friction {c["friction"]}. '
           'All three analyses replay identical trajectories driven by the fixed 40 g policy input. '
           'Each trajectory is a new reset of the same cube physics. Each sequence begins with the same prior.',
           '', '| End of trial | Raw reset MAE (g) | Carried filter MAE (g) | Reset filter MAE (g) | Carried variance (g²) |',
           '|---:|---:|---:|---:|---:|']
    for i in range(k):
        values=[interval(metrics[a]['mae_g_mean_lo_hi'][i]) for a in arms]
        values.append(interval(metrics[arms[1]]['variance_g2_mean_lo_hi'][i]))
        lines.append('| '+' | '.join([str(i+1),*values])+' |')
    lines += ['', 'Brackets give percentile 95% bootstrap intervals across entire seed sequences. '
              'The raw estimator produces a point estimate. Posterior variance exists only for filtered arms.',
              '',f'Trial-five paired MAE difference, carry minus raw: {interval(diffs[arms[0]][-1])} g. '
              f'Carry minus reset-filter: {interval(diffs[arms[2]][-1])} g. '
              f'Within the carried arm, trial five minus trial one was {interval(metrics["carried_mae_trial_five_minus_one_g"])} g.',
              '', f'The fixed 40 g probe succeeded on {metrics["fixed_probe_successes"]}/{n*k} final steps. All attempts contributed evidence.',
              '',f'Within trial, {len(settled)}/{n*k} episodes entered ±{c["settle_tolerance_g"]} g and stayed there through the end for at least '
              f'{c["settle_consecutive_steps"]} steps. Median first settled step among these episodes: {metrics["settling"]["median_step"]}.',
              '',f'Nominal 95% posterior intervals contained the true mass in {metrics[arms[1]]["nominal_95_coverage"][-1]:.0%} of sequences on trial five. '
              'Every sequence uses identical measurement times and calibrated noise. The resulting variance curves are deterministic and their bootstrap intervals have zero width. The model assumes constant mass. Coverage here concerns one object mass.',
              '', f'Training used {fit["fresh_train"]} randomized simulation episodes plus {len(fit["historical_train_files"])} nominal-geometry, mass-only archived episodes. '
              f'Model selection used {fit["validation"]} independent randomized episodes. Best validation window-endpoint MAE was {fit["best_validation_mae_g"]:.1f} g. '
              f'A further {c["calibration_episodes"]} independent episodes fitted the observation model for the filter.',
              '', 'The estimator predicts grams at every step from a causal window of joint positions/velocities, finger compression, and past applied commands. '
              'The filter assimilates predictions at the ends of non-overlapping 25-step windows. Each endpoint has its own calibrated observation slope, offset and residual variance. '
              'Observation noise is inflated fourfold to discount correlation between windows of a single lift. This is a modeling choice. '
              'Finger compression is a simulated contact-force proxy. Results are conditional on one heavy mass, one policy and fixed friction.',
              '', 'The side check drives a fresh, matched simulator rollout with the current belief or true mass at every policy step.']
    if outcomes:
        lines.append(f'Belief feedback succeeded on {outcomes["belief"]}/{n*k} final steps and true mass on {outcomes["true"]}/{n*k}. '
                     f'The paired difference in success was {interval(metrics["side_belief_minus_true_success_pp"])} percentage points.')
    else:
        lines.append('Side-check outcomes pending.')
    lines += ['', '![Absolute error and posterior variance](mass_across_trials.png)','', '![Within-episode absolute error](within_trial.png)']
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    report(p.parse_args().run)
