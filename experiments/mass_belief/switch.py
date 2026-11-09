"""Continue saved five-grasp sequences with different cube physics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_adaptation.collection.collect import write_json
from robot_adaptation.estimators.mass_belief.model import MassBelief
from .collect import episode, make_env, policy_and_limits
from .evaluate import prior, write_csv
from .report import read
from .train import load_model, predict


def run_switch(base, output, mass_g, friction, start_seed):
    c = json.loads((base/'config.json').read_text())
    calibration = json.loads((base/'calibration.json').read_text())
    history = read(base/'trials.csv')
    count, trials = c['sequences'], c['trials']
    old_seeds = {int(row['seed']) for row in history}
    if old_seeds.intersection(range(start_seed, start_seed+count*trials)):
        raise ValueError('Choose fresh post-switch seeds')
    output.mkdir(parents=True, exist_ok=False)
    records = output/'episodes'
    records.mkdir()
    write_json(output/'config.json', dict(base=str(base.resolve()), mass_g=mass_g,
               friction=friction, start_seed=start_seed, sequences=count, post_trials=trials))
    model = load_model(base)
    policy, limits = policy_and_limits(c)
    torch.set_num_threads(1)
    env = make_env(dict(c, friction=friction), mass_g, limits)
    rows = []
    try:
        for seq in range(count):
            last, = [r for r in history if int(r['sequence']) == seq and
                     int(r['trial']) == trials and r['arm'] == 'B: filter carried']
            carry = MassBelief(float(last['estimate_g']), float(last['variance_g2']))
            fresh = prior(c)  # Comparator given the exact switch time.
            for t in range(trials):
                seed = start_seed + seq*trials + t
                record = episode(env, policy, seed, c['horizon'], c['probe_g'])
                materials = env.physics_readback()['object_materials']
                assert np.isclose(record['mass_g'], mass_g)
                for material in materials:
                    for key in ('static_friction', 'dynamic_friction'):
                        assert np.isclose(material[key], friction)
                np.savez_compressed(records/f'{seq:02}_{t+1}.npz', **record,
                                    static_friction=materials[0]['static_friction'],
                                    dynamic_friction=materials[0]['dynamic_friction'])
                estimates = predict(model, record['x'][None])[0]
                reset = prior(c)
                for step, observation in calibration.items():
                    for belief in (carry, fresh, reset):
                        belief.update(float(estimates[int(step)]), observation)
                for arm, estimate, var in [('Raw', float(estimates[-1]), None),
                        ('Carry', carry.mean, carry.variance), ('Reset every trial', reset.mean, reset.variance),
                        ('Reset once at switch', fresh.mean, fresh.variance)]:
                    rows.append(dict(sequence=seq, trial=trials+t+1, seed=seed, arm=arm,
                                     mass_g=record['mass_g'], estimate_g=estimate, variance_g2=var,
                                     error_g=abs(estimate-record['mass_g'])))
    finally:
        env.close()
    write_csv(output/'trials.csv', rows)
    plot_results(base, output, rows, history, c, mass_g, friction)


def plot_results(base, output, rows, history, c, mass_g, friction):
    n, k = c['sequences'], c['trials']
    draws = np.random.default_rng(c['seed']).integers(n, size=(10000,n))
    arms = {'Carry': 'B: filter carried', 'Reset every trial': 'Filter reset',
            'Raw': 'A: estimator reset', 'Reset once at switch': 'B: filter carried'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    times = np.arange(1, 2*k+1)
    summaries = {}
    for ax, pre_key, post_key, label in zip(axes, ['estimate_g', 'abs_error_g'],
            ['estimate_g', 'error_g'], ['Mean estimated mass (g)', 'Mean absolute error (g)']):
        for arm, previous in arms.items():
            before = np.array([float(r[pre_key]) for r in history if r['arm']==previous]).reshape(n,k)
            after = np.array([r[post_key] for r in rows if r['arm']==arm]).reshape(n,k)
            values = np.column_stack((before, after))
            mean = values.mean(axis=0)
            lo, hi = np.quantile(values[draws].mean(1), [.025,.975], axis=0)
            summaries.setdefault(arm, {})[post_key] = dict(mean=mean.tolist(), lower=lo.tolist(), upper=hi.tolist())
            show = slice(k, None) if arm=='Reset once at switch' else slice(None)
            line, = ax.plot(times[show], mean[show], marker='o', ms=3, label=arm)
            ax.fill_between(times[show], lo[show], hi[show], color=line.get_color(), alpha=.10)
        ax.axvline(k+.5, color='0.4', ls=':')
        ax.set(xlabel='Grasp attempt', ylabel=label, xticks=times)
    axes[0].step(times, [c['heavy_g']]*k+[mass_g]*k, where='mid', color='black', ls='--', label='True mass')
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Cube switch after grasp {k}: {c['heavy_g']:g} g → {mass_g:g} g, friction {c['friction']:g} → {friction:g}")
    fig.savefig(output/'switch.png', dpi=180)
    fig.savefig(output/'switch.pdf')
    plt.close(fig)
    write_json(output/'summary.json', summaries)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('base', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--mass-g', default=80., type=float)
    p.add_argument('--friction', default=1., type=float)
    p.add_argument('--start-seed', default=450000, type=int)
    a = p.parse_args()
    run_switch(a.base, a.output, a.mass_g, a.friction, a.start_seed)
