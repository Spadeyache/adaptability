"""Render brief figures from saved results only; never load/evaluate a model.

Run from the repository: MPLCONFIGDIR=/tmp/brief-mpl-primary .venv/bin/python results/brief/primary_figures.py
"""
from pathlib import Path
import csv
import json
import hashlib
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/brief-mpl-primary')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
RUN = ROOT / 'clean/shove_gate/push_latent'
SAVED = RUN / 'results'
COLORS = dict(plain='#64748b', fitted_z='#00897b', oracle='#7c3aed', mean_z='#94a3b8', wrong_z='#b8a6a0')
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 14,
    'axes.labelsize': 16, 'axes.titlesize': 17, 'xtick.labelsize': 14,
    'ytick.labelsize': 14, 'legend.fontsize': 12, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.edgecolor': '#64748b',
    'text.color': '#172b3a', 'axes.labelcolor': '#172b3a',
    'xtick.color': '#334155', 'ytick.color': '#334155',
    'svg.fonttype': 'none', 'savefig.facecolor': 'white',
})
sources = set()

def read(path):
    sources.add(path)
    return json.loads(path.read_text())

def dump_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def finish(fig, name):
    for ax in fig.axes:
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e2e8f0', linewidth=.8)
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=220, bbox_inches='tight', pad_inches=.16)
    plt.close(fig)

summary = read(SAVED / 'summary.json')
seeds = summary['seeds']
metric = 'horizontal_velocity_delta_nmse'

# B: both independently saved actions-only probe families, including negative R².
probe = read(SAVED / 'action_probe.json')['generator']
families = [('extra_trees', 'Tree predictor', '#64748b'),
            ('polynomial_ridge', 'Polynomial predictor', '#b8a6a0')]
targets = [('mass_kg', 'Mass'), ('combined_friction', 'Cube–floor friction')]
fig, ax = plt.subplots(figsize=(8.0, 5.6))
fig.subplots_adjust(bottom=.27, top=.84)
rows = []
for k, (family, label, color) in enumerate(families):
    vals = [probe['aggregate'][family][key]['mean_r2'] for key, _ in targets]
    sds = [probe['aggregate'][family][key]['seed_std_r2'] for key, _ in targets]
    x = np.arange(len(targets)) + (k - .5) * .32
    ax.bar(x, vals, width=.28, color=color, label=label, yerr=sds,
           error_kw=dict(ecolor='#172b3a', capsize=5, linewidth=1.5), zorder=3)
    for xi, value, sd in zip(x, vals, sds):
        ax.text(xi, value-sd-.008, f'{value:.3f}', ha='center', va='top', fontsize=13)
    for key, _ in targets:
        for rec in probe['records']:
            if rec['family'] == family:
                rows.append(dict(family=family, target=key, seed=rec['seed'], r2=rec['r2'][key],
                    source_file=str(SAVED/'action_probe.json'), source_key=f'generator.records[family={family},seed={rec["seed"]}].r2.{key}'))
ax.axhline(0, color='#172b3a', linewidth=1.3)
ax.text(1.47, .006, 'Zero: constant-mean reference', ha='right', va='bottom', fontsize=11)
ax.set(xticks=[0,1], xticklabels=[x[1] for x in targets], ylabel='Prediction R² (unitless)', ylim=(-.16,.055), xlim=(-.52,1.52))
fig.legend(loc='upper center', bbox_to_anchor=(.55,.99), ncol=2, frameon=False)
fig.text(.5,.08, 'Actions alone: no useful prediction in these checks', ha='center', fontsize=14)
fig.text(.5,.025, 'Bars: mean ± seed SD; negative scores are retained.', ha='center', fontsize=11, color='#475569')
finish(fig, 'fig_leak'); dump_csv('fig_leak_data.csv', rows)

# C: paired-seed plain/oracle summaries.
def model_stat(model):
    return summary['by_model'][model]['push'][metric]

def prediction_bars(ax, models, labels, ceiling):
    means = [model_stat(m)['mean'] for m in models]
    sds = [model_stat(m)['sample_sd'] for m in models]
    ax.bar(np.arange(len(models)), means, width=.62, color=[COLORS[m] for m in models],
           yerr=sds, error_kw=dict(ecolor='#172b3a', capsize=5, linewidth=1.5), zorder=3)
    for i, (v, sd) in enumerate(zip(means,sds)):
        ax.text(i,v+sd+.011,f'{v:.3f}',ha='center',fontsize=14)
    ax.set(xticks=np.arange(len(models)), xticklabels=labels,
           ylabel='Push prediction error\n(normalized; unitless)', ylim=(0,ceiling))

fig, ax = plt.subplots(figsize=(6.6,5.7))
fig.subplots_adjust(bottom=.25, top=.87)
prediction_bars(ax, ['plain','oracle'], ['Plain','Oracle\ntrue mass + friction'], .28)
gain = summary['c1']['relative_mean_improvement']*100
fig.text(.5,.95,f'{gain:.2f}% lower error with true properties',ha='center',fontsize=17)
fig.text(.5,.065,'Horizontal velocity change during push; lower is better.',ha='center',fontsize=11)
fig.text(.5,.018,'30 unseen cubes · 5 training seeds · error bars: seed SD',ha='center',fontsize=11,color='#475569')
finish(fig,'fig_physics')
barrows=[]
for model in ['plain','mean_z','wrong_z','fitted_z','oracle']:
    st=model_stat(model)
    assert np.isclose(np.mean(st['seed_values']),st['mean'])
    assert np.isclose(np.std(st['seed_values'],ddof=1),st['sample_sd'])
    for seed,value in zip(seeds,st['seed_values']):
        barrows.append(dict(model=model, seed=seed, push_velocity_change_nmse=value,
            source_file=str(SAVED/'summary.json'), source_key=f'by_model.{model}.push.{metric}.seed_values'))
dump_csv('fig_physics_data.csv',[r for r in barrows if r['model'] in ['plain','oracle']])

# D: specificity controls plus independent support-prefix fits.
fig, axes = plt.subplots(1,2,figsize=(14.7,5.9),gridspec_kw={'width_ratios':[1.5,1]})
fig.subplots_adjust(left=.07,right=.98,bottom=.26,top=.86,wspace=.32)
prediction_bars(axes[0], ['plain','mean_z','wrong_z','fitted_z','oracle'],
    ['Plain','Average\nbelief','Another\ncube’s belief','Own fitted\nbelief','Oracle'],.34)
axes[0].set_title('Fit the belief; freeze the network',pad=18)
x=np.arange(1,6)
ys=np.array([summary['prefix_curve'][str(k)]['mean'] for k in x])
errs=np.array([summary['prefix_curve'][str(k)]['sample_sd'] for k in x])
ax=axes[1]
ax.errorbar(x,ys,yerr=errs,color=COLORS['fitted_z'],linewidth=2.8,marker='o',markersize=7,capsize=5,label='Own fitted belief')
ax.axhline(model_stat('plain')['mean'],color=COLORS['plain'],linestyle='--',linewidth=1.6,label='Plain')
ax.set(xlabel='Shoves used to fit the belief',xticks=x,ylim=(.11,.27),xlim=(.7,5.3),ylabel='Push prediction error\n(normalized; unitless)')
ax.set_title('More evidence from the same cube',pad=18)
ax.legend(loc='upper right',frameon=False,fontsize=11)
for xi in [0,4]:
    ax.annotate(f'{ys[xi]:.3f}', (x[xi],ys[xi]), xytext=(0,-22),textcoords='offset points',ha='center',fontsize=12,color=COLORS['fitted_z'])
fig.text(.5,.075,'All scores use the same five other shoves on each unseen cube; lower is better.',ha='center',fontsize=12)
fig.text(.5,.025,'30 unseen cubes · 5 training seeds · error bars: seed SD · each prefix fit starts again at the average belief',ha='center',fontsize=11,color='#475569')
finish(fig,'fig_belief'); dump_csv('fig_belief_bars_data.csv',barrows)
prefixrows=[]
for count,st in summary['prefix_curve'].items():
    for seed,value in zip(seeds,st['seed_values']):
        prefixrows.append(dict(support_shoves=int(count),seed=seed,push_velocity_change_nmse=value,
             source_file=str(SAVED/'summary.json'),source_key=f'prefix_curve.{count}.seed_values'))
dump_csv('fig_belief_curve_data.csv',prefixrows)

# E: every saved seed's predictions, without any ensemble or new fitted decoder.
cube_data=read(RUN/'data/cubes.json')
mass={c['cube_id']:c['mass_kg'] for c in cube_data['cubes']}
mass_runs=[read(SAVED/f'mass_scores_seed_{seed}.json') for seed in seeds]
fig,axes=plt.subplots(1,5,figsize=(17.2,5.1),sharex=True,sharey=True)
fig.subplots_adjust(left=.06,right=.99,bottom=.25,top=.73,wspace=.13)
massrows=[]
for ax,run in zip(axes,mass_runs):
    pts=run['predicted_mass_by_cube']
    true=np.array([mass[p['cube_id']] for p in pts]); pred=np.array([p['prediction_kg'] for p in pts])
    recomputed=1-np.sum((true-pred)**2)/np.sum((true-true.mean())**2)
    assert np.isclose(recomputed,run['r2']), 'Saved mass predictions/R² disagree'
    ax.plot([.14,.54],[.14,.54],color='#94a3b8',linestyle='--',linewidth=1.3)
    ax.scatter(true,pred,s=40,color=COLORS['fitted_z'],alpha=.85,edgecolors='white',linewidth=.45,zorder=3)
    ax.set(xlim=(.14,.54),ylim=(.14,.54),xticks=[.2,.3,.4,.5],yticks=[.2,.3,.4,.5],aspect='equal')
    ax.set_title(f'Seed {run["seed"]}\nR² = {run["r2"]:.3f}',fontsize=15,pad=12)
    for p in pts:
        massrows.append(dict(seed=run['seed'],cube_id=p['cube_id'],true_mass_kg=mass[p['cube_id']],
            predicted_mass_kg=p['prediction_kg'],seed_r2=run['r2'],prediction_source=str(SAVED/f'mass_scores_seed_{run["seed"]}.json'),
            prediction_key=f'predicted_mass_by_cube[cube_id={p["cube_id"]}].prediction_kg',label_source=str(RUN/'data/cubes.json')))
axes[0].set_ylabel('Predicted mass (kg)')
r2=summary['mass_readout']['r2']
fig.text(.5,.94,f'Mass readout on unseen cubes: R² = {r2["mean"]:.3f} ± {r2["sample_sd"]:.3f}',ha='center',fontsize=19)
fig.text(.5,.86,'Mean ± SD across the five separately trained models below',ha='center',fontsize=13,color='#475569')
fig.text(.5,.16,'True mass (kg)',ha='center',fontsize=16)
fig.text(.5,.075,'Each panel: the same 30 unseen cubes; dashed line: exact prediction.',ha='center',fontsize=12)
fig.text(.5,.022,'Belief fitting uses no mass labels; the separate straight-line decoder is trained with training-cube mass labels.',ha='center',fontsize=12,color='#475569')
finish(fig,'fig_mass');dump_csv('fig_mass_data.csv',massrows)

manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
(OUT/'primary_sources_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'primary_sources.md').write_text('''# Primary-figure provenance

All figures are regenerated by `primary_figures.py` from saved result files only. No checkpoint is loaded, no prediction is recomputed, and no decoder is fitted. Arithmetic is limited to plotting and checking saved summary statistics against saved values. Source SHA-256 hashes are in `primary_sources_manifest.json`.

| Figure | Saved source and key | Scope |
|---|---|---|
| fig_leak | `clean/shove_gate/push_latent/results/action_probe.json`, `generator.records`, `generator.aggregate` | Both tree and polynomial actions-only predictors; mass and combined cube–floor friction; all five seeds; negative R² retained. |
| fig_physics | `clean/shove_gate/push_latent/results/summary.json`, `by_model.{plain,oracle}.push.horizontal_velocity_delta_nmse` and `c1.relative_mean_improvement` | Push steps of test shoves 5–9, cube IDs 260–289; mean and sample SD across five seeds. Oracle receives both mass and combined friction. |
| fig_belief | Same `summary.json`, `by_model.{plain,mean_z,wrong_z,fitted_z,oracle}.push.horizontal_velocity_delta_nmse` and `prefix_curve` | Bars use five support shoves; every prefix is separately fitted from the same initial mean belief, then scored on the same query shoves. |
| fig_mass | `clean/shove_gate/push_latent/results/mass_scores_seed_{101,211,307,401,503}.json`, `predicted_mass_by_cube`, `r2`; true masses from `clean/shove_gate/push_latent/data/cubes.json` | Five panels preserve each seed's result; annotation is mean and sample SD of seed-specific R², not an ensemble R². All 30 test cubes appear per seed. |

Human-readable line references: `clean/shove_gate/push_latent/RESULTS.md:7` (population), `:11` (model comparisons), `:47` (prefix curve), `:57` (restart and wrong-cube control), `:61` (supervised mass decoder), `:65` (mass score), `:76` (normalization), `:287` (actions-only check). All normalized prediction errors are unitless; they measure horizontal velocity change during push. They are not whole-state or stopping-position errors. Probe R² zero is the constant test-mean reference; negative scores do not establish a mathematical guarantee of independence.

CSV companions contain individual seed values and source keys. `fig_mass_data.csv` includes true and predicted mass in kg for every saved point. There are no missing values among these requested figures.
''')
print('Saved figures B–E, per-seed CSVs, source notes, and source hashes to', OUT)
