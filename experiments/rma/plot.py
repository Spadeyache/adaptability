"""Offline report: consumes saved records only, never simulator truth as predictions."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(output):
    out = Path(output)
    rows = list(csv.DictReader((out/'episodes.csv').open()))
    if not rows:
        raise ValueError('No completed episodes to plot')
    modes = sorted({r['mode'] for r in rows})
    masses = sorted({float(r['mass_kg']) for r in rows})
    frictions = sorted({float(r['friction']) for r in rows})
    fig, axes = plt.subplots(1, len(modes), figsize=(6*len(modes), 4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        grid = np.full((len(frictions), len(masses)), np.nan)
        for i, friction in enumerate(frictions):
            for j, mass in enumerate(masses):
                values = [int(r['success']) for r in rows if r['mode']==mode and float(r['mass_kg'])==mass and float(r['friction'])==friction]
                if values:
                    grid[i,j] = np.mean(values)
                    ax.text(j, i, f'{sum(values)}/{len(values)}', ha='center', va='center', color='black')
        im = ax.imshow(grid, vmin=0, vmax=1, cmap='YlGn', origin='lower', aspect='auto')
        ax.set(xticks=range(len(masses)), xticklabels=[f'{m*1000:g}' for m in masses],
               yticks=range(len(frictions)), yticklabels=[f'{x:g}' for x in frictions],
               xlabel='Mass (g)', ylabel='Object friction', title=mode)
        fig.colorbar(im, ax=ax, label='Final-step success rate')
    fig.tight_layout(); fig.savefig(out/'success_maps.png', dpi=160); plt.close(fig)
    conditions = [x for x in ['in_range','mass_ood','friction_ood','combined_ood'] if any(r['condition']==x for r in rows)]
    fig, ax = plt.subplots(figsize=(8,4))
    rng = np.random.default_rng(0)
    summary = []
    for mi, mode in enumerate(modes):
        means, errors = [], []
        for cond in conditions:
            subset = [r for r in rows if r['mode']==mode and r['condition']==cond]
            # Bootstrap paired starting-seed blocks, retaining all physical cells.
            seeds = sorted({r['seed'] for r in subset})
            blocks = np.array([np.mean([int(r['success']) for r in subset if r['seed']==s]) for s in seeds])
            mean = blocks.mean()
            lo, hi = np.quantile(rng.choice(blocks, (2000,len(blocks)), replace=True).mean(1), [.025,.975]) if len(seeds)>1 else (np.nan,np.nan)
            means.append(mean); errors.append((mean-lo,hi-mean))
            summary.append(dict(mode=mode, condition=cond, episodes=len(subset), success_rate=mean,
                                interval=None if len(seeds)<2 else [lo,hi]))
        ax.bar(np.arange(len(conditions))+mi*.35, means, .35, label=mode,
               yerr=np.nan_to_num(np.array(errors).T), capsize=3)
    ax.set(xticks=np.arange(len(conditions))+.175*(len(modes)-1), xticklabels=conditions,
           ylim=(0,1.05), ylabel='Final-step success rate', title='Seed-block 95% bootstrap intervals; absent for one seed')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'condition_success.png', dpi=160); plt.close(fig)
    links = []
    for row in rows:
        trace = np.load(out/row['trace'])
        fig, axes = plt.subplots(3,1,figsize=(8,7), sharex=True)
        axes[0].plot(trace['clearance_m'], label='Cube clearance'); axes[0].plot(trace['goal_distance_m'], label='Goal distance')
        axes[0].set_ylabel('Metres'); axes[0].legend()
        axes[1].plot(trace['z_hat']); axes[1].set_ylabel('Latent components')
        axes[2].plot(np.sqrt(((trace['z_hat']-trace['z'])**2).mean(-1)), label='Latent RMSE')
        axes[2].plot(trace['success'], label='Success'); axes[2].legend(); axes[2].set_xlabel('Step')
        fig.suptitle(f"{row['mode']}: {float(row['mass_kg'])*1000:g} g, friction {row['friction']}, seed {row['seed']}")
        name = f"episode_{int(row['episode']):05}.png"
        fig.tight_layout(); fig.savefig(out/name, dpi=100); plt.close(fig)
        video = f" · <a href='{html.escape(row['video'])}'>Replay GIF</a>" if row['video'] else ''
        links.append(f"<li><a href='{name}'>Episode {row['episode']} · {row['mode']} · success {row['success']}</a>{video}</li>")
    manifest = json.loads((out/'manifest.json').read_text())
    label = 'INTEGRATION CHECK — not evidence of learned performance' if manifest['smoke'] else 'RMA baseline evaluation'
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    (out/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>RMA evaluation</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#24313c}}img{{max-width:100%}}a{{color:#176b8e}}</style>
<h1>{label}</h1><p>Actual saved episodes. Training range: 20–120 g, friction 0.3–1.0. Fixed cube geometry; simulated state sensing.</p>
<img src="success_maps.png"><img src="condition_success.png"><h2>Episode traces</h2><ul>{''.join(links)}</ul>''')
    lines = [f'# {label}', '', 'Final-step custom held-goal success; sustained success is also logged.', '',
             '| Mode | Condition | Episodes | Success |', '|---|---|---:|---:|']
    lines += [f"| {r['mode']} | {r['condition']} | {r['episodes']} | {r['success_rate']:.1%} |" for r in summary]
    lines += ['', 'Uncertainty: bootstrap starting-seed blocks; conditional on the specified physics grid. One seed cannot estimate uncertainty.',
              'No comparison to the paper’s task-specific percentages is justified. No novelty detector is trained here.',
              'Oracle failure does not prove physical impossibility. Latent error is a diagnostic, not a novelty score.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('output'); report(p.parse_args().output)
