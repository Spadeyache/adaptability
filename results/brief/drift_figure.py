"""Render saved Step-3 scores only; no model imports, predictions, or evaluation."""
from pathlib import Path
import csv
import hashlib
import json
import os
import statistics

os.environ.setdefault('MPLCONFIGDIR', '/tmp/brief-mpl-drift')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SRC = ROOT / 'clean/shove_gate/adaptation_compare/results'
SEEDS = [101, 211, 307, 401, 503]
COLORS = {1e-5: '#2563eb', 1e-4: '#d97706'}
GRAY = '#64748b'
SOURCES = {}
RECORDS = []


def saved(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_csv(path):
    with saved(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows, 2):
        r['_line'] = i
    return rows


def stats(v):
    return statistics.mean(v), statistics.stdev(v)


def values(r):
    return [float(r[f'seed_{s}']) for s in SEEDS]


def record(panel, method, rate, visit, kind, vals, source):
    m, sd = stats(vals)
    r = dict(panel=panel, method=method, learning_rate=rate, visit=visit,
             observation=kind, mean=m, seed_sd=sd, source=source)
    r.update({f'seed_{s}': v for s, v in zip(SEEDS, vals)})
    RECORDS.append(r)
    return m, sd


def source_ref(name, rows):
    return str((SRC / name).relative_to(ROOT)) + ':lines=' + ','.join(str(r['_line']) for r in rows)


def band(ax, x, perseed, color, linestyle='-', label=None, marker=None):
    pairs = [stats(v) for v in perseed]
    means, sds = zip(*pairs)
    ax.plot(x, means, color=color, lw=2.3, ls=linestyle, marker=marker, ms=4, label=label)
    ax.fill_between(x, [m-s for m,s in pairs], [m+s for m,s in pairs], color=color, alpha=.12, lw=0)
    return means, sds


def main():
    current = read_csv(SRC / 'current_cube_sequence_scores.csv')
    sequence = read_csv(SRC / 'sequence_scores.csv')
    prefixes = read_csv(SRC / 'prefix_scores.csv')
    baselines = {s: json.loads(saved(SRC / f'baselines/plain_seed{s}.json').read_text()) for s in SEEDS}
    plain_by_cube = {}
    for s, b in baselines.items():
        rows = b['query']['aggregates']['push_velocity_nmse']['per_cube']
        assert [r['cube_id'] for r in rows] == list(range(290, 320))
        assert all(r['episodes'] == 5 for r in rows)
        plain_by_cube[s] = [r['mean'] for r in rows]
    plain_training = [baselines[s]['training_validation']['scores']['composite'] for s in SEEDS]
    baseline_source = 'clean/shove_gate/adaptation_compare/results/baselines/plain_seed{101,211,307,401,503}.json'

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':13,
        'axes.labelsize':14, 'axes.titlesize':17, 'xtick.labelsize':12,
        'ytick.labelsize':12, 'svg.fonttype':'none', 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.edgecolor':'#cbd5e1', 'axes.labelcolor':'#334155',
        'text.color':'#0f172a', 'xtick.color':'#475569', 'ytick.color':'#475569'})
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.5))
    fig.subplots_adjust(left=.075, right=.98, bottom=.32, top=.84, wspace=.26)
    visits = list(range(1, 31))
    plain_cumulative = [[statistics.mean(plain_by_cube[s][:n]) for s in SEEDS] for n in visits]
    for n, v in zip(visits, plain_cumulative):
        record('new_cubes', 'Plain', '', n, 'cumulative mean through visit', v,
               baseline_source + f':query.aggregates.push_velocity_nmse.per_cube[first {n}]')
    band(axes[0], visits, plain_cumulative, GRAY, '--')
    record('training_cubes', 'Plain', '', 0, 'frozen baseline for all visits', plain_training,
           baseline_source + ':training_validation.scores.composite')
    band(axes[1], [0, 30], [plain_training, plain_training], GRAY, '--')

    for rate in COLORS:
        color = COLORS[rate]
        rows = sorted([r for r in current if r['method']=='B' and
            float(r['learning_rate'])==rate and r['metric']=='push_velocity_nmse'],
            key=lambda r: int(r['total_visits']))
        assert [int(r['total_visits']) for r in rows] == visits
        assert [int(r['cube_id']) for r in rows] == list(range(290, 320))
        cums = [[statistics.mean(float(r[f'seed_{s}']) for r in rows[:n]) for s in SEEDS] for n in visits]
        for n, vals in zip(visits, cums):
            record('new_cubes', 'Fine-tune', rate, n, 'cumulative mean through visit', vals,
                   source_ref('current_cube_sequence_scores.csv', rows[:n]))
        band(axes[0], visits, cums, color)
        rr = [r for r in prefixes if r['method']=='B' and float(r['learning_rate'])==rate
              and r['support_shoves']=='5' and r['metric']=='push_velocity_nmse']
        assert len(rr)==1
        m, sd = record('new_cubes', 'Fine-tune', rate, '', 'independent reset per cube, mean over all 30',
                       values(rr[0]), source_ref('prefix_scores.csv', rr))
        axes[0].errorbar(34, m, yerr=sd, fmt='D', color=color, ms=7, capsize=4, lw=1.5)
        tr = sorted([r for r in sequence if r['method']=='B' and float(r['learning_rate'])==rate
             and r['population']=='training_validation' and r['metric']=='composite'],
             key=lambda r:int(r['total_visits']))
        for r in tr:
            record('training_cubes', 'Fine-tune', rate, int(r['total_visits']), 'current network snapshot',
                   values(r), source_ref('sequence_scores.csv', [r]))
        band(axes[1], [int(r['total_visits']) for r in tr], [values(r) for r in tr], color, marker='o')

    pm, ps = record('new_cubes', 'Plain', '', '', 'independent reset comparison, all 30', plain_cumulative[-1],
                    baseline_source + ':query.aggregates.push_velocity_nmse.per_cube[all]')
    axes[0].errorbar(34, pm, yerr=ps, fmt='D', color=GRAY, ms=7, capsize=4, lw=1.5)
    axes[0].axvline(31.8, color='#cbd5e1', ls=':', lw=1)
    axes[0].set_xlim(.3, 36)
    axes[0].set_xticks([1, 5, 10, 15, 20, 25, 30, 34],
                       ['1','5','10','15','20','25','30','Reset\nper cube'])
    axes[0].set_ylabel('Push velocity-change error (NMSE; unitless)')
    axes[0].set_xlabel('Cubes visited in sequence', labelpad=8)
    axes[1].set_xlim(-.4, 30.7)
    axes[1].set_xticks([0, 5, 10, 15, 20, 25, 30])
    axes[1].set_ylabel('Combined prediction error (NMSE; unitless)')
    axes[1].set_xlabel('Cubes visited in sequence', labelpad=8)
    for ax in axes:
        ax.grid(axis='y', color='#e2e8f0', lw=.8)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(6))
    axes[0].set_title('New cubes', loc='left', pad=37, fontweight='bold')
    axes[1].set_title('Training cubes', loc='left', pad=37, fontweight='bold')
    axes[0].text(0, 1.035, 'Cumulative average through each visit', transform=axes[0].transAxes,
                 fontsize=12, color='#475569')
    axes[1].text(0, 1.035, 'Held-out shoves, evaluated at saved snapshots', transform=axes[1].transAxes,
                 fontsize=12, color='#475569')
    legend = [Line2D([0],[0],color=COLORS[1e-5], lw=2.5, label='Fine-tune: learning rate 10⁻⁵'),
              Line2D([0],[0],color=COLORS[1e-4], lw=2.5, label='Fine-tune: learning rate 10⁻⁴'),
              Line2D([0],[0],color=GRAY, lw=2.5, ls='--', label='No adaptation (Plain)'),
              Line2D([0],[0],color='#334155', marker='D', lw=0, ms=7, label='Reset weights before each cube')]
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.51,.055), ncol=2,
               frameon=False, fontsize=12, columnspacing=2.5)
    fig.text(.075,.015,'Lower is better · Bands and error bars: seed SD · Five seeds · One fixed visiting order',
             fontsize=11, color='#475569')
    fig.savefig(OUT/'fig_drift.png', dpi=220, facecolor='white')
    fig.savefig(OUT/'fig_drift.svg', facecolor='white')
    plt.close(fig)
    with (OUT/'drift_numbers.csv').open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(RECORDS[0])); w.writeheader(); w.writerows(RECORDS)
    (OUT/'drift_source_hashes.json').write_text(json.dumps(SOURCES, indent=2)+'\n')
    print(json.dumps({'new_cube_final': [r for r in RECORDS if r['panel']=='new_cubes' and r['visit']==30],
                      'training_final': [r for r in RECORDS if r['panel']=='training_cubes' and r['visit']==30]}, indent=2))

if __name__=='__main__': main()
