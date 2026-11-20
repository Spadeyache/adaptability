"""One partial gain/damage figure, strictly from saved Step 3 results.

No experiment module, checkpoint, trace, predictor, or optimizer is imported.
"""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
SEEDS = [101, 211, 307, 401, 503]
CUBES = list(range(290, 320))
SOURCES = {}


def read_json(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def mean(values): return statistics.mean(values)
def sd(values): return statistics.stdev(values)
def close(a, b): assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14), (a, b)


def main():
    current_path = RESULTS/'current_cube_sequence_scores.csv'
    SOURCES[str(current_path.relative_to(ROOT))] = hashlib.sha256(current_path.read_bytes()).hexdigest()
    with current_path.open(newline='') as f: current_rows = list(csv.DictReader(f))
    selected = read_json(RESULTS/'selected_config.json')
    grid = read_json(RESULTS/'validation_grid.json')
    baselines = {s: read_json(RESULTS/f'baselines/plain_seed{s}.json') for s in SEEDS}
    plain_push = {}
    plain_training = {}
    for seed, rec in baselines.items():
        cubes = rec['query']['aggregates']['push_velocity_nmse']['per_cube']
        assert [r['cube_id'] for r in cubes] == CUBES
        assert all(r['episodes'] == 5 for r in cubes)
        plain_push[seed] = {r['cube_id']: r['mean'] for r in cubes}
        plain_training[seed] = rec['training_validation']['scores']['composite']
    records = []
    a_sequences = {}
    labels = [('B', 1e-5, 'B: lr 10⁻⁵', 'B_lr1e-05'),
              ('B', 1e-4, 'B: lr 10⁻⁴', 'B_lr1e-04'),
              ('A', .01, 'A: selected z4, λ=1', 'A')]
    for method, lr, label, stem in labels:
        rows = sorted([r for r in current_rows if r['method'] == method and
                       float(r['learning_rate']) == lr and r['metric'] == 'push_velocity_nmse'],
                      key=lambda r: int(r['total_visits']))
        assert [int(r['total_visits']) for r in rows] == list(range(1, 31))
        assert [int(r['cube_id']) for r in rows] == CUBES
        gains, damages, pushes, training = [], [], [], []
        for seed in SEEDS:
            seq = read_json(RESULTS/f'sequence/{stem}_seed{seed}.json')
            assert [v['cube_id'] for v in seq['visits']] == CUBES
            values = [float(r[f'seed_{seed}']) for r in rows]
            for value, visit in zip(values, seq['visits']):
                close(value, visit['query']['scores']['push_velocity_nmse'])
            pushes.append(mean(values))
            gains.append(mean([plain_push[seed][c]-v for c, v in zip(CUBES, values)]))
            final = [r for r in seq['snapshots'] if r['visits'] == 30]
            assert len(final) == 1
            training.append(final[0]['training_validation']['scores']['composite'])
            damages.append(training[-1]-plain_training[seed])
            if method == 'A':
                a_sequences[seed] = seq
                assert len({r['weights_sha256'] for r in seq['snapshots']}) == 1
                assert all(r['training_validation']['scores']['composite'] == training[-1]
                           for r in seq['snapshots'])
                assert seq['initial_training_validation']['scores']['composite'] == training[-1]
        rec = dict(point_id=stem, label=label, plotted=True, method=method, population='Step 3 test cubes 290–319',
            z_dimension=selected['A']['z_dim'] if method == 'A' else '',
            penalty=selected['A']['penalty'] if method == 'A' else '', learning_rate=lr,
            support_shoves=5, gain_cube_count=30, visiting_order='290,291,...,319', damage_after_visit=30,
            gain_mean_nmse=mean(gains), gain_seed_sd=sd(gains), damage_mean_nmse=mean(damages), damage_seed_sd=sd(damages),
            model_current_cube_push_mean=mean(pushes), plain_same_cube_push_mean=mean([mean(list(plain_push[s].values())) for s in SEEDS]),
            model_training_validation_composite_mean=mean(training), plain_training_validation_composite_mean=mean(plain_training.values()),
            gain_definition='mean over30 cubes of Plain push NMSE minus current-cube push NMSE immediately after that cube adaptation',
            damage_definition='training-cube validation composite after visit30 minus frozen Plain composite; IDs0–179 shoves8–9',
            unavailable_reason='')
        for seed, x, y in zip(SEEDS, gains, damages):
            rec[f'gain_seed_{seed}'] = x; rec[f'damage_seed_{seed}'] = y
        records.append(rec)
    for dim in [4, 8, 16]:
        for penalty in [.0001, .001, .01, .1, 1.]:
            entries = sorted([r for r in grid['a_records'] if r['z_dim'] == dim and r['penalty'] == penalty], key=lambda r: r['seed'])
            assert [r['seed'] for r in entries] == SEEDS
            values = [r['query']['scores']['push_velocity_nmse'] for r in entries]
            assert all(r['query']['cube_ids'] == list(range(180, 200)) and r['query']['episode_ids'] == list(range(5, 10)) for r in entries)
            reason = 'No saved unadapted Plain push-query baseline on validation cubes180–199, shoves5–9; gain unavailable.'
            rec = dict(point_id=f'validation_z{dim}_penalty{penalty:g}',
                label=f'Validation z{dim}, penalty {penalty:g}', plotted=False, method='z-grid',
                population='VALIDATION cubes 180–199, query shoves5–9', z_dimension=dim, penalty=penalty,
                learning_rate=.01, support_shoves=5, gain_cube_count=20,
                saved_validation_push_mean=mean(values), saved_validation_push_seed_sd=sd(values),
                gain_definition='Unavailable: matching validation Plain baseline was not saved',
                damage_definition='Composite difference from Plain on IDs0–179 shoves8–9', unavailable_reason=reason)
            if dim == 4:
                # All penalties use identical frozen z4 weights/original training embeddings.
                for r in entries:
                    assert r['model_state_sha256'] == a_sequences[r['seed']]['snapshots'][0]['weights_sha256']
                reference = records[2]
                for key in ['damage_mean_nmse', 'damage_seed_sd', 'model_training_validation_composite_mean',
                            'plain_training_validation_composite_mean'] + [f'damage_seed_{s}' for s in SEEDS]:
                    rec[key] = reference[key]
                rec['unavailable_reason'] += ' Saved z4 training-validation offset is reusable across penalties; it is a base-network offset, not adaptation damage.'
            else:
                rec['unavailable_reason'] += f' No saved z{dim} training-validation coast/composite score; damage unavailable.'
            for seed, value in zip(SEEDS, values): rec[f'validation_push_seed_{seed}'] = value
            records.append(rec)
    assert len(records) == 18 and sum(r['plotted'] for r in records) == 3
    for rec in records:
        for value in rec.values():
            if isinstance(value, float): assert math.isfinite(value)
    fields = list(dict.fromkeys(key for rec in records for key in rec))
    output_csv = RESULTS/'gain_damage_partial.csv'
    with output_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(records)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(10.7, 8.0))
    fig.subplots_adjust(left=.13, right=.97, top=.85, bottom=.28)
    colors = ['#1779A6', '#C06018', '#7951AA']
    offsets = [(12, -22), (12, 12), (12, 12)]
    for rec, color, offset, marker in zip(records[:3], colors, offsets, ['o', 's', 'D']):
        ax.errorbar(rec['gain_mean_nmse'], rec['damage_mean_nmse'],
            xerr=rec['gain_seed_sd'], yerr=rec['damage_seed_sd'], fmt=marker,
            color=color, markeredgecolor='white', markeredgewidth=.8, markersize=9,
            elinewidth=1.7, capsize=5, capthick=1.7, zorder=3)
        ax.annotate(rec['label'], (rec['gain_mean_nmse'], rec['damage_mean_nmse']),
                    xytext=offset, textcoords='offset points', color=color, fontweight='bold')
    ax.axvline(0, color='#777777', lw=1, ls='--', zorder=1)
    ax.axhline(0, color='#777777', lw=1, ls='--', zorder=1)
    ax.grid(color='#dddddd', alpha=.65, linewidth=.6)
    ax.set_xlabel('Gain on new cubes: push-velocity NMSE reduction vs Plain\nMean over each cube immediately after its own adaptation')
    ax.set_ylabel('Damage: training-validation composite NMSE increase vs Plain\nMeasured after visit 30')
    ax.set_title('Step 3: gain on visited cubes versus training-validation damage', fontsize=15, pad=16)
    ax.margins(x=.25, y=.16)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis='x', style='plain')
    ax.text(.98, .04, 'Right = more gain   •   Down = less damage', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, color='#555555')
    caption = ('Partial figure: three points; seed SD error bars; one fixed visiting order (290 → 319).\n'
        'Horizontal errors are SDs of paired per-seed gains, averaged over all 30 current-cube scores;\n'
        'vertical errors are SDs of paired differences from Plain on training-cube validation after visit 30.\n'
        'A’s damage is constant across visits: its offset from Plain comes from the two-part base network,\n'
        'not from adaptation. All 15 z-grid settings are omitted here and listed as unavailable in the CSV.\n'
        'Saved results only. Gain uses successive model states; it is not a final-model rescore of all 30 cubes.')
    fig.text(.13, .185, caption, ha='left', va='top', fontsize=10, linespacing=1.45)
    figures = RESULTS/'figures'; figures.mkdir(exist_ok=True)
    outputs = [figures/'gain_damage_partial.png', figures/'gain_damage_partial.svg']
    for path in outputs: fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=.20)
    plt.close(fig)
    for name, expected in SOURCES.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    manifest = dict(saved_results_only=True, no_collection_training_adaptation_or_evaluation=True,
        source_files=SOURCES, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plotted_points=3, csv_rows=18, omitted_validation_settings=15,
        error_bars='sample SD (ddof=1) of paired per-seed differences; seeds101,211,307,401,503',
        gain_units='absolute NMSE reduction, not percent', damage_units='absolute composite NMSE increase',
        caption=caption, output_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in outputs+[output_csv]})
    (RESULTS/'gain_damage_partial_sources.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['point_id', 'gain_mean_nmse', 'gain_seed_sd', 'damage_mean_nmse', 'damage_seed_sd']}
                      for r in records[:3]], indent=2))


if __name__ == '__main__': main()
