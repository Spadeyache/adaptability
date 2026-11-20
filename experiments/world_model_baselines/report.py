"""Assemble saved runs only; unavailable and unrun measurements stay explicit."""
import json
from pathlib import Path

import numpy as np

from .data import RESULTS, ROOT, TASKS, digest, write_json

METRICS = ('one_step_mse', 'one_step_r2', 'rollout_10_mse')
NAMES = ('B0', 'B1_K1', 'B1_K5', 'B1_K10', 'B1_K20', 'B2')


def moments(values):
    available = [float(value) for value in values if value is not None]
    complete = len(available) == len(values) and bool(values)
    return dict(n=len(available), expected_n=len(values), mean=float(np.mean(available)) if complete else None,
                sd=float(np.std(available, ddof=1)) if complete and len(available) > 1 else None,
                values=values)


def summarize(groups):
    result = {key: moments([group[key] for group in groups]) for key in METRICS}
    for key in ('moving_rows', 'rollout_starts', 'selected_rows'):
        result[key] = [group[key] for group in groups]
    return result


def fmt(summary):
    if summary['mean'] is None:
        return 'N/A'
    value = f"{summary['mean']:.6g}"
    return value + (f" ± {summary['sd']:.3g}" if summary['sd'] is not None else ' (1 seed)')


def report():
    protocol = json.loads((RESULTS / 'protocol.json').read_text())
    sources, runs = {}, []
    for path in sorted((ROOT / 'runs/baselines').glob('*/seed_*/evaluation.json')):
        value = json.loads(path.read_text())
        if value['protocol_sha256'] != digest(RESULTS / 'protocol.json'):
            raise ValueError(f'Protocol mismatch: {path}')
        sources[str(path.relative_to(ROOT))] = digest(path)
        runs.append(value)
    groups = {name: sorted([run for run in runs if run['model'] == name], key=lambda x:x['seed'])
              for name in NAMES}
    identities = [(run['model'],run['seed']) for run in runs]
    if len(set(identities)) != len(identities) or any(name not in NAMES or seed not in range(5) for name,seed in identities):
        raise ValueError('Duplicate or unknown model/seed in saved evaluation files')
    summary, profiles, windows = {}, {}, {}
    table = ['| Model / evaluation | Task | One-step normalized MSE | R² | 10-step normalized MSE |',
             '| --- | --- | ---: | ---: | ---: |']
    for name, values in groups.items():
        if not values:
            continue
        summary[name], windows[name], profiles[name] = {}, {}, {}
        for task in (*TASKS, 'overall'):
            selected = [x['evaluation']['overall'] if task == 'overall' else x['evaluation']['per_task'][task]
                        for x in values]
            summary[name][task] = summarize(selected)
            stats = summary[name][task]
            table.append(f"| {name} | {task} | " + ' | '.join(fmt(stats[key]) for key in METRICS) + ' |')
            if name.startswith('B1'):
                windows[name][task] = {}
                for window in ('early','late'):
                    section = [x['evaluation']['early_late']['overall'][window] if task == 'overall' else
                               x['evaluation']['early_late']['per_task'][task][window] for x in values]
                    stats = windows[name][task][window] = summarize(section)
                    table.append(f"| {name}, {'steps 1–5' if window == 'early' else 'steps 6+'} | {task} | " +
                                 ' | '.join(fmt(stats[key]) for key in METRICS) + ' |')
                sequences = [x['evaluation']['step_profile']['overall'] if task == 'overall' else
                             x['evaluation']['step_profile']['per_task'][task] for x in values]
                indexed = [{row['step']: row for row in seq} for seq in sequences]
                profiles[name][task] = [dict(step=step, **summarize([seq[step] for seq in indexed]))
                                       for step in sorted(indexed[0])]
    headline = [row for row in table[2:] if row.split('|')[1].strip() in NAMES
                and row.split('|')[2].strip() == 'lift']
    table = table[:2] + headline + [row for row in table[2:] if row not in headline]
    adaptation_path = RESULTS / 'adaptation/aggregate.json'
    adaptation = json.loads(adaptation_path.read_text()) if adaptation_path.exists() else None
    if adaptation is not None:
        if (adaptation['protocol_sha256'] != digest(RESULTS / 'protocol.json') or
                adaptation['split_sha256'] != digest(RESULTS / 'split.json')):
            raise ValueError('Adaptation aggregate uses another protocol or split')
        sources[str(adaptation_path.relative_to(ROOT))] = digest(adaptation_path)
    valid_runs = all(all(x['evaluation']['per_task'][task][key] is not None
                        for task in TASKS for key in METRICS)
                     and x['evaluation']['per_task']['lift']['one_step_r2'] > 0.01 for x in runs)
    complete = valid_runs and all({x['seed'] for x in values} == set(range(5)) for values in groups.values()) and bool(
        adaptation is not None and adaptation['complete_design'] and adaptation['complete_primary_metrics'])
    metrics = dict(schema_version=1, status='complete' if complete else 'partial', protocol=protocol,
                   source_files=sources, completed_model_seeds={name:[x['seed'] for x in values] for name,values in groups.items()},
                   summary=summary, b1_step_profiles=profiles, b1_early_late=windows,
                   runs=runs, adaptation=adaptation,
                   note='No fabricated or extrapolated results. First-five-step moving-only errors are undefined (zero moving rows). SD is sample SD across seeds.')
    availability_path = RESULTS / 'history_availability.json'
    if availability_path.exists():
        metrics['history_availability'] = json.loads(availability_path.read_text())
        sources[str(availability_path.relative_to(ROOT))] = digest(availability_path)
    write_json(RESULTS / 'metrics.json', metrics)
    if not runs:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    present = [name for name in NAMES if name in summary]
    fig, axes = plt.subplots(1,2,figsize=(12,4), constrained_layout=True)
    for axis, key, title in zip(axes, METRICS[:2], ('One-step error (lower is better)', 'Variance explained (higher is better)')):
        means = [summary[name]['lift'][key]['mean'] if summary[name]['lift'][key]['mean'] is not None else np.nan for name in present]
        sd = [summary[name]['lift'][key]['sd'] or 0 for name in present]
        axis.bar(present, means, yerr=sd, capsize=4, color=['#405c91' if name == 'B0' else '#2c9377' if name == 'B2' else '#cb8344' for name in present])
        axis.set_title(title)
        axis.set_ylabel('Normalized MSE' if key.endswith('mse') else 'R²')
        axis.tick_params(axis='x',rotation=30)
    full_plot = all({x['seed'] for x in groups[n]} == set(range(5)) and all(
        summary[n]['lift'][key]['n'] == 5 for key in METRICS[:2]) for n in present)
    fig.suptitle('Lift evaluation: ' + ('5 seeds, mean ± SD' if full_plot else 'Partial saved runs; no error bar for a single seed'))
    for ext in ('png','pdf'):
        fig.savefig(RESULTS / f'plot1_lift_baselines.{ext}',dpi=160)
    plt.close(fig)
    history_names = [name for name in present if name.startswith('B1')]
    if history_names:
        fig, axes = plt.subplots(2,2,figsize=(12,8), constrained_layout=True)
        for axis, task in zip(axes.flat, (*TASKS,'overall')):
            for name in history_names:
                records = profiles[name][task]
                x = np.array([r['step'] + 1 for r in records])
                y = np.array([r['one_step_mse']['mean'] if r['one_step_mse']['mean'] is not None else np.nan for r in records])
                sd = np.array([r['one_step_mse']['sd'] or 0 for r in records])
                line, = axis.plot(x,y,label=name)
                axis.fill_between(x,y-sd,y+sd,alpha=.12,color=line.get_color())
            axis.set(title=task,xlabel='Step within episode (1-based)',ylabel='Moving-only normalized MSE')
            axis.set_xlim(1, max(row['step'] + 1 for row in profiles[history_names[0]][task]))
            axis.axvspan(1, 5, color='grey', alpha=.12)
            axis.legend()
        fig.suptitle('History baseline: error as the episode progresses; mean ± SD across seeds')
        for ext in ('png','pdf'):
            fig.savefig(RESULTS / f'plot2_history_by_step.{ext}',dpi=160)
        plt.close(fig)
    # Adaptation plotting/table is added from its saved aggregate schema only.
    if adaptation is not None:
        from .adaptation import render_report
        table.extend(render_report(adaptation, RESULTS))
    pilot_path = RESULTS / 'pilot.json'
    pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else None
    lines = ['# No-belief world-model baselines', '',
             ('All requested model seeds and adaptation trajectories are complete.' if complete else
              'Partial results: the tables contain only assembled saved results. The study is still incomplete; unfinished studies have no final aggregate yet.'), '',
             'Frozen split: `results/baselines/split.json`. Full protocol: `results/baselines/protocol.json`.',
             'Shared train-only scaling arrays: `results/baselines/normalization.npz` (provenance and formulas in `normalization.json`). Reuse these arrays for later model comparisons.',
             'Dataset: `runs/lift_slide_belief/slip_probe/paired/fixed/data`; 200 train and 30 test cubes, 10 episodes per task. Existing data and z models are unchanged.',
             'Test episodes 1–5 are context and 6–10 evaluation (stored IDs 0–4 / 5–9). The 51-state projection matches the latest fixed-grip z model.', '',
             'Each metric uses observed cube-moving rows (>0.01 m/s). MSE pools rows and state coordinates. R² is 1−SSE/SST for normalized state deltas with each coordinate centered separately. Ten-step MSE is terminal state error scaled by the training delta SD; it uses recorded actions and predicted future states/history. The two MSE horizons use the same scale.',
             'Mean ± sample SD is across model seeds. Every model uses the same training episode split and train-only normalizers. Only B2 reads mass/friction. B1 resets at each episode and sees only completed past transitions.',
             'All models use a 3 × 256 ReLU MLP, Adam at 1e-3, batch 1024, and the existing early stopping rule (patience 12). B1 adds a GRU with hidden size 64 for K = 1, 5, 10, or 20 previous transitions. No model has a z table.',
             'B2 versus B0 tests the added value of supplying true physical constants. B2 does not test learning a separate belief. To assess the accuracy cost of separating belief from dynamics, compare the later learned-belief model against B1 with matched access to past observations. B0 may already infer some physical information from the measured current state.',
             'Weight adaptation crosses five B0 seeds with five fixed cube orders at each learning rate (50 trajectories). It runs on CPU while the base models train on GPU. Each cube gets a fresh Adam optimizer for 100 steps; all learned weights carry forward and normalization stays fixed. Adaptation summaries average orders within each model seed, then report mean ± sample SD across the five model seeds; raw order variability is also saved.',
             'The first five literal steps contain no moving samples in any eval episode. Their moving-only metrics are N/A; the window has not been shifted.', '',
             'Every moving evaluation row already has at least 20 completed past transitions. Consequently every scored B1 window is full for all requested K values. The step-index curve measures error through the episode, but does not isolate the effect of filling the history window; task phase and history content also change.', '',
             *table, '',
             'All per-seed, per-cube, per-step results and sample counts are in `results/baselines/metrics.json`. Checkpoints and training logs are in `runs/baselines/`.', '']
    if pilot:
        lines.extend([f"B0 seed 0 pilot: {pilot['training']['training_seconds']:.3f} seconds training; {pilot['wall_seconds']:.3f} seconds loading, fitting, saving and evaluating; {pilot['training']['epochs']} epochs (best {pilot['training']['best_epoch']}). These are measured times.", ''])
    lines.extend(['## Exact rerun commands', '', '```bash', 'cd /home/ubuntu/faliure',
        'source scripts/activate_grip.sh',
        'export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}',
        'python -m experiments.world_model_baselines freeze',
        'python -m experiments.world_model_baselines pilot --device cuda',
        '# Report the measured pilot runtime and review its saved metrics before the remaining commands.',
        'python -m experiments.world_model_baselines acknowledge-pilot',
        'python -m experiments.world_model_baselines sweep --device cuda',
        'python -m experiments.world_model_baselines adapt --device cpu --threads 2',
        'python -m experiments.world_model_baselines report', '```', '',
        'Commands reuse completed checkpoints/results. For a fresh rerun, archive `runs/baselines/` and the generated result JSON/plots first, keeping the frozen `protocol.json`, `split.json`, and `implementation.json`. Report the new pilot before acknowledging its new checksum. Do not alter the frozen split.',
        'B1 adds a GRU and therefore parameters as well as history. B2 is the privileged-information comparator, not a guaranteed numerical optimum. Recorded-action rollouts measure forecasting, not closed-loop task success.', ''])
    (ROOT / 'BASELINE.md').write_text('\n'.join(lines))
    print(f'Saved {RESULTS / "metrics.json"} and {ROOT / "BASELINE.md"}',flush=True)
