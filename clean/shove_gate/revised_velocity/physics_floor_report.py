"""Rebuild the pre-training physics report solely from saved measurements."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    data_path = ROOT / 'results/physics_floor.json'
    r = json.loads(data_path.read_text())
    motion = json.loads((ROOT / 'results/physics_motion_categories.json').read_text())['counts']
    full = r['splits']['all_training']
    exact = full['metrics']['requested_unclipped']
    clipped = full['metrics']['stop_clipped_diagnostic']
    diag = full['diagnostics']
    reduction = 100 * (1 - clipped['nmse'] / exact['nmse'])
    figure_dir = ROOT / 'results/figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    names = ['Fitting', 'Validation', 'All training']
    x = np.arange(3)
    for offset, key, label, color in [(-.18, 'requested_unclipped', 'Requested formula', '#3178a8'),
                                      (.18, 'stop_clipped_diagnostic', 'Stop-clipped diagnostic', '#bc652c')]:
        values = [r['splits'][s]['metrics'][key]['nmse'] for s in ['fitting', 'validation', 'all_training']]
        ax[0].bar(x + offset, values, width=.36, label=label, color=color)
    ax[0].axhline(.01, color='black', linestyle='--', linewidth=1, label='Predeclared near-zero threshold')
    ax[0].set_yscale('log'); ax[0].set_xticks(x, names)
    ax[0].set_ylabel('Horizontal velocity-change NMSE (log scale)')
    ax[0].set_title('Physics sanity reference; no learning')
    ax[0].legend(fontsize=8)
    keys = ['speed_0.001_to_0.01_m_s', 'speed_0.01_to_0.03_m_s', 'speed_0.03_to_0.1_m_s',
            'speed_0.1_to_0.3_m_s', 'speed_0.3_to_1_m_s', 'speed_1_to_10_m_s']
    ratios = [diag[k]['observed_deceleration_over_mu_g'] for k in keys]
    ax[1].plot(np.arange(len(keys)), ratios, 'o-', color='#3178a8')
    ax[1].axhline(1, color='black', linestyle='--', linewidth=1)
    ax[1].set_xticks(np.arange(len(keys)), ['.001–.01', '.01–.03', '.03–.1', '.1–.3', '.3–1', '1–10'], rotation=25)
    ax[1].set_xlabel('Horizontal speed bin (m/s)')
    ax[1].set_ylabel('Observed deceleration / (combined friction × 9.81)')
    ax[1].set_title('A full-step decrement overshoots near rest')
    for a in ax:
        a.grid(axis='y', alpha=.2); a.set_axisbelow(True)
    for suffix in ['png', 'svg']:
        fig.savefig(figure_dir / f'physics_floor.{suffix}', dpi=180)
    plt.close(fig)
    lines = [
        '# Physics sanity reference, measured before revised training', '',
        'The requested unclipped formula has **NMSE '
        f'{exact["nmse"]:.8f}** on all 200 training cubes: per-coordinate horizontal velocity-change '
        f'RMSE **{exact["rmse_m_s"]:.8f} m/s**. It just meets the predeclared near-zero threshold of 0.01; '
        'validation episodes alone are slightly above that threshold. No dynamics model was fit to obtain these values.', '',
        '| Training subset | Requested formula NMSE | Requested formula RMSE (m/s) | Stop-clipped diagnostic NMSE | Moving-coast starts |',
        '|---|---:|---:|---:|---:|',
    ]
    for title, split in zip(names, ['fitting', 'validation', 'all_training']):
        a = r['splits'][split]['metrics']['requested_unclipped']
        b = r['splits'][split]['metrics']['stop_clipped_diagnostic']
        lines.append(f'| {title} | {a["nmse"]:.8f} | {a["rmse_m_s"]:.8f} | {b["nmse"]:.8f} | {a["eligible_starts"]:,} |')
    high = diag['speed_ge_0p1m_s']
    clean = diag['ideal_step_would_not_cross_zero']
    near = diag['ideal_step_would_cross_zero']
    direction_ratios = [d['observed_deceleration_over_mu_g'] for d in full['high_speed_direction_bins']]
    lines += [
        '',
        'Most discrepancy is confined to the final partial sliding step. The requested formula applies '
        'a full 5 ms decrement even when the remaining speed is smaller than that decrement; '
        'the recorded cube stops instead of acquiring a reversed velocity. '
        f'There are {near["eligible_starts"]:,} such starts out of {exact["eligible_starts"]:,}. '
        f'Capping the decrement at the available speed reduces aggregate error by {reduction:.2f}%, '
        f'to NMSE {clipped["nmse"]:.8f}. This capped result is a separate diagnostic; the primary '
        'reference and final Oracle comparison retain the exact requested unclipped formula.', '',
        f'Outside those stopping crossings, NMSE is {clean["unclipped_nmse"]:.8f}. '
        f'At horizontal speeds ≥0.1 m/s, observed deceleration divided by combined_friction×9.81 '
        f'is {high["observed_deceleration_over_mu_g"]:.8f}, NMSE is {high["unclipped_nmse"]:.8f}, '
        f'and mean perpendicular acceleration is {high["mean_perpendicular_acceleration_m_s2"]:.8f} m/s². '
        f'The high-speed deceleration ratio ranges from {min(direction_ratios):.8f} to '
        f'{max(direction_ratios):.8f} across twelve 30° world-direction bins. '
        'These measurements agree closely with simple sliding friction over the sampled regime; '
        'they do not show a material alternative friction law. Small residuals remain in the discrete '
        'contact simulation, and this analysis does not identify their internal solver cause.', '',
        f'Of {exact["eligible_starts"]:,} moving-coast starts, '
        f'{diag["horizontal_speed_gt_1mm_s"]["eligible_starts"]:,} have horizontal speed >1 mm/s; '
        f'{diag["horizontal_speed_le_1mm_s"]["eligible_starts"]} is included only because the preserved '
        'full-linear/angular movement rule still marks it moving. '
        f'Using the original full-linear/angular thresholds, {motion["linear_only"]:,} starts are linear-only, '
        f'{motion["angular_only"]:,} is angular-only, and {motion["both_linear_and_angular"]:,} satisfy both. '
        'No samples are removed from the primary score.', '',
        f'![Physics sanity measurements]({figure_dir / "physics_floor.png"})', '',
        'The normalization scalar is '
        f'`S_delta_v² = {r["normalization"]["scalar_horizontal_delta_mse"]:.17g} (m/s)²`, '
        'computed only from episodes 0–7 of cubes 0–199. A zero-change predictor has NMSE 1 on that fitting subset. '
        'All scores first average steps within an episode, then episodes within a cube, then cubes equally. '
        'Conditional speed/direction diagnostics average only represented episodes and cubes.', '',
        'No retired or fresh test trajectories were opened. The analytic reference is a diagnostic baseline, '
        'not a proven irreducible error bound. It is possible for a predictor that handles the partial stopping '
        'step to outperform the unclipped reference.', '',
        f'[Saved measurements]({data_path}) · [Written revised protocol]({ROOT / "REVISED_PROTOCOL.md"})', '',
        f'Measurement SHA-256: `{hashlib.sha256(data_path.read_bytes()).hexdigest()}`',
        f'Protocol SHA-256 recorded before scoring: `{r["protocol_sha256"]}`', '',
    ]
    (ROOT / 'PHYSICS_FLOOR_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(ROOT / 'PHYSICS_FLOOR_REPORT.md'),
                          measurements_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
