"""Generate the pilot-only report from saved evidence; never collect or train."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def fmt(value, digits=5):
    return f'{float(value):.{digits}g}'


def link(root, name, label=None):
    return f'[{label or name}]({root / name})'


def generate(root=ROOT):
    root = Path(root).resolve()
    paths = {
        'timing': root / 'results/timing.json',
        'metrics': root / 'results/pilot_metrics.json',
        'audit': root / 'results/pilot_audit.json',
        'probe': root / 'results/action_probe.json',
        'protocol': root / 'data/protocol.json',
        'plans': root / 'data/plans.json',
    }
    values = {name: read(path) for name, path in paths.items()}
    timing, metrics, audit, probe = (values[k] for k in ('timing', 'metrics', 'audit', 'probe'))
    protocol, plans = values['protocol'], values['plans']
    if audit.get('status') != 'passed':
        raise RuntimeError('The corrected pilot audit must pass before publishing its report')
    if len(metrics) != 100 or audit.get('episodes') != 100 or timing.get('collected_episodes') != 100:
        raise RuntimeError('Expected all 100 accepted episodes before reporting completion')
    unique_ids = sorted({int(row['cube_id']) for row in metrics})
    if unique_ids != list(range(10)):
        raise RuntimeError('Expected exactly pilot training cube IDs 0–9')
    if timing.get('plain_oracle_training_started') or timing.get('belief_model_started'):
        raise RuntimeError('Pilot-only authorization boundary has changed')
    if timing.get('approval_status') != 'AWAITING_USER_GO':
        raise RuntimeError('Expected explicit waiting-for-go status')

    plan_by_id = {(int(p['cube_id']), int(p['episode_id'])): p for p in plans}
    archived = []
    for version in ('v1', 'v2'):
        metric_path = root / 'versions' / version / 'results/pilot_metrics.json'
        rows = read(metric_path)
        archived.append((version, rows))
        paths[f'{version}_metrics'] = metric_path
        if not {int(r['cube_id']) for r in rows}.issubset(unique_ids):
            raise RuntimeError('Archived execution includes an unauthorized identity')
    v1_timing_path = root / 'versions/v1/results/timing.json'
    v1_timing = read(v1_timing_path)
    paths['v1_timing'] = v1_timing_path
    total_executions = len(metrics) + sum(len(rows) for _, rows in archived)
    measured_episode_time = sum(float(row['wall_seconds']) for row in metrics)
    measured_episode_time += sum(float(row['wall_seconds']) for _, rows in archived for row in rows)

    max_coast = max(float(row['stopping_distance_m']) for row in metrics)
    total_displacements = []
    for row in metrics:
        plan = plan_by_id[(int(row['cube_id']), int(row['episode_id']))]
        total_displacements.append(float(np.linalg.norm(np.asarray(row['stopping_xy_m']) - plan['start_xy'])))
    launch_count = sum(float(row['launch_speed_m_s']) > .001 for row in metrics)
    stronger_launch_count = sum(float(row['launch_speed_m_s']) > .02 for row in metrics)
    largest_force_row = max(metrics, key=lambda row: float(row['max_contact_force_n']))
    largest_force_plan = plan_by_id[(int(largest_force_row['cube_id']), int(largest_force_row['episode_id']))]
    force_ratio = max(float(row['cap_ratio']) for row in metrics)
    coast_count = int(audit['coast_transitions'])
    moving_coast = int(audit['moving_coast_transitions'])
    mass_lo, mass_hi = protocol['mass_range_kg']
    friction_lo, friction_hi = protocol['cube_friction_range']
    floor = float(protocol['floor_material_friction'])
    combined_lo, combined_hi = ((friction_lo + floor) / 2, (friction_hi + floor) / 2)
    active_seconds = float(timing['this_invocation_wall_seconds'])
    episode_seconds = float(timing['episode_wall_seconds_sum'])

    lines = [
        '# Varied-shove gate: ten-cube pilot', '',
        '**Status: pilot complete; awaiting your go. The Oracle-versus-Plain prediction gate has not been run.**', '',
        f'The active dataset contains **{len(metrics)} shoves on {len(unique_ids)} newly sampled training cubes** '
        f'and **{int(audit["transitions"]):,} transitions**. No motion was collected for any other cube identity. '
        'No Plain, Oracle, or belief model has been trained. The only fitted models are the explicitly requested '
        'actions-only generator probes, described below.', '',
        '## Range, randomization, and data', '',
        f'Mass is independently sampled uniformly over **{fmt(mass_lo)}–{fmt(mass_hi)} kg**, narrowed from '
        'the previous 0.04–0.50 kg range to avoid the very long light-cube slides. Cube material friction is '
        f'independently sampled over **{fmt(friction_lo)}–{fmt(friction_hi)}**. The fixed floor material is '
        f'{fmt(floor)}; configured combined friction uses PhysX averaging, '
        f'`(cube friction + floor friction)/2`, giving **{fmt(combined_lo)}–{fmt(combined_hi)}**. '
        'Per-cube mass, cube friction, and combined friction values reside only in `data/cubes.json`.', '',
        'The split and command plans are fixed for 200 training cubes and 30 test cubes, with 10 shoves each. '
        'Only training cube IDs 0–9 were physically simulated. Every shove independently randomizes its '
        '2–4 N cap, world push direction, face contact offset, cube start XY, and cube yaw. '
        'Property sampling and command sampling use separate RNG streams; no plan is accepted or rejected '
        'based on true properties.', '',
        'The physical setup remains an idealized 5 g spherical pusher and a six-DOF 4 cm cube on an infinite plane. '
        'This is not a Panda controller evaluation. In the accepted revision, the pusher translates freely '
        'in XYZ, has locked rotation, and has gravity and damping disabled. Each force pulse lasts 80 ms, '
        'followed by a fixed 1 s coast. Physics runs at 1 ms; model transitions are saved every 5 ms.', '',
        'Each transition contains `cube_id`, `episode_id`, `step`, `phase`, `state`, `action`, and `next_state`. '
        'IDs and phase labels select bookkeeping and metrics; numeric IDs are never model features. '
        'State contains observed cube pose/velocities and pusher pose/velocities. Actions contain only '
        'commanded force, cap, engagement flag, and a release target computed from the initial plan. '
        'Measured force, true properties, stop time, and variable episode length are excluded from actions.', '',
        f'Every episode has **16 push and 200 coast transitions**: **{int(audit["push_transitions"]):,} push** '
        f'and **{coast_count:,} coast** transitions in the pilot. Recording continues after rest, so sequence '
        'length cannot reveal the properties.', '',
        '## Measured collection time', '',
        '| Item | Recorded time |', '|---|---:|',
        f'| Accepted 100-episode invocation, including startup | {active_seconds:.3f} s |',
        f'| Accepted episode collection + compression, summed | {episode_seconds:.3f} s |',
        f'| Mean accepted episode | {float(timing["mean_episode_wall_seconds"]):.4f} s |',
        f'| Median accepted episode | {float(timing["median_episode_wall_seconds"]):.4f} s |',
        f'| Generator-only action probe, separate from collection | {float(probe["elapsed_seconds"]):.3f} s |', '',
        f'Linear extrapolation of the measured serial episode time gives **'
        f'{float(timing["total_2300_episode_projection_seconds"]):.1f} s '
        f'({float(timing["total_2300_episode_projection_seconds"])/60:.2f} min)** for all 2,300 episodes, '
        f'or **{float(timing["remaining_2200_episode_projection_seconds"]):.1f} s '
        f'({float(timing["remaining_2200_episode_projection_seconds"])/60:.2f} min)** for the remaining 2,200. '
        'These are projections, not completed runs; they exclude startup, audits, probes, model training, '
        'and possible failures or reruns.', '',
        '### Corrected attempts retained in the archive', '',
        'The first 100-episode attempt used the contact API export as a full-force measurement. '
        'Oblique pushes revealed that it omitted tangential impulse, so that attempt was archived and excluded '
        'from the active dataset. The second attempt stopped after one episode because an other-contact '
        'guard counted zero-impulse proximity points as force-bearing contacts. The accepted third revision '
        'uses full XYZ momentum balance and verifies that other pusher contacts have zero normal impulse.', '',
        f'Across these corrections there were **{total_executions} physical episode executions** '
        f'({len(archived[0][1])}+{len(archived[1][1])}+{len(metrics)}), all on the same ten identities '
        'and the same preplanned settings. Only the final 100 episodes are accepted. '
        f'The archived first invocation took **{float(v1_timing["this_invocation_wall_seconds"]):.3f} s**. '
        f'The second attempt records **{sum(float(r["wall_seconds"]) for r in archived[1][1]):.3f} s** '
        'of episode time; its complete process wall time was not saved. '
        f'The summed recorded episode-processing time over all {total_executions} executions is '
        f'**{measured_episode_time:.3f} s**. This sum excludes startup and debugging time and is not presented '
        'as total elapsed project time.', '',
        '## Physical and leakage checks', '',
        f'The independent saved-trace audit **{audit["status"]}**. '
        'It reconstructs every command from the property-blind plan, checks fixed recording lengths, '
        'state/next-state alignment, immutable source/data seals, stopping windows, and the ten-cube boundary.', '',
        'Actual transmitted contact force is reconstructed every physics step as '
        '`F_contact = F_command − m_pusher × Δv_pusher/Δt`. This includes all three components. '
        'The reconstruction uses the known actuator mass, not the hidden cube mass. There are no pusher '
        'translation constraints, gravity, damping, or force-bearing contacts with other bodies. '
        'The normal component independently agrees with the simulator contact export. '
        'The instantaneous release velocity reset is excluded from contact-force reconstruction.', '',
        f'- Largest full contact-force/cap ratio: **{100*force_ratio:.3f}%**; no cap exceedances.',
        f'- Largest full contact force: **{float(largest_force_row["max_contact_force_n"]):.4f} N**, '
        f'under its **{float(largest_force_plan["cap_n"]):.4f} N** cap.',
        f'- Maximum pusher–cube contact force after release: **{fmt(audit["maximum_coast_contact_force_n"])} N**.',
        f'- All **{sum(bool(r["stopped"]) for r in metrics)}/{len(metrics)}** episodes reached confirmed rest.',
        f'- Maximum release-to-rest time: **{max(float(r["time_to_stop_s"]) for r in metrics):.3f} s** '
        '(confirmation time excluded).',
        f'- Maximum release-to-rest displacement: **{100*max_coast:.3f} cm**.',
        f'- Maximum start-to-rest horizontal displacement: **{100*max(total_displacements):.3f} cm** '
        '(displacement, not accumulated path length).',
        f'- Maximum launch speed: **{max(float(r["launch_speed_m_s"]) for r in metrics):.4f} m/s**.',
        f'- **{launch_count}/{len(metrics)}** launches exceeded 1 mm/s '
        f'(**{len(metrics)-launch_count}** were at or below that threshold); '
        f'**{stronger_launch_count}/{len(metrics)}** exceeded 2 cm/s.',
        f'- **{moving_coast:,}/{coast_count:,}** coast transitions '
        f'(**{100*moving_coast/coast_count:.3f}%**) meet the predefined moving mask: '
        'full linear speed>1 mm/s or angular speed>0.05 rad/s.', '',
        f'**Only {100*moving_coast/coast_count:.3f}% of coast transitions are moving; '
        f'{100*(1-moving_coast/coast_count):.3f}% are at or below the motion thresholds.** '
        'Many randomized low-strength or oblique shoves barely launch, and the fixed coast includes a long '
        'stationary tail. Thus the primary all-coast metric is dominated in sample count by resting states. '
        'The future report will also show the predefined moving-coast '
        'score; it will not substitute that score for the frozen primary gate after viewing test results.', '',
        '## Actions-only generator probe', '',
        'This probe uses all 2,300 preplanned action schedules: 2,000 from the 200 training identities '
        'and 300 from the 30 test identities. It does not read their motion or simulate additional cubes. '
        'Ten pilot cubes alone would be insufficient for the requested held-out-cube probe.', '',
        'The inputs are ten features describing the commanded force/cap, direction, contact offset, '
        'start XY, and yaw. No IDs, measured force, stop time, or duration features are used. Polynomial ridge '
        'and ExtraTrees each use five seeds, cube-disjoint 180/20 training validation splits, then refit '
        'all 200 training identities. Hyperparameters are never selected using test labels. '
        'The two independent fitted targets are mass and combined friction; cube-friction predictions '
        'use their train-only affine relationship for reporting.', '',
        '| Probe | Mass test R², mean ± seed SD | Cube friction test R² | Combined friction test R² |',
        '|---|---:|---:|---:|',
    ]
    for family, label in [('polynomial_ridge', 'Polynomial ridge'), ('extra_trees', 'ExtraTrees')]:
        aggregate = probe['aggregate'][family]
        cells = [f'{aggregate[t]["mean_r2"]:.5f} ± {aggregate[t]["seed_std_r2"]:.5f}'
                 for t in ('mass_kg', 'friction', 'combined_friction')]
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    baseline = probe['constant_train_mean_test_r2']
    lines += [
        '', f'For comparison, a constant training-mean predictor has test R² **{baseline["mass_kg"]:.5f}** '
        f'for mass and **{baseline["combined_friction"]:.5f}** for combined friction. '
        'Every fitted probe score is below 0.1 and none is positive. The observed values are close to zero '
        'and provide no useful predictive signal from commands. Negative R² values are reported literally. '
        'Ridge has zero seed spread because every validation split selected the same regularization and '
        'the final full-training fit is deterministic.', '',
        'Cube-cluster bootstrap intervals are saved with the scores. Thirty test identities still give '
        'limited precision; this empirical check and the code-path audit do not prove statistical independence. '
        '**This action-probe result is not a PASS of the Oracle-versus-Plain gate.**', '',
        '## Frozen next step, requiring your go', '',
        'After approval: collect the remaining 190 training cubes and 30 test cubes, retaining the accepted '
        'pilot episodes. Train Plain and Oracle with five paired seeds, three 256-unit hidden layers, '
        'shared training-only normalization, and normalized state-change targets. Plain receives state+action; '
        'Oracle additionally receives true mass and combined friction. No belief model is planned in this step.', '',
        'Report one-step NMSE, ten-step recursive rollout error, and stopping-position error separately '
        'for push and coast. Rollout windows stay within their named phase; no observed intermediate state '
        'is fed into recursive predictions. Stopping-position rollouts begin at the first push or first coast '
        'state and compare terminal predicted XY with recorded confirmed-rest XY. Aggregate equally by cube, '
        'then episode, and report five-seed mean, sample SD, and paired improvements.', '',
        'The frozen primary PASS rule is **all-coast one-step NMSE**: '
        '`mean(Oracle) ≤ 0.8 × mean(Plain)` and '
        '`mean(Oracle) + SD(Oracle) < mean(Plain) − SD(Plain)`. '
        'Moving-coast, ten-step, stopping-position, and cube-only scores remain required diagnostics. '
        'No test-driven tuning or metric substitution is allowed. If the gate fails, stop and report.', '',
        '**Collection and gate-model training are stopped here, pending your go.**', '',
        '## Saved evidence', '',
        '- ' + link(root, 'results/timing.json', 'Recorded timing'),
        '- ' + link(root, 'results/pilot_metrics.json', 'Per-episode physical metrics'),
        '- ' + link(root, 'results/pilot_audit.json', 'Independent data and force audit'),
        '- ' + link(root, 'results/action_probe.json', 'Actions-only probe scores and intervals'),
        '- ' + link(root, 'data/protocol.json', 'Frozen protocol'),
        '- ' + link(root, 'data/schema.json', 'Saved state/action schema'), '',
        'This report reads saved evidence only. Rebuild it with:', '',
        '```bash', 'cd /home/ubuntu/faliure',
        'PYTHONPATH=clean/vendor:. .venv/bin/python -m clean.shove_gate.report', '```', '',
        'Input SHA-256 hashes:', '',
    ]
    for name, path in paths.items():
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f'- `{path.relative_to(root)}`: `{sha}`')
    output = root / 'PILOT_REPORT.md'
    output.write_text('\n'.join(lines) + '\n')
    print(json.dumps({'report': str(output), 'accepted_episodes': len(metrics),
                      'unique_cubes': len(unique_ids), 'physical_episode_executions_including_archives': total_executions,
                      'status': 'AWAITING_USER_GO'}))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    generate(args.root)
