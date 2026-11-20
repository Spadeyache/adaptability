"""Audit the discrete simulator force oracle without changing the original study."""
import argparse
import csv
from functools import partial
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from .collect import ROOT, start_worker
from .scripted import shove_actions
from .target_control import (
    CANDIDATE_FORCES_N, FINAL_COUNTS, TARGET_DISTANCES_M, TARGET_TOLERANCE_M,
    _episodes, _simulate_variant,
)


EXPLORATORY_TARGETS_M = (.020, .040, .060)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def filename(cube, episode, candidate):
    return f'cube_{cube:04}_episode_{episode}_candidate_{candidate:02}.npz'


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected a nonempty finite distribution')
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                min=float(values.min()), p10=float(np.quantile(values, .1)),
                p90=float(np.quantile(values, .9)), p95=float(np.quantile(values, .95)),
                max=float(values.max()))


def read_variant(path, episode, candidate, cfg):
    """Verify every reused and new trajectory against the immutable setup and commands."""
    with np.load(path, allow_pickle=False) as saved:
        start, stop = int(saved['force_start_step'][0]), int(saved['stop_step'][0])
        if (start, stop) != (episode['force_start'], episode['stop']):
            raise ValueError(f'Changed shove interval in {path}')
        assert len(saved['state']) == cfg['task_steps'][2] == stop + 1
        assert np.all(saved['cube_id'] == episode['cube_id'])
        assert np.all(saved['episode_id'] == episode['episode_id'])
        assert int(saved['candidate_index']) == candidate
        assert float(saved['selected_force_n']) == CANDIDATE_FORCES_N[candidate]
        assert np.all(saved['reset_seed'] == episode['reset_seed'])
        np.testing.assert_allclose(saved['state'][start], episode['initial_state'], atol=1e-5, rtol=0)
        np.testing.assert_array_equal(saved['state'][1:], saved['next_state'][:-1])
        expected_actions = shove_actions(episode['direction'], CANDIDATE_FORCES_N[candidate], stop-start+1, cfg)
        np.testing.assert_allclose(saved['action'][start:stop+1], expected_actions, atol=1e-7, rtol=0)
        assert np.isfinite(saved['state']).all() and np.isfinite(saved['next_state']).all()
        tail = round(cfg['rest_duration_s'] * cfg['control_hz'])
        rest = (int(saved['shove_release_step'][0]) < len(saved['state']) - tail
                and np.all(np.linalg.norm(saved['next_state'][-tail:, 28:31], axis=1) < cfg['speed_threshold_m_s'])
                and np.all(np.linalg.norm(saved['next_state'][-tail:, 31:34], axis=1) < cfg['rest_angular_speed_rad_s'])
                and np.all(np.linalg.norm(saved['contact_force_substeps_n'][-tail:], axis=2)
                           < cfg['rest_contact_threshold_n']))
        assert np.all(saved['observed_stopped'] == rest)
        endpoint = saved['next_state'][stop, 19:21].astype(np.float64)
        displacement = endpoint - episode['initial_state'][19:21].astype(np.float64)
        direction = episode['direction']
        return dict(cube_id=episode['cube_id'], episode_id=episode['episode_id'], candidate_index=candidate,
                    force_n=float(CANDIDATE_FORCES_N[candidate]), endpoint_x_m=float(endpoint[0]),
                    endpoint_y_m=float(endpoint[1]), actual_stopped=bool(rest),
                    signed_distance_m=float(displacement @ direction),
                    lateral_distance_m=float(displacement @ np.array([-direction[1], direction[0]])),
                    displacement_m=float(np.linalg.norm(displacement)),
                    ready_state_max_difference=float(np.max(np.abs(saved['state'][start] - episode['initial_state']))))


def oracle_case(episode, variants, distance):
    """Minimize full XY error over the same finite grid used by the planner."""
    endpoint = np.array([[v['endpoint_x_m'], v['endpoint_y_m']] for v in variants])
    target = episode['initial_state'][19:21].astype(np.float64) + distance * episode['direction']
    delta = endpoint - target
    errors = np.hypot(delta[:, 0], delta[:, 1])
    stopped = np.array([v['actual_stopped'] for v in variants])
    best = int(np.argmin(errors))
    stopped_best = int(np.argmin(np.where(stopped, errors, np.inf))) if stopped.any() else None
    return dict(cube_id=episode['cube_id'], episode_id=episode['episode_id'], target_distance_m=float(distance),
                target_x_m=float(target[0]), target_y_m=float(target[1]),
                oracle_candidate_index=best, oracle_force_n=float(CANDIDATE_FORCES_N[best]),
                oracle_error_m=float(errors[best]), oracle_selected_stopped=bool(stopped[best]),
                oracle_success=bool(np.any(stopped & (errors <= TARGET_TOLERANCE_M))),
                best_stopped_error_m=float(errors[stopped_best]) if stopped_best is not None else None,
                best_stopped_force_n=float(CANDIDATE_FORCES_N[stopped_best]) if stopped_best is not None else None,
                stopped_candidates=int(stopped.sum()),
                signed_distance_min_m=min(v['signed_distance_m'] for v in variants),
                signed_distance_max_m=max(v['signed_distance_m'] for v in variants),
                min_absolute_lateral_distance_m=min(abs(v['lateral_distance_m']) for v in variants),
                max_absolute_lateral_distance_m=max(abs(v['lateral_distance_m']) for v in variants))


def summarize_cases(cases):
    return dict(cases=len(cases), error_m=distribution([v['oracle_error_m'] for v in cases]),
                successes=sum(v['oracle_success'] for v in cases),
                success_rate=float(np.mean([v['oracle_success'] for v in cases])),
                minimum_selected_stopped_count=sum(v['oracle_selected_stopped'] for v in cases),
                signed_distance_min_m=distribution([v['signed_distance_min_m'] for v in cases]),
                signed_distance_max_m=distribution([v['signed_distance_max_m'] for v in cases]))


def audit_choices(trials, by_setup, originals):
    keys = list(zip(trials['seed'].tolist(), trials['cube_id'].tolist(), trials['episode_id'].tolist(),
                    trials['condition'].tolist()))
    expected = {(seed, cube, episode, condition) for seed in range(5) for cube, episode in by_setup
                for condition in FINAL_COUNTS}
    assert len(keys) == len(set(keys)) == 4500 and set(keys) == expected
    rows = []
    for i, (seed, cube, ep, condition) in enumerate(keys):
        predicted = trials['candidate_target_errors_m'][i]
        finite = np.isfinite(predicted)
        selected = int(trials['candidate_index'][i])
        best = int(np.argmin(np.where(finite, predicted, np.inf))) if finite.any() else 6
        assert selected == best, (seed, cube, ep, condition, selected, best)
        assert trials['valid_candidates'][i] == int(finite.sum())
        assert trials['force_n'][i] == CANDIDATE_FORCES_N[selected]
        assert trials['planner_status'][i] == ('finite_prediction' if finite.any() else 'all_nonfinite_midpoint_fallback')
        variant = by_setup[cube, ep][selected]
        oracle = originals[cube, ep]
        target = np.array([oracle['target_x_m'], oracle['target_y_m']])
        endpoint = np.array([variant['endpoint_x_m'], variant['endpoint_y_m']])
        np.testing.assert_allclose(trials['target_xy'][i], target, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(trials['actual_endpoint_xy'][i], endpoint)
        error = float(np.linalg.norm(endpoint-target))
        np.testing.assert_allclose(trials['actual_target_error_m'][i], error, rtol=1e-12, atol=1e-12)
        assert bool(trials['actual_stopped'][i]) == variant['actual_stopped']
        success = variant['actual_stopped'] and error <= TARGET_TOLERANCE_M
        assert bool(trials['target_success'][i]) == success
        regret = error - oracle['oracle_error_m']
        assert regret >= -1e-12
        rows.append(dict(seed=seed, cube_id=cube, episode_id=ep, condition=condition,
                         target_distance_m=oracle['target_distance_m'], selected_force_n=float(CANDIDATE_FORCES_N[selected]),
                         selected_candidate_index=selected, model_argmin_matches=True,
                         actual_error_m=error, actual_stopped=variant['actual_stopped'], actual_success=bool(success),
                         oracle_error_m=oracle['oracle_error_m'], oracle_force_n=oracle['oracle_force_n'],
                         oracle_success=oracle['oracle_success'], actual_regret_m=regret,
                         selected_oracle_force=selected == oracle['oracle_candidate_index']))
    return rows


def plot_response(by_setup, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    for axis, field, label in zip(axes, ('signed_distance_m', 'lateral_distance_m'),
                                  ('Forward displacement (mm)', 'Lateral displacement (mm)')):
        values = np.array([[v[field]*1000 for v in variants] for variants in by_setup.values()])
        axis.plot(CANDIDATE_FORCES_N, values.T, color='#64748b', alpha=.12, linewidth=.6)
        axis.fill_between(CANDIDATE_FORCES_N, *np.quantile(values, [.1, .9], axis=0), color='#208b8b', alpha=.17)
        axis.plot(CANDIDATE_FORCES_N, np.median(values, axis=0), color='#176d6d', label='Median across 150 setups')
        axis.set(xlabel='Commanded horizontal TCP force (N)', ylabel=label)
        axis.spines[['right', 'top']].set_visible(False)
        axis.grid(alpha=.2)
    for distance in sorted(set(TARGET_DISTANCES_M)):
        axes[0].axhline(distance*1000, color='#ad5a35', linestyle='--', alpha=.7, linewidth=.8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Simulator force response on the complete 13-force grid')
    fig.supxlabel('Thin lines are complete setups. Shading spans the 10th–90th percentiles. Dashed lines mark the original target distances.', fontsize=8)
    fig.savefig(output / 'force_response.png', dpi=180)
    plt.close(fig)


def run(data_dir, study_dir, output, workers):
    cfg = json.loads((data_dir / 'dataset.json').read_text())
    original = study_dir / 'target_control'
    old_cache = original / 'trajectories'
    episodes = _episodes(data_dir, cfg)
    assert len(episodes) == 150
    assert set(episodes) == {(cube, ep) for cube in range(200, 230) for ep in range(5)}
    old_files = sorted(old_cache.glob('*.npz'))
    assert len(old_files) == 1538
    fingerprint_paths = [*old_files, original/'trials.npz', original/'protocol.json', study_dir/'results.csv',
                         data_dir/'dataset.json', *sorted((data_dir/'test').glob('cube_*.npz'))]
    source_protocol = json.loads((original/'protocol.json').read_text())
    for name, expected in source_protocol['source_sha256'].items():
        assert digest(name) == expected, f'Original protocol fingerprint changed: {name}'
    for name, expected in cfg['code_sha256'].items():
        assert digest(ROOT/name) == expected, f'Simulator source changed: {name}'
    original_hashes = {str(p.resolve()): digest(p) for p in fingerprint_paths}
    output.mkdir(parents=True, exist_ok=True)
    new_cache = output / 'trajectories'
    new_cache.mkdir(exist_ok=True)
    protocol = dict(original_inputs=original_hashes, original_protocol_source_sha256=source_protocol['source_sha256'],
                    simulator_source_sha256=cfg['code_sha256'], candidate_forces_n=CANDIDATE_FORCES_N.tolist(),
                    original_target_distances_m=list(TARGET_DISTANCES_M), tolerance_m=TARGET_TOLERANCE_M,
                    exploratory_offline_targets_m=list(EXPLORATORY_TARGETS_M),
                    oracle='Minimum actual XY Euclidean error over the same 13-force grid; success requires observed rest',
                    scope='Post-study discrete-grid oracle diagnostic; no model fitting or new model-based target selection')
    manifest = output/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('Diagnostic cache has a different simulator or immutable input protocol')
    write_json(manifest, protocol)
    paths = {}
    jobs = []
    for (cube, ep), episode in episodes.items():
        for candidate in range(len(CANDIDATE_FORCES_N)):
            name = filename(cube, ep, candidate)
            source = old_cache/name
            if not source.exists():
                source = new_cache/name
                if not source.exists():
                    jobs.append((cube, ep, candidate, episode))
            paths[cube, ep, candidate] = source
    print(f'Oracle grid uses {len(old_files)} immutable trajectories; {len(jobs)} missing variants to simulate.', flush=True)
    if jobs:
        worker = partial(_simulate_variant, cfg=cfg, cache_dir=new_cache)
        with mp.get_context('spawn').Pool(min(workers, len(jobs)), initializer=start_worker) as pool:
            for count, _ in enumerate(pool.imap_unordered(worker, jobs), 1):
                if count % 25 == 0 or count == len(jobs):
                    print(f'Oracle diagnostic simulation {count}/{len(jobs)}', flush=True)
    assert len(paths) == 1950 and all(p.exists() for p in paths.values())
    assert set(p.name for p in old_files) <= {p.name for p in paths.values()}
    by_setup, grid = {}, []
    for key, episode in episodes.items():
        variants = [read_variant(paths[key + (candidate,)], episode, candidate, cfg) for candidate in range(13)]
        by_setup[key] = variants
        grid.extend(variants)
    cases = [oracle_case(episode, by_setup[key], TARGET_DISTANCES_M[key[1]]) for key, episode in episodes.items()]
    print(json.dumps({'original_targets_oracle': summarize_cases(cases)}, indent=2), flush=True)
    with np.load(original/'trials.npz', allow_pickle=False) as saved:
        trials = {k: saved[k] for k in saved.files}
    audited = audit_choices(trials, by_setup, {(v['cube_id'], v['episode_id']): v for v in cases})
    condition_summary = {}
    for condition in FINAL_COUNTS:
        selected = [v for v in audited if v['condition'] == condition]
        condition_summary[condition] = dict(actual_error_m=distribution([v['actual_error_m'] for v in selected]),
            actual_regret_m=distribution([v['actual_regret_m'] for v in selected]),
            success_rate=float(np.mean([v['actual_success'] for v in selected])),
            oracle_force_selection_rate=float(np.mean([v['selected_oracle_force'] for v in selected])),
            oracle_attainable_success_rate=float(np.mean([v['oracle_success'] for v in selected])))
    exploratory = [oracle_case(episode, by_setup[key], distance) for distance in EXPLORATORY_TARGETS_M
                   for key, episode in episodes.items()]
    by_goal = {str(distance): summarize_cases([v for v in cases if v['target_distance_m'] == distance])
               for distance in sorted(set(TARGET_DISTANCES_M))}
    by_cube = [dict(cube_id=cube, mean_oracle_error_m=float(np.mean([v['oracle_error_m'] for v in cases if v['cube_id']==cube])),
                    oracle_success_rate=float(np.mean([v['oracle_success'] for v in cases if v['cube_id']==cube])))
               for cube in range(200,230)]
    for name, records in [('force_grid.csv',grid), ('oracle_cases.csv',cases), ('oracle_by_cube.csv',by_cube),
                           ('chosen_force_audit.csv',audited), ('exploratory_oracle_cases.csv',exploratory)]:
        write_csv(output/name, records)
    plot_response(by_setup, output)
    for name, expected in original_hashes.items():
        assert digest(name) == expected, f'Original study artifact changed during diagnostics: {name}'
    summary = dict(original_targets=summarize_cases(cases), original_targets_by_goal=by_goal,
                   conditions=condition_summary,
                   exploratory_offline_oracles={str(distance): summarize_cases([v for v in exploratory if v['target_distance_m']==distance])
                                               for distance in EXPLORATORY_TARGETS_M},
                   audit=dict(complete_grid_trajectories=len(grid), reused_original_trajectories=len(old_files),
                              diagnostic_trajectories=len(grid)-len(old_files), audited_planner_choices=len(audited),
                              chosen_force_equals_model_argmin=True, all_ready_states_and_actions_verified=True,
                              all_endpoints_finite=True, all_trajectories_stopped=all(v['actual_stopped'] for v in grid),
                              original_artifacts_unchanged=True,
                              maximum_ready_state_difference=max(v['ready_state_max_difference'] for v in grid)),
                   diagnostic_code_sha256=digest(__file__),
                   trajectory_sha256={str(path.resolve()):digest(path) for path in paths.values()})
    write_json(output/'summary.json', summary)
    print(json.dumps({'audit':summary['audit'], 'oracle_mean_error_m':summary['original_targets']['error_m']['mean'],
                      'oracle_success_rate':summary['original_targets']['success_rate']}, indent=2), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data',type=Path,default=Path('runs/lift_slide_belief/three_task/data'))
    cli.add_argument('--study',type=Path,default=Path('runs/lift_slide_belief/three_task/study'))
    cli.add_argument('--output',type=Path,default=Path('runs/lift_slide_belief/three_task/diagnostics/oracle'))
    cli.add_argument('--workers',type=int,default=8)
    args = cli.parse_args()
    if args.workers < 1:
        cli.error('workers must be positive')
    run(args.data.resolve(),args.study.resolve(),args.output.resolve(),args.workers)


if __name__ == '__main__':
    main()
