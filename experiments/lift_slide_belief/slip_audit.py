"""Read-only audit of paired fixed/random grip-force simulator datasets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .collect import physics
from .slip_data import FIXED_GRIP_CAP_N, classify_lift, grip_cap_for_episode, project_rows


def _read(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _stats(values):
    values = np.asarray(values)
    return dict(count=len(values), minimum=float(values.min()), maximum=float(values.max()),
                mean=float(values.mean()), median=float(np.median(values)))


def audit(root, original, output=None):
    root, original = Path(root), Path(original)
    cfg = json.loads((root / 'fixed/data/dataset.json').read_text())
    other_cfg = json.loads((root / 'random/data/dataset.json').read_text())
    assert cfg['grip_arm'] == 'fixed' and other_cfg['grip_arm'] == 'random'
    assert {k: v for k, v in cfg.items() if k != 'grip_arm'} == {
        k: v for k, v in other_cfg.items() if k != 'grip_arm'}
    source_hashes = {}
    repo = Path(__file__).resolve().parents[2]
    for name, expected in cfg['code_sha256'].items():
        actual = hashlib.sha256((repo / name).read_bytes()).hexdigest()
        assert actual == expected, f'Frozen source changed: {name}'
        source_hashes[name] = actual
    original_cfg = json.loads((original / 'dataset.json').read_text())
    counters = {arm: Counter() for arm in ('fixed', 'random')}
    rises = {arm: [] for arm in counters}
    cap_samples = {task: [] for task in (0, 1)}
    slide_endpoint_differences = []
    prove_moving = []
    per_cube = []
    raw_fields = None
    shared_fields = None
    contact_max = 0.
    contact_max_location = None
    phase_contact_peaks = dict(approach=0., pulse=0., withdrawal=0., coast=0.)
    total = cfg['train_cubes'] + cfg['test_cubes']
    for cube in range(total):
        split = 'train' if cube < cfg['train_cubes'] else 'test'
        old_split = 'train' if cube < original_cfg['train_cubes'] else 'test'
        fixed = _read(root / 'fixed/data' / split / f'cube_{cube:04}.npz')
        random = _read(root / 'random/data' / split / f'cube_{cube:04}.npz')
        old = _read(original / old_split / f'cube_{cube:04}.npz')
        assert set(fixed) == set(random)
        raw_fields, shared_fields = list(old), list(fixed)
        for name in old:
            np.testing.assert_array_equal(fixed[name], old[name], err_msg=f'Original field {name}, cube {cube}')
        mass, friction = physics(cfg['seed'], cube, cfg)
        assert np.all(fixed['true_mass'] == mass) and np.all(fixed['true_friction'] == friction)
        paired_metadata = ('cube_id', 'task', 'episode_id', 'step', 'true_mass', 'true_friction',
                           'reset_seed', 'motion_direction', 'lift_speed_m_s', 'lift_height_m', 'slide_speed_m_s')
        for name in paired_metadata:
            np.testing.assert_array_equal(fixed[name], random[name], err_msg=f'Pairing {name}, cube {cube}')
        shove = fixed['task'] == 2
        for name in fixed:
            np.testing.assert_array_equal(fixed[name][shove], random[name][shove],
                                          err_msg=f'Shared shove {name}, cube {cube}')
        prefix = (fixed['task'] < 2) & (fixed['step'] < 45)
        state_fields = ['state', 'action', 'next_state', 'relative_position', 'relative_velocity',
                        'finger_position_error', 'next_relative_position', 'next_relative_velocity',
                        'next_finger_position_error']
        for name in state_fields:
            np.testing.assert_array_equal(fixed[name][prefix], random[name][prefix],
                                          err_msg=f'Pre-switch {name}, cube {cube}')
        cube_moving = 0
        for arm, rows in (('fixed', fixed), ('random', random)):
            assert all(np.isfinite(v).all() for v in rows.values())
            assert rows['state'].shape == (sum(cfg['task_steps']) * 10, 43)
            assert rows['action'].shape == (len(rows['state']), 8)
            assert np.all(rows['cube_id'] == cube)
            projected = project_rows(rows)
            assert projected['state'].shape[1] == 51
            for name in ('relative_position', 'relative_velocity', 'relative_velocity_instantaneous'):
                assert rows[name].shape == (len(rows['state']), 3)
            for name in ('finger_position_error', 'finger_target_position', 'finger_contact_normal_n'):
                assert rows[name].shape == (len(rows['state']), 2)
            assert np.all(rows['finger_contact_normal_n'] >= 0)
            low, high = cfg['grip_force_limits_n']
            issued = low + (rows['action'][:, 4] + 1) * .5 * (high-low)
            np.testing.assert_allclose(rows['issued_grip_force_cap_n'],
                                       np.repeat(issued[:, None], 2, axis=1), atol=3e-6, rtol=0)
            for task in range(3):
                for episode in range(10):
                    mask = (rows['task'] == task) & (rows['episode_id'] == episode)
                    ep = {name: value[mask] for name, value in rows.items()}
                    assert len(ep['state']) == cfg['task_steps'][task]
                    np.testing.assert_array_equal(ep['step'], np.arange(cfg['task_steps'][task]))
                    np.testing.assert_array_equal(projected['next_state'][mask][:-1],
                                                  projected['state'][mask][1:])
                    for name in ('relative_position', 'relative_velocity', 'relative_velocity_instantaneous',
                                 'finger_position_error', 'tcp_pose', 'finger_target_position',
                                 'finger_contact_normal_n', 'finger_drive_cap_n'):
                        np.testing.assert_array_equal(ep['next_'+name][:-1], ep[name][1:])
                    np.testing.assert_array_equal(ep['relative_velocity'][0], np.zeros(3))
                    np.testing.assert_array_equal(ep['next_relative_velocity'],
                                                  (ep['next_relative_position']-ep['relative_position'])*20)
                    for pref in ('', 'next_'):
                        np.testing.assert_allclose(ep[pref+'finger_position_error'],
                                                   ep[pref+'finger_target_position']-ep[pref+'state'][:, 7:9],
                                                   atol=4e-9, rtol=0)
                    counters[arm]['episodes'] += 1
                    counters[arm]['rows'] += len(ep['state'])
                    counters[arm][f'task_{task}_episodes'] += 1
                    if task < 2:
                        wanted = FIXED_GRIP_CAP_N if arm == 'fixed' else grip_cap_for_episode(
                            cfg['seed'], cube, task, episode)
                        np.testing.assert_allclose(ep['issued_grip_force_cap_n'][:45],
                                                   FIXED_GRIP_CAP_N, atol=3e-6, rtol=0)
                        np.testing.assert_allclose(ep['issued_grip_force_cap_n'][45:], wanted, atol=3e-6, rtol=0)
                        if arm == 'random':
                            cap_samples[task].append(wanted)
                    if task == 0:
                        label = classify_lift(ep)
                        for name, expected in [('grasp_acquired', label['acquisition_step'] >= 0),
                                               ('slip_after_acquisition', label['outcome'] == 'slip_after_acquisition'),
                                               ('rotating_loss_after_acquisition', label['outcome'] == 'rotating_loss_after_acquisition'),
                                               ('acquisition_step', label['acquisition_step']),
                                               ('slip_step', label['slip_step'])]:
                            assert np.all(ep[name] == expected)
                        counters[arm][label['outcome']] += 1
                        counters[arm]['acquired_before_cap_switch'] += int(0 <= label['acquisition_step'] < 45)
                        counters[arm]['clean_slip_at_or_after_cap_switch'] += int(
                            label['outcome'] == 'slip_after_acquisition' and label['slip_step'] >= 45)
                        if episode < 5:
                            counters[arm]['first_five_lifts_'+label['outcome']] += 1
                        rises[arm].append(label['maximum_cube_rise_m'])
                    elif task == 1 and arm == 'random':
                        slide_endpoint_differences.append(float(np.linalg.norm(
                            ep['next_state'][-1, 19:21] - fixed['next_state'][mask][-1, 19:21])))
                    elif task == 2:
                        assert np.all(ep['observed_stopped'])
                        assert np.all(ep['stop_step'] == len(ep['state'])-1)
                        commanded = np.linalg.norm(ep['action'][:, 5:7], axis=1)
                        assert commanded.max() <= cfg['shove_force_cap_n'] + 1e-6
                        peak_step = int(np.argmax(ep['contact_force_max_n']))
                        peak = float(ep['contact_force_max_n'][peak_step])
                        if peak > contact_max:
                            contact_max = peak
                            contact_max_location = dict(cube_id=cube, episode_id=episode, step=peak_step,
                                                        phase=int(ep['phase'][peak_step]))
                        start, release = int(ep['force_start_step'][0]), int(ep['shove_release_step'][0])
                        for phase, selection in dict(approach=slice(0, start), pulse=slice(start, release),
                                                     withdrawal=slice(release, release+cfg['retract_steps']),
                                                     coast=slice(release+cfg['retract_steps'], None)).items():
                            phase_contact_peaks[phase] = max(phase_contact_peaks[phase],
                                float(ep['contact_force_max_n'][selection].max()))
                        counters[arm]['stopped_shoves'] += 1
                        if arm == 'fixed' and split == 'test' and episode < 5:
                            cube_moving += int(ep['cube_moving'].sum())
            for task in (0, 1):
                if arm == 'random':
                    for block in (0, 1):
                        caps = [grip_cap_for_episode(cfg['seed'], cube, task, ep) for ep in range(block*5, block*5+5)]
                        bins = np.searchsorted(cfg['grip_probe']['log_bin_edges_n'], caps, side='right') - 1
                        np.testing.assert_array_equal(np.sort(bins), np.arange(5))
        if split == 'test':
            assert cube_moving > 0
            prove_moving.append(cube_moving)
        per_cube.append(dict(cube_id=cube, split=split, true_mass_kg=mass, true_friction=friction,
                             test_prove_moving_rows=cube_moving if split == 'test' else None))
    summary = dict(
        passed=True, cubes_per_arm=total, train_cubes=cfg['train_cubes'], test_cubes=cfg['test_cubes'],
        paired_rows=2*sum(cfg['task_steps'])*10*total, unique_simulated_episodes=50*total,
        raw_state_dimension=43, projected_state_dimension=51, action_dimension=8,
        source_sha256=source_hashes, all_frozen_sources_verified=True,
        all_fixed_original_fields_bit_exact=raw_fields, all_shared_shove_fields_bit_exact=shared_fields,
        all_pre_switch_raw_and_projected_features_bit_exact=True,
        all_projected_transition_alignment_exact=True, all_causal_gap_velocities_exact=True,
        all_action_caps_match_realized_drive_limits=True, all_random_cap_blocks_cover_five_log_bins=True,
        all_shoves_stopped=True, shove_command_force_bound_n=cfg['shove_force_cap_n'],
        maximum_horizontal_reported_normal_contact_force_n=contact_max,
        maximum_contact_scope='All steps of unchanged shove episodes, including approach; not limited to the force pulse.',
        maximum_contact_location=contact_max_location,
        maximum_shove_contact_by_phase_n=phase_contact_peaks,
        arms={arm: dict(counts=dict(counters[arm]), lift_rise_m=_stats(rises[arm])) for arm in counters},
        random_cap_n={str(task): _stats(caps) for task, caps in cap_samples.items()},
        paired_slow_push_endpoint_difference_m=_stats(slide_endpoint_differences),
        test_prove_shove_moving_rows_per_cube=_stats(prove_moving), cubes=per_cube,
        acquisition_note='Airborne bilateral grasp with causal TCP-frame gap motion below 1 mm/control interval for three consecutive intervals.',
        measurement_note='Causal 20 Hz backward difference of measured TCP-frame gap; native instantaneous analytical velocity retained only as a diagnostic.',
        contact_note='Reported normal-contact impulse vectors exclude tangential friction; the 4 N bound is on commanded horizontal TCP wrench.')
    output = root / 'data_audit.json' if output is None else Path(output)
    output.write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--original', type=Path, default=Path('runs/lift_slide_belief/three_task/data'))
    parser.add_argument('--output', type=Path)
    result = audit(**vars(parser.parse_args()))
    print(json.dumps({k: v for k, v in result.items() if k not in ('cubes', 'source_sha256',
        'all_fixed_original_fields_bit_exact', 'all_shared_shove_fields_bit_exact')}, indent=2))
