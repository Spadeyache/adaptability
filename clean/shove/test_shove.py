"""Independent saved-trace audit; never simulates or trains anything.

Run after collection: python -m clean.shove.test_shove --expected-trials 126
The physical momentum balance checks that reported contact impulses really
account for force transmitted by the free-x pusher, including its inertia.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(expected_trials):
    rows = json.loads((ROOT / 'results/metrics.json').read_text())
    plan = json.loads((ROOT / 'data/trials.json').read_text())
    assert len(rows) == expected_trials, (len(rows), expected_trials)
    assert len({r['trial_id'] for r in rows}) == len(rows)
    for name, expected in json.loads((ROOT / 'data/seal.json').read_text()).items():
        assert sha(ROOT / 'data' / name) == expected, name
    for name, expected in json.loads((ROOT / 'results/source_seal.json').read_text()).items():
        assert sha(ROOT.parent / name) == expected, name

    policy = ast.parse((ROOT / 'script.py').read_text())
    imports = [n.module for n in ast.walk(policy) if isinstance(n, ast.ImportFrom)]
    assert imports == ['dataclasses']
    forbidden = {'mass', 'mass_kg', 'friction', 'cube_id', 'properties', 'open', 'load'}
    assert not ({n.id for n in ast.walk(policy) if isinstance(n, ast.Name)} & forbidden)
    assert not any(isinstance(n, ast.Import) for n in ast.walk(policy))
    simulation = ast.parse((ROOT / 'simulation.py').read_text())
    prop_uses = [n for n in ast.walk(simulation) if isinstance(n, ast.Name) and n.id == 'properties']
    assert len(prop_uses) == 1, 'Properties must be used only by the physical-property setter'
    setter = [n for n in ast.walk(simulation) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == 'set_cube_properties']
    assert len(setter) == 1 and prop_uses[0] in setter[0].args

    reference_actions = {}
    cap_ratios = []
    momentum_residuals = []
    max_points = 0
    stopped_count = 0
    for row in rows:
        tid = row['trial_id']
        assert {k: row[k] for k in plan[tid]} == plan[tid]
        path = ROOT / 'data/trajectories' / f'trial_{tid:03d}.npz'
        assert sha(path) == row['trace_sha256'], tid
        with np.load(path, allow_pickle=False) as trace:
            assert not (set(trace.files) & forbidden), trace.files
            state = trace['state']
            action = trace['action']
            phase = trace['phase']
            time = trace['time_s']
            dt = time[1] - time[0]
            r = int(trace['release_index'])
            s = int(trace['stop_index'])
            force = trace['contact_force_bound_n']
            impulse = trace['contact_impulse']
            points = trace['contact_point_count']
            assert len(state) == len(action) + 1 == len(time)
            assert np.all(np.isfinite(state)) and np.all(np.isfinite(force))
            assert np.allclose(np.diff(time), dt, rtol=0, atol=1e-12)
            assert r == 980 and dt == .001
            assert list(np.bincount(phase[:r])) == [800, 100, 80]
            assert np.all(phase[r:] == 3) and np.all(action[r:] == 0)
            assert np.all(action[phase == 2] == .95 * row['cap_n'])
            if row['cap_n'] not in reference_actions:
                reference_actions[row['cap_n']] = action[:r].copy()
            assert np.array_equal(action[:r], reference_actions[row['cap_n']])
            # Conservative impulse aggregation bounds the vector resultant.
            resultant = np.linalg.norm(impulse, axis=1) / dt
            assert np.all(force + 1e-12 >= resultant)
            assert np.allclose(force[points <= 1], resultant[points <= 1], rtol=1e-10, atol=1e-12)
            assert np.max(force) <= row['cap_n'] * 1.00001
            assert np.max(force[r:]) == 0 and np.max(resultant[r:]) == 0
            assert np.isclose(row['max_contact_force_n'], force.max(), rtol=1e-12)
            assert np.isclose(row['mean_push_contact_force_n'], force[phase == 2].mean(), rtol=1e-12)
            cap_ratios.append(float(force.max() / row['cap_n']))
            max_points = max(max_points, int(points.max()))

            # m_pusher * delta_v = commanded impulse - transmitted impulse.
            # X is the pusher's sole unconstrained DOF. Exclude withdrawal.
            expected_transmitted = action[:r] * dt - .005 * np.diff(state[:r+1, 16])
            residual = np.abs(expected_transmitted - np.abs(impulse[:r, 0]))
            assert residual.max() < 1e-7, (tid, residual.max())
            momentum_residuals.append(float(residual.max()))

            # Independently locate the first 100-sample post-release rest window.
            resting = ((np.linalg.norm(state[:, 3:6], axis=1) < .001)
                       & (np.linalg.norm(state[:, 6:9], axis=1) < .05))
            windows = np.convolve(resting[r+1:].astype(int), np.ones(100, dtype=int), mode='valid')
            candidates = np.flatnonzero(windows == 100)
            assert candidates.size > 0 and s >= r
            independent_s = r + 1 + int(candidates[0])
            if independent_s == r + 1 and resting[r]:
                independent_s = r
            assert s == independent_s, (tid, s, independent_s)
            stopped_count += 1
            assert np.isclose(row['time_to_stop_s'], (s-r)*dt, rtol=0, atol=1e-12)
            distance = np.linalg.norm(state[s, :2] - state[r, :2])
            speed = np.linalg.norm(state[r, 3:5])
            assert np.isclose(row['stopping_distance_m'], distance, rtol=1e-12, atol=1e-12)
            assert np.isclose(row['launch_speed_m_s'], speed, rtol=1e-12, atol=1e-12)

    result = dict(status='passed', trials=len(rows), stopped_trials=stopped_count,
                  max_measured_contact_force_over_cap=max(cap_ratios),
                  max_pusher_momentum_balance_residual_Ns=max(momentum_residuals),
                  max_contact_points=max_points,
                  checks=['original and probe seals', 'raw trajectory hashes', 'property-blind fixed script',
                          'same approach/preload/push commands across properties and repeats',
                          'contact impulse aggregation and actual force cap',
                          'pusher momentum balance including its inertia',
                          'zero contact force after release', 'independent stop-window detection',
                          'saved launch speed, stopping distance, and stopping time recomputed'])
    (ROOT / 'results/audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-trials', type=int, default=126)
    print(json.dumps(audit(parser.parse_args().expected_trials), indent=2))
