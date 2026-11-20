"""Training-only horizontal-velocity sanity baseline for the revised gate.

This module does not fit a dynamics model. Per-cube true properties are read
only from cubes.json and are never repeated in the diagnostic outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
DT = .005
GRAVITY = 9.81


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def moving_coast_mask(state, phase):
    return ((phase == 'coast') &
            ((np.linalg.norm(state[:, 9:12], axis=1) > .001) |
             (np.linalg.norm(state[:, 12:15], axis=1) > .05)))


def horizontal_floor(state, combined_friction, dt=DT, clip_at_zero=False):
    """User's exact formula; optional clipping is a separate diagnostic only."""
    v = np.asarray(state)[:, 9:11]
    speed = np.linalg.norm(v, axis=1)
    direction = np.divide(v, speed[:, None], out=np.zeros_like(v),
                          where=speed[:, None] > 0)
    magnitude = np.broadcast_to(np.asarray(combined_friction) * GRAVITY * dt,
                                speed.shape)
    if clip_at_zero:
        magnitude = np.minimum(magnitude, speed)
    return -magnitude[:, None] * direction


def load_train(source=SOURCE):
    source = Path(source)
    props = {int(c['cube_id']): c for c in
             json.loads((source / 'data/cubes.json').read_text())['cubes']
             if int(c['cube_id']) < 200}
    if set(props) != set(range(200)):
        raise ValueError('Exactly the original 200 training cubes are required')
    episodes = []
    source_hashes = {'data/cubes.json': sha256(source / 'data/cubes.json')}
    for cid in range(200):
        for ep in range(10):
            path = source / 'data/episodes' / f'cube_{cid:03d}_episode_{ep:02d}.npz'
            with np.load(path, allow_pickle=False) as data:
                state = np.asarray(data['state'], dtype=np.float64)
                nxt = np.asarray(data['next_state'], dtype=np.float64)
                phase = np.asarray(data['phase']).copy()
                if not np.all(data['action'][phase == 'coast'] == 0):
                    raise ValueError('Nonzero coast action')
                if not (np.all(data['cube_id'] == cid) and
                        np.all(data['episode_id'] == ep)):
                    raise ValueError('Episode identity mismatch')
            episodes.append(dict(cube_id=cid, episode_id=ep, state=state,
                                 next_state=nxt, phase=phase,
                                 combined_friction=props[cid]['combined_friction']))
            source_hashes[str(path.relative_to(source))] = sha256(path)
    return episodes, source_hashes


def aggregate(episodes, values, masks):
    by_cube = {}
    starts = 0
    episode_count = 0
    for ep, value, mask in zip(episodes, values, masks):
        if not np.any(mask):
            continue
        if not np.isfinite(value[mask]).all():
            raise ValueError('Nonfinite diagnostic')
        mean = float(np.mean(value[mask]))
        by_cube.setdefault(ep['cube_id'], []).append(mean)
        starts += int(np.sum(mask))
        episode_count += 1
    cube_means = {k: float(np.mean(v)) for k, v in by_cube.items()}
    return dict(mean=float(np.mean(list(cube_means.values()))) if cube_means else None,
                eligible_starts=starts, episodes=episode_count, cubes=len(by_cube),
                per_cube=[dict(cube_id=k, mean=v) for k, v in cube_means.items()])


def scale_from_episodes(episodes):
    selected = [e for e in episodes if e['episode_id'] < 8]
    masks = [moving_coast_mask(e['state'], e['phase']) for e in selected]
    values = [np.mean((e['next_state'][:, 9:11] - e['state'][:, 9:11]) ** 2, axis=1)
              for e in selected]
    result = aggregate(selected, values, masks)
    if result['mean'] is None or result['mean'] <= 0:
        raise ValueError('Nonpositive training delta normalizer')
    return result


def compute_metrics(episodes, scale):
    masks = [moving_coast_mask(e['state'], e['phase']) for e in episodes]
    truth = [e['next_state'][:, 9:11] - e['state'][:, 9:11] for e in episodes]
    pred = [horizontal_floor(e['state'], e['combined_friction']) for e in episodes]
    clipped = [horizontal_floor(e['state'], e['combined_friction'], clip_at_zero=True)
               for e in episodes]
    speeds = [np.linalg.norm(e['state'][:, 9:11], axis=1) for e in episodes]
    labels = {'requested_unclipped': pred, 'stop_clipped_diagnostic': clipped,
              'zero_delta': [np.zeros_like(v) for v in truth]}
    metrics = {}
    for label, guesses in labels.items():
        errors = [np.mean((p - y) ** 2, axis=1) for p, y in zip(guesses, truth)]
        result = aggregate(episodes, errors, masks)
        result['raw_mse_m2_s2'] = result['mean']
        result['rmse_m_s'] = float(np.sqrt(result['mean']))
        result['nmse'] = result['mean'] / scale
        metrics[label] = result
    predicted_length = [np.linalg.norm(p, axis=1) for p in pred]
    ratios = []
    perpendicular = []
    angles = []
    for e, y, speed in zip(episodes, truth, speeds):
        direction = np.divide(e['state'][:, 9:11], speed[:, None],
                              out=np.zeros_like(y), where=speed[:, None] > 0)
        parallel = -np.sum(y * direction, axis=1)
        ratios.append(parallel / (e['combined_friction'] * GRAVITY * DT))
        perpendicular.append(np.linalg.norm(y + parallel[:, None] * direction, axis=1) / DT)
        angles.append(np.degrees(np.arctan2(direction[:, 1], direction[:, 0])))
    exact_errors = [np.mean((p - y) ** 2, axis=1) for p, y in zip(pred, truth)]
    clip_errors = [np.mean((p - y) ** 2, axis=1) for p, y in zip(clipped, truth)]
    categories = {
        'horizontal_speed_gt_1mm_s': [m & (s > .001) for m, s in zip(masks, speeds)],
        'horizontal_speed_le_1mm_s': [m & (s <= .001) for m, s in zip(masks, speeds)],
        'ideal_step_would_cross_zero': [m & (s < p) for m, s, p in zip(masks, speeds, predicted_length)],
        'ideal_step_would_not_cross_zero': [m & (s >= p) for m, s, p in zip(masks, speeds, predicted_length)],
        'speed_ge_0p1m_s': [m & (s >= .1) for m, s in zip(masks, speeds)],
        'speed_ge_0p1m_s_low_rotation': [m & (s >= .1) & (np.linalg.norm(e['state'][:, 12:15], axis=1) < .05)
                                      for e, m, s in zip(episodes, masks, speeds)],
    }
    for lo, hi in [(0, .001), (.001, .01), (.01, .03), (.03, .1), (.1, .3), (.3, 1), (1, 10)]:
        categories[f'speed_{lo:g}_to_{hi:g}_m_s'] = [m & (s >= lo) & (s < hi)
                                                   for m, s in zip(masks, speeds)]
    diagnostics = {}
    for label, chosen in categories.items():
        error = aggregate(episodes, exact_errors, chosen)
        ce = aggregate(episodes, clip_errors, chosen)
        ratio = aggregate(episodes, ratios, chosen)
        perp = aggregate(episodes, perpendicular, chosen)
        diagnostics[label] = dict(
            eligible_starts=error['eligible_starts'], episodes=error['episodes'], cubes=error['cubes'],
            unclipped_nmse=None if error['mean'] is None else error['mean'] / scale,
            clipped_nmse=None if ce['mean'] is None else ce['mean'] / scale,
            observed_deceleration_over_mu_g=ratio['mean'],
            mean_perpendicular_acceleration_m_s2=perp['mean'])
    directional = []
    for lo in range(-180, 180, 30):
        chosen = [m & (s >= .1) & (a >= lo) & (a < lo + 30)
                  for m, s, a in zip(masks, speeds, angles)]
        ratio = aggregate(episodes, ratios, chosen)
        error = aggregate(episodes, exact_errors, chosen)
        directional.append(dict(direction_min_degrees=lo, direction_max_degrees=lo+30,
                                eligible_starts=ratio['eligible_starts'],
                                observed_deceleration_over_mu_g=ratio['mean'],
                                unclipped_nmse=None if error['mean'] is None else error['mean']/scale))
    return dict(metrics=metrics, diagnostics=diagnostics, high_speed_direction_bins=directional)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    # Parent freezes this file before authorizing any measurements.
    protocol = args.root / 'REVISED_PROTOCOL.md'
    if not protocol.exists():
        raise RuntimeError('Written revised protocol is required before physics scoring')
    out = args.root / 'results'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'physics_floor.json'
    if path.exists():
        raise RuntimeError('Physics floor already recorded; do not overwrite evidence')
    episodes, hashes = load_train(args.source)
    fitted = scale_from_episodes(episodes)
    scale = fitted['mean']
    result = dict(
        status='complete', source_root=str(args.source.resolve()),
        protocol_sha256=sha256(protocol), source_sha256=sha256(__file__),
        formula='delta_vxy=-combined_friction*9.81*.005*vxy/|vxy|; zero vector at zero speed; no clipping',
        normalization=dict(scalar_horizontal_delta_mse=scale,
                           definition='Equal cube, then episode, then eligible moving-coast step mean of (dvx^2+dvy^2)/2 on training IDs0..199 episodes0..7.',
                           fit_cubes=list(range(200)), fit_episodes=list(range(8)),
                           eligible_starts=fitted['eligible_starts']),
        aggregation='Equal weight cube, then episode, then eligible step. Conditional diagnostic bins average only represented episodes/cubes.',
        near_zero_threshold_nmse=.01,
        splits={name: compute_metrics(chosen, scale) for name, chosen in [
            ('fitting', [e for e in episodes if e['episode_id'] < 8]),
            ('validation', [e for e in episodes if e['episode_id'] >= 8]),
            ('all_training', episodes)]},
        input_sha256=hashes)
    floor = result['splits']['all_training']['metrics']['requested_unclipped']['nmse']
    result['near_zero'] = floor <= result['near_zero_threshold_nmse']
    with path.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(path=str(path), training_floor_nmse=floor,
                          near_zero=result['near_zero'], scalar=scale)), flush=True)


if __name__ == '__main__':
    main()
