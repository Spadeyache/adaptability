"""Observed inputs, episode boundaries, and immutable split loading."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'runs/lift_slide_belief/slip_probe/paired/fixed/data'
RESULTS = ROOT / 'results/baselines'
TASKS = ('lift', 'slow_slide', 'shove')
EXTRA = ('relative_position', 'relative_velocity', 'finger_position_error')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_rows(split, *, oracle=False, directory=DATA):
    """Physical labels are loaded only on explicit oracle requests."""
    pieces = {}
    fields = ('cube_id', 'task', 'episode_id', 'step', 'cube_moving', 'action')
    for path in sorted((Path(directory) / split).glob('cube_*.npz')):
        with np.load(path, allow_pickle=False) as saved:
            row = {key: saved[key] for key in fields}
            for prefix in ('', 'next_'):
                row[prefix + 'state'] = np.concatenate(
                    [saved[prefix + 'state'], *(saved[prefix + key] for key in EXTRA)], axis=1
                ).astype(np.float32)
            if oracle:
                row['true_mass'], row['true_friction'] = saved['true_mass'], saved['true_friction']
            for key, value in row.items():
                pieces.setdefault(key, []).append(value)
    if not pieces:
        raise ValueError(f'No {split} cubes in {directory}')
    rows = {key: np.concatenate(value) for key, value in pieces.items()}
    order = np.lexsort((rows['step'], rows['episode_id'], rows['task'], rows['cube_id']))
    return {key: value[order] for key, value in rows.items()}


def episode_bounds(rows):
    """Return episode start and exclusive end for each row; require sorted complete episodes."""
    keys = np.column_stack([rows[key] for key in ('cube_id', 'task', 'episode_id')])
    starts = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1]
    ends = np.r_[starts[1:], len(keys)]
    lengths = ends - starts
    start, end = np.repeat(starts, lengths), np.repeat(ends, lengths)
    if not np.array_equal(rows['step'], np.arange(len(keys)) - start):
        raise ValueError('Episode steps must be complete, contiguous, zero-based and ordered')
    if len(np.unique(keys[starts], axis=0)) != len(starts):
        raise ValueError('An episode appears in multiple blocks')
    return start, end


def history_indices(rows, k):
    """Last K completed transitions, in chronological order, with right padding -1."""
    start, _ = episode_bounds(rows)
    index = np.arange(len(start))
    lengths = np.minimum(index - start, k)
    offsets = np.arange(k)[None, :]
    history = index[:, None] - lengths[:, None] + offsets
    return np.where(offsets < lengths[:, None], history, -1), lengths


def masks(rows, split_file=RESULTS / 'split.json'):
    saved = json.loads(Path(split_file).read_text())
    valid = {tuple(entry) for entry in saved['train_validation_episodes']}
    test = {(x['cube_id'], x['task_code'], x['episode_id']): x['role'] for x in saved['test_episodes']}
    starts, ends = episode_bounds(rows)
    unique = np.unique(starts)
    roles = np.empty(len(starts), dtype='<U10')
    for i in unique:
        key = tuple(int(rows[field][i]) for field in ('cube_id', 'task', 'episode_id'))
        role = test[key] if key[0] >= 200 else ('validation' if key in valid else 'train')
        roles[i:ends[i]] = role
    return {role: roles == role for role in ('train', 'validation', 'context', 'eval')}
