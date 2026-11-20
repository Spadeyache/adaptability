"""Pooled moving-row metrics and ten-step autonomous forecasts for frozen baselines."""
import numpy as np
import torch

from .data import TASKS, episode_bounds, history_indices


HORIZON = 10
STATE_DIM = 51


def _selection(selection, count):
    if selection is None:
        return np.ones(count, dtype=bool)
    value = np.asarray(selection)
    if value.dtype == bool:
        if value.shape != (count,):
            raise ValueError('Selection mask must align with the complete observations')
        return value.copy()
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError('Selection must be a Boolean row mask or unique integer indices')
    if np.any((value < 0) | (value >= count)) or len(np.unique(value)) != len(value):
        raise ValueError('Selection contains repeated or out-of-range rows')
    mask = np.zeros(count, dtype=bool)
    mask[value] = True
    return mask


def _history_tokens(model, state, action, following):
    # The final 51 entries are a normalized next state, not a normalized delta.
    current = (torch.cat((state, action), -1) - model.input_mean) / model.input_sd
    future = (following - model.input_mean[:STATE_DIM]) / model.input_sd[:STATE_DIM]
    return torch.cat((current, future), -1)


def _initial_history(model, indices, lengths, state, action, following):
    device = model.input_mean.device
    valid = torch.as_tensor(indices >= 0, device=device)
    lookup = torch.as_tensor(np.maximum(indices, 0), device=device)
    tokens = _history_tokens(model, state[lookup], action[lookup], following[lookup])
    tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
    return tokens, torch.as_tensor(lengths, dtype=torch.long, device=device)


def _append_history(history, lengths, token):
    """Append a completed predicted transition to each finite rolling window."""
    capacity = history.shape[1]
    full = lengths == capacity
    if capacity > 1:
        history[full, :-1] = history[full, 1:].clone()
    position = torch.minimum(lengths, lengths.new_full(lengths.shape, capacity - 1))
    history[torch.arange(len(history), device=history.device), position] = token
    return history, torch.clamp(lengths + 1, max=capacity)


def _aggregate(mask, selected, moving, starts, one_sse, targets, one_valid, terminal_sse, rollout_valid):
    one = mask[moving]
    roll = mask[starts]
    target = targets[one]
    count, rollout_count = int(one.sum()), int(roll.sum())
    failures = int((~one_valid[one]).sum())
    rollout_failures = int((~rollout_valid[roll]).sum())
    sst = float(np.square(target - target.mean(0)).sum()) if count else None
    sse = float(one_sse[one].sum()) if count and not failures else None
    return dict(
        selected_rows=int(np.sum(mask & selected)), moving_rows=count,
        one_step_valid_rows=count-failures, one_step_nonfinite_rows=failures,
        one_step_mse=sse/(count*STATE_DIM) if sse is not None else None,
        one_step_r2=1-sse/sst if sse is not None and sst > 0 else None,
        one_step_sse=sse, one_step_sst=sst,
        rollout_starts=rollout_count, rollout_valid_starts=rollout_count-rollout_failures,
        rollout_nonfinite_starts=rollout_failures,
        rollout_10_mse=float(terminal_sse[roll].mean()/STATE_DIM)
            if rollout_count and not rollout_failures else None,
    )


@torch.inference_mode()
def evaluate(model, rows, selection=None, batch_size=1024):
    """Evaluate selected rows while retaining causal, same-episode B1 histories.

    Inputs must contain complete ordered episodes. Selection affects scored rows
    only. One-step targets are normalized deltas. Rollouts start at each selected
    moving row with ten remaining transitions; all 51 state coordinates are then
    predicted, using only the recorded actions and B1's own predicted history.
    Any nonfinite prediction makes its aggregate metric null and increments an
    explicit failure count. Returned values are JSON-serializable.
    """
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    count = len(rows['state'])
    if not count or np.shape(rows['state']) != (count, STATE_DIM):
        raise ValueError('Expected complete nonempty episodes with 51-dimensional states')
    if np.shape(rows['next_state']) != (count, STATE_DIM) or np.shape(rows['action']) != (count, 8):
        raise ValueError('State, next-state, and action rows must align')
    chosen = _selection(selection, count)
    begin, end = episode_bounds(rows)
    if not np.isin(rows['task'], np.arange(len(TASKS))).all():
        raise ValueError('Unknown task code')
    moving = np.flatnonzero(chosen & np.asarray(rows['cube_moving'], dtype=bool))
    starts = moving[moving + HORIZON <= end[moving]]
    kind = str(model.kind).upper()
    if kind not in ('B0', 'B1', 'B2'):
        raise ValueError('Expected a B0, B1, or B2 baseline')
    history_k = int(model.history_k) if kind == 'B1' else 0
    if kind == 'B1' and history_k < 1:
        raise ValueError('B1 requires a positive history length')
    if history_k:
        histories, history_lengths = history_indices(rows, history_k)
    device, dtype = model.input_mean.device, model.input_mean.dtype
    state, action, following = [torch.as_tensor(rows[key], dtype=dtype, device=device)
                                for key in ('state', 'action', 'next_state')]
    if not all(torch.isfinite(value).all() for value in (state, action, following)):
        raise ValueError('Observed inputs and targets must be finite')
    oracle = None
    if kind == 'B2':
        oracle = torch.as_tensor(np.column_stack((rows['true_mass'], rows['true_friction'])),
                                 dtype=dtype, device=device)
        if not torch.isfinite(oracle).all():
            raise ValueError('Oracle mass and friction must be finite')
    one_sse = np.empty(len(moving), dtype=np.float64)
    targets = np.empty((len(moving), STATE_DIM), dtype=np.float64)
    one_valid = np.empty(len(moving), dtype=bool)
    terminal_sse = np.empty(len(starts), dtype=np.float64)
    rollout_valid = np.empty(len(starts), dtype=bool)
    was_training = model.training
    model.eval()
    try:
        for offset in range(0, len(moving), batch_size):
            batch = moving[offset:offset+batch_size]
            idx = torch.as_tensor(batch, device=device)
            history = lengths = None
            if history_k:
                history, lengths = _initial_history(model, histories[batch], history_lengths[batch],
                                                     state, action, following)
            normalized = (torch.cat((state[idx], action[idx]), 1)-model.input_mean)/model.input_sd
            target = (following[idx]-state[idx]-model.delta_mean)/model.delta_sd
            labels = (oracle[idx]-model.oracle_mean)/model.oracle_sd if oracle is not None else None
            prediction = model(normalized, history=history, lengths=lengths, oracle=labels)
            valid = torch.isfinite(prediction).all(1)
            errors = (prediction.double()-target.double()).square().sum(1)
            output = slice(offset, offset+len(batch))
            one_valid[output] = valid.cpu().numpy()
            one_sse[output] = errors.cpu().numpy()
            targets[output] = target.double().cpu().numpy()
        for offset in range(0, len(starts), batch_size):
            batch = starts[offset:offset+batch_size]
            idx = torch.as_tensor(batch, device=device)
            history = lengths = None
            if history_k:
                history, lengths = _initial_history(model, histories[batch], history_lengths[batch],
                                                     state, action, following)
            current = state[idx].clone()
            valid = torch.ones(len(batch), dtype=torch.bool, device=device)
            labels = oracle[idx] if oracle is not None else None
            for step in range(HORIZON):
                command = action[idx+step]
                # advance receives raw states/actions/oracle labels; history is already normalized.
                predicted = model.advance(current, command, history=history, lengths=lengths, oracle=labels)
                valid &= torch.isfinite(predicted).all(1)
                if history_k:
                    token = _history_tokens(model, current, command, predicted)
                    history, lengths = _append_history(history, lengths, token)
                current = predicted
            terminal = following[idx+HORIZON-1]
            errors = ((current.double()-terminal.double())/model.delta_sd.double()).square().sum(1)
            output = slice(offset, offset+len(batch))
            terminal_sse[output] = errors.cpu().numpy()
            rollout_valid[output] = valid.cpu().numpy()
    finally:
        model.train(was_training)

    def summarize(mask):
        return _aggregate(mask, chosen, moving, starts, one_sse, targets, one_valid, terminal_sse, rollout_valid)

    all_rows = np.ones(count, dtype=bool)
    tasks = {name:np.asarray(rows['task']) == code for code, name in enumerate(TASKS)}
    per_cube = {}
    for cube in np.unique(np.asarray(rows['cube_id'])[chosen]):
        mask = np.asarray(rows['cube_id']) == cube
        per_cube[str(int(cube))] = dict(overall=summarize(mask),
                                      per_task={name:summarize(mask & task) for name,task in tasks.items()})
    profiles, early_late = {}, {}
    for name, mask in {'overall':all_rows, **tasks}.items():
        profiles[name] = [dict(step=int(step), **summarize(mask & (rows['step'] == step)))
                          for step in np.unique(np.asarray(rows['step'])[chosen & mask])]
        early_late[name] = dict(early=summarize(mask & (rows['step'] < 5)),
                               late=summarize(mask & (rows['step'] >= 5)))
    return dict(kind=kind, history_k=history_k, overall=summarize(all_rows),
        per_task={name:summarize(mask) for name,mask in tasks.items()}, per_cube=per_cube,
        step_profile=dict(overall=profiles.pop('overall'), per_task=profiles),
        early_late=dict(overall=early_late.pop('overall'), per_task=early_late),
        subset_metadata=dict(total_rows=count, selected_rows=int(chosen.sum()),
            selected_episode_count=len(np.unique(begin[chosen])),
            selected_cube_count=len(np.unique(np.asarray(rows['cube_id'])[chosen])),
            selected_episode_ids=np.unique(np.asarray(rows['episode_id'])[chosen]).astype(int).tolist(),
            moving_threshold_m_s=.01, moving_mask='Stored pre-action cube_moving flag',
            rollout_horizon_transitions=HORIZON,
            moving_starts_without_full_horizon=int(len(moving)-len(starts)),
            early_definition='Literal episode step < 5; late is step >= 5',
            history_scope='Completed transitions within the same cube/task/episode; selection does not remove history rows',
            aggregation='Pooled rows and 51 outputs; R2 uses per-feature centered normalized-delta targets',
            nonfinite_policy='Any nonfinite included prediction makes its group metric null; counts remain visible'))
