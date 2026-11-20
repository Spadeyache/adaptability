"""Dynamics baselines with no object-latent table or test-time latent fitting.

All models predict normalized state increments using the same three-layer MLP.
B1 adds a GRU encoding of completed, same-episode transitions. B2 alone receives
privileged mass and friction. Scaling is fitted on training episodes only.
"""
from copy import deepcopy
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch import nn

from .data import episode_bounds, history_indices, write_json


MAX_EPOCHS = 1000
PATIENCE = 12
BATCH_SIZE = 1024
MIN_RELATIVE_IMPROVEMENT = 0.001
SD_FLOOR = 1e-4


class BaselineModel(nn.Module):
    """B0: current observations; B1: recent history; B2: true mass/friction.

    ``forward`` consumes normalized tensors, including history and oracle values.
    ``advance`` consumes physical state/action/oracle values, but its history is
    still normalized. Use ``normalize_history`` for observed or predicted rows.
    """

    def __init__(self, state_dim=51, action_dim=8, kind='B0', history_k=0):
        super().__init__()
        if kind not in ('B0', 'B1', 'B2'):
            raise ValueError(f'Unknown baseline: {kind}')
        if min(state_dim, action_dim) < 1:
            raise ValueError('State and action dimensions must be positive')
        if (kind == 'B1' and history_k < 1) or (kind != 'B1' and history_k != 0):
            raise ValueError('Only B1 has a positive history length')
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        self.kind, self.history_k = kind, int(history_k)
        input_dim = self.state_dim + self.action_dim
        self.history_dim = 2 * self.state_dim + self.action_dim
        if kind == 'B1':
            self.encoder = nn.GRU(self.history_dim, 64, batch_first=True)
        context_dim = {'B0': 0, 'B1': 64, 'B2': 2}[kind]
        self.network = nn.Sequential(
            nn.Linear(input_dim + context_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, self.state_dim),
        )
        for name, size in (('input', input_dim), ('delta', self.state_dim), ('oracle', 2)):
            self.register_buffer(f'{name}_mean', torch.zeros(size))
            self.register_buffer(f'{name}_sd', torch.ones(size))

    @property
    def device(self):
        return self.input_mean.device

    def normalize(self, inputs, deltas):
        return ((inputs - self.input_mean) / self.input_sd,
                (deltas - self.delta_mean) / self.delta_sd)

    def normalize_oracle(self, oracle):
        return (oracle - self.oracle_mean) / self.oracle_sd

    def normalize_history(self, state, action, next_state):
        """Normalize (s,a,s_next); next state uses state, not delta, statistics."""
        inputs = (torch.cat((state, action), dim=-1) - self.input_mean) / self.input_sd
        next_normalized = (next_state - self.input_mean[:self.state_dim]) / self.input_sd[:self.state_dim]
        return torch.cat((inputs, next_normalized), dim=-1)

    def forward(self, normalized_inputs, history=None, lengths=None, oracle=None):
        if normalized_inputs.ndim != 2 or normalized_inputs.shape[1] != len(self.input_mean):
            raise ValueError('Expected a batch of normalized state/action inputs')
        if self.kind == 'B1':
            if (history is None) != (lengths is None):
                raise ValueError('History and its lengths must be supplied together')
            if history is None:
                context = normalized_inputs.new_zeros((len(normalized_inputs), 64))
            else:
                if (history.shape != (len(normalized_inputs), self.history_k, self.history_dim)
                        or lengths.shape != (len(normalized_inputs),)):
                    raise ValueError('Unexpected history or length shape')
                # A unidirectional GRU cannot see the right padding from earlier
                # outputs. Gathering the last valid output avoids packed-sequence
                # CPU synchronization and masks every padded transition.
                encoded, _ = self.encoder(history)
                last = (lengths - 1).clamp_min(0)
                context = encoded[torch.arange(len(encoded), device=encoded.device), last]
                context = torch.where(lengths[:, None] > 0, context, torch.zeros_like(context))
            normalized_inputs = torch.cat((normalized_inputs, context), dim=-1)
        elif history is not None or lengths is not None:
            raise ValueError('Only B1 consumes history')
        if self.kind == 'B2':
            if oracle is None or oracle.shape != (len(normalized_inputs), 2):
                raise ValueError('B2 requires normalized mass/friction for each row')
            normalized_inputs = torch.cat((normalized_inputs, oracle), dim=-1)
        elif oracle is not None:
            raise ValueError('Physical labels are only allowed for B2')
        return self.network(normalized_inputs)

    def advance(self, state, action, history=None, lengths=None, oracle=None):
        inputs = (torch.cat((state, action), dim=-1) - self.input_mean) / self.input_sd
        if oracle is not None:
            oracle = self.normalize_oracle(oracle)
        delta = self(inputs, history=history, lengths=lengths, oracle=oracle)
        return state + delta * self.delta_sd + self.delta_mean


@torch.no_grad()
def _fit_moments(values, indices, mean, sd):
    """Population moments accumulated in float64, matching the existing z model."""
    sums = torch.zeros_like(mean, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    for block in indices.split(8192):
        batch = values[block].double()
        sums += batch.sum(0)
        squares += batch.square().sum(0)
    center = sums / len(indices)
    mean.copy_(center)
    sd.copy_((squares / len(indices) - center.square()).clamp_min(0).sqrt().clamp_min(SD_FLOOR))


def _validate_masks(rows, train_mask, val_mask):
    n = len(rows['state'])
    train_mask, val_mask = np.asarray(train_mask), np.asarray(val_mask)
    if (train_mask.dtype != bool or val_mask.dtype != bool
            or train_mask.shape != (n,) or val_mask.shape != (n,)):
        raise ValueError('Train and validation masks must be boolean vectors aligned to rows')
    if not train_mask.any() or not val_mask.any() or np.any(train_mask & val_mask):
        raise ValueError('Training and validation must both be nonempty and disjoint')
    if not np.all(train_mask | val_mask):
        raise ValueError('Every supplied training-cube row must have one split assignment')
    starts, _ = episode_bounds(rows)
    if not np.array_equal(train_mask, train_mask[starts]):
        raise ValueError('Training and validation must hold out whole episodes')
    return np.flatnonzero(train_mask), np.flatnonzero(val_mask)


def _history_batch(inputs, next_normalized, indices, lengths, block):
    selected = indices[block]
    valid = selected >= 0
    selected = selected.clamp_min(0)
    features = torch.cat((inputs[selected], next_normalized[selected]), dim=-1)
    return torch.where(valid[:, :, None], features, torch.zeros_like(features)), lengths[block]


def save_checkpoint(path, model, report):
    """Save model weights/statistics and training metadata atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(dict(schema_version=1, state_dim=model.state_dim, action_dim=model.action_dim,
                    kind=model.kind, history_k=model.history_k,
                    model=model.state_dict(), training=report), temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    """Return ``(frozen model, training report)`` from a baseline checkpoint."""
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved['schema_version'] != 1:
        raise ValueError('Unknown baseline checkpoint version')
    model = BaselineModel(saved['state_dim'], saved['action_dim'], saved['kind'], saved['history_k']).to(device)
    model.load_state_dict(saved['model'])
    model.eval().requires_grad_(False)
    return model, saved['training']


def train_model(rows, train_mask, val_mask, seed, kind, history_k, device, output_dir, log=None):
    """Fit one baseline; return ``(frozen best model, training report)``.

    Hyperparameters are fixed to the frozen baseline protocol. B0/B1 never read
    physical label fields. Episode IDs are used only for split/history boundaries.
    The caller must authorize and time the B0 pilot before scheduling other runs.
    """
    started = monotonic()
    train_indices, val_indices = _validate_masks(rows, train_mask, val_mask)
    torch.manual_seed(seed)
    state, action, next_state = [torch.as_tensor(rows[key], dtype=torch.float32, device=device)
                                 for key in ('state', 'action', 'next_state')]
    if (state.ndim != 2 or next_state.shape != state.shape
            or action.ndim != 2 or len(action) != len(state)):
        raise ValueError('Expected aligned state/action/next-state matrices')
    inputs, targets = torch.cat((state, action), dim=1), next_state - state
    if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
        raise ValueError('Nonfinite input transition')
    model = BaselineModel(state.shape[1], action.shape[1], kind, history_k).to(device)
    train = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    validation = torch.as_tensor(val_indices, dtype=torch.long, device=device)
    _fit_moments(inputs, train, model.input_mean, model.input_sd)
    _fit_moments(targets, train, model.delta_mean, model.delta_sd)
    inputs, targets = model.normalize(inputs, targets)
    next_normalized = histories = history_lengths = oracle = None
    if kind == 'B1':
        next_normalized = (next_state - model.input_mean[:model.state_dim]) / model.input_sd[:model.state_dim]
        indices, lengths = history_indices(rows, history_k)
        histories = torch.as_tensor(indices, dtype=torch.long, device=device)
        history_lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)
    elif kind == 'B2':
        # Do not even access these keys for either observation-only baseline.
        values = np.column_stack((rows['true_mass'], rows['true_friction']))
        oracle = torch.as_tensor(values, dtype=torch.float32, device=device)
        if oracle.shape != (len(state), 2) or not torch.isfinite(oracle).all():
            raise ValueError('Expected finite per-row oracle mass and friction')
        _fit_moments(oracle, train, model.oracle_mean, model.oracle_sd)
        oracle = model.normalize_oracle(oracle)
    del state, action, next_state

    def predict(block):
        if kind == 'B1':
            sequence, lengths = _history_batch(inputs, next_normalized, histories, history_lengths, block)
            return model(inputs[block], history=sequence, lengths=lengths)
        if kind == 'B2':
            return model(inputs[block], oracle=oracle[block])
        return model(inputs[block])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    best_mse = best_significant = float('inf')
    history = []
    last_improvement = 0
    # Record synchronized setup time so training throughput excludes data/scaling.
    if model.device.type == 'cuda':
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    training_started = monotonic()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = train[torch.randperm(len(train), generator=generator, device=device)]
        for block in order.split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            loss = (predict(block) - targets[block]).square().mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            # Match the existing trainer's per-row float32 MSE then float64 mean.
            total = torch.zeros((), dtype=torch.float64, device=device)
            for block in validation.split(4096):
                total += (predict(block) - targets[block]).square().mean(dim=1).double().sum()
            mse = (total / len(validation)).item()
        if not np.isfinite(mse):
            raise ValueError(f'Nonfinite validation MSE at epoch {epoch}')
        if mse < best_mse:
            best_mse, best_epoch = mse, epoch
            best_weights = deepcopy(model.state_dict())
        if mse < best_significant * (1 - MIN_RELATIVE_IMPROVEMENT):
            best_significant, last_improvement = mse, epoch
        record = dict(epoch=epoch, validation_mse=mse, elapsed_s=monotonic() - training_started)
        history.append(record)
        if log is not None:
            log(record)
        if epoch - last_improvement >= PATIENCE:
            break
    model.load_state_dict(best_weights)
    model.eval().requires_grad_(False)
    training_seconds = monotonic() - training_started
    report = dict(schema_version=1, kind=kind, history_k=history_k, seed=int(seed),
                  state_dim=model.state_dim, action_dim=model.action_dim,
                  best_epoch=best_epoch, best_validation_mse=best_mse, epochs=epoch,
                  stopped_for='patience' if epoch - last_improvement >= PATIENCE else 'max_epochs',
                  train_rows=len(train), validation_rows=len(validation), history=history,
                  setup_seconds=training_started - started, training_seconds=training_seconds,
                  elapsed_seconds=monotonic() - started, device=str(model.device),
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(model.device)
                  if model.device.type == 'cuda' else None,
                  training=dict(optimizer='Adam', lr=1e-3, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                                patience=PATIENCE, min_relative_improvement=MIN_RELATIVE_IMPROVEMENT,
                                loss='mean squared normalized delta error', sd_floor=SD_FLOOR),
                  normalization='Training-episode population moments only; float64 accumulation.',
                  parameter_count=sum(p.numel() for p in model.parameters()))
    output_dir = Path(output_dir)
    save_checkpoint(output_dir / 'checkpoint.pt', model, report)
    write_json(output_dir / 'training.json', report)
    return model, report
