"""Fit a shared one-step dynamics MLP and one four-dimensional vector per train cube."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch import nn


LATENT_DIM = 4


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 1000
    patience: int = 12
    batch_size: int = 1024
    heldout_episodes: int = 2  # per task per training cube
    split_seed: int = 721
    z_penalty: float = 1e-4
    min_relative_improvement: float = 0.001  # reset patience after this much progress
    inference_prior: str = 'training_distribution'
    prior_variance_floor: float = 1e-4  # per-coordinate variance, giving a 0.01 SD floor

    def __post_init__(self):
        if min(self.max_epochs, self.patience, self.batch_size, self.heldout_episodes) < 1:
            raise ValueError("Epochs, patience, batch size and held-out episodes must be positive")
        if not 0 <= self.min_relative_improvement < 1 or not np.isfinite(self.z_penalty) or self.z_penalty < 0:
            raise ValueError("Invalid early-stopping tolerance or z penalty")
        if self.inference_prior not in ('training_distribution', 'zero'):
            raise ValueError('Unknown inference prior')
        if not np.isfinite(self.prior_variance_floor) or self.prior_variance_floor <= 0:
            raise ValueError('The inference prior variance floor must be positive and finite')


def _episode_split(rows, config):
    cube_ids, cube_index = np.unique(rows['cube_id'], return_inverse=True)
    if len(cube_ids) < 2:
        raise ValueError("The shuffled-z comparison requires at least two training cubes")
    keys = np.column_stack((cube_index, rows['task'], rows['episode_id']))
    episodes, index = np.unique(keys, axis=0, return_inverse=True)
    selected = np.zeros(len(episodes), dtype=bool)
    rng = np.random.default_rng(config.split_seed)
    tasks = np.unique(episodes[:, 1])
    if tasks.tolist() not in ([0, 1], [0, 1, 2]):
        raise ValueError("Expected lift (0), slide (1), and optional shove (2) task codes")
    for cube in range(len(cube_ids)):
        for task in tasks:
            available = np.flatnonzero((episodes[:, 0] == cube) & (episodes[:, 1] == task))
            if len(available) <= config.heldout_episodes:
                raise ValueError(f"Cube {cube_ids[cube]} task {task} has too few episodes to split")
            selected[rng.choice(available, config.heldout_episodes, replace=False)] = True
    held = episodes[selected].copy()
    held[:, 0] = cube_ids[held[:, 0]]
    return cube_ids, cube_index, selected[index], held.tolist()


class DeltaModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        input_dim = state_dim + action_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim + LATENT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, state_dim),
        )
        for name, size in (('input', input_dim), ('delta', state_dim)):
            self.register_buffer(f'{name}_mean', torch.zeros(size))
            self.register_buffer(f'{name}_sd', torch.ones(size))

    def normalize(self, inputs, deltas):
        return (inputs - self.input_mean) / self.input_sd, (deltas - self.delta_mean) / self.delta_sd

    def forward(self, normalized_inputs, z):
        return self.network(torch.cat((normalized_inputs, z), dim=-1))

    def advance(self, state, action, z):
        inputs = torch.cat((state, action), dim=-1)
        normalized = (inputs - self.input_mean) / self.input_sd
        delta = self(normalized, z) * self.delta_sd + self.delta_mean
        return state + delta


def _transitions(rows, device):
    state, action, next_state = [torch.as_tensor(rows[key], dtype=torch.float32, device=device)
                                 for key in ('state', 'action', 'next_state')]
    if (state.ndim != 2 or not len(state) or state.shape != next_state.shape
            or action.ndim != 2 or len(action) != len(state)):
        raise ValueError("Expected aligned, nonempty state, action and next_state matrices")
    inputs, delta = torch.cat((state, action), dim=1), next_state - state
    if not torch.isfinite(inputs).all() or not torch.isfinite(delta).all():
        raise ValueError("Nonfinite transition in data")
    return inputs, delta


@torch.no_grad()
def _fit_scaling(model, inputs, deltas, train):
    # Train-episode statistics only. Accumulate in double precision to retain small changes.
    for values, mean, sd in ((inputs, model.input_mean, model.input_sd),
                             (deltas, model.delta_mean, model.delta_sd)):
        sums = torch.zeros_like(mean, dtype=torch.float64)
        squares = torch.zeros_like(sums)
        for indices in train.split(8192):
            batch = values[indices].double()
            sums += batch.sum(0)
            squares += batch.square().sum(0)
        center = sums / len(train)
        mean.copy_(center)
        sd.copy_((squares / len(train) - center.square()).clamp_min(0).sqrt().clamp_min(1e-4))


@torch.no_grad()
def _row_errors(model, inputs, targets, indices, cube_index, table):
    values = []
    for block in indices.split(4096):
        prediction = model(inputs[block], table[cube_index[block]])
        values.append((prediction - targets[block]).square().mean(dim=1))
    return torch.cat(values)


def _validation_gate(model, inputs, targets, indices, cube_index, cube_ids, table, seed, prior):
    # Compare a fixed wrong donor per recipient cube on exactly the same moving rows.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cube_ids))
    donors = np.empty_like(order)
    donors[order] = np.roll(order, 1)
    required_controls = ['zero', 'shuffled'] if prior == 'zero' else ['shuffled']
    result = dict(passed=False, moving_rows=len(indices), min_gain_fraction=0.01,
                  required_controls=required_controls,
                  shuffled_donor_ids=cube_ids[donors].tolist(), bootstrap_draws=2000,
                  mse={}, comparisons={})
    if len(indices) == 0:
        return dict(result, moving_cubes=0, reason="No moving validation rows")
    cube = cube_index[indices].cpu().numpy()
    counts = np.bincount(cube, minlength=len(cube_ids))
    active = np.flatnonzero(counts)
    result['moving_cubes'] = len(active)
    totals = {}
    baseline = 'mean' if prior == 'training_distribution' else 'zero'
    baseline_z = table.mean(0).expand_as(table) if baseline == 'mean' else torch.zeros_like(table)
    for name, vectors in [('own', table), (baseline, baseline_z),
                          ('shuffled', table[torch.as_tensor(donors, device=table.device)])]:
        losses = _row_errors(model, inputs, targets, indices, cube_index, vectors).cpu().numpy()
        if not np.isfinite(losses).all():
            raise ValueError(f"Nonfinite {name} validation loss")
        totals[name] = np.bincount(cube, weights=losses, minlength=len(cube_ids))
        result['mse'][name] = float(totals[name].sum() / counts.sum())
    draws = rng.choice(active, size=(result['bootstrap_draws'], len(active)), replace=True)
    for name in (baseline, 'shuffled'):
        # Paired resampling of whole cubes keeps the step-weighted MSE estimand.
        difference = totals[name] - totals['own']
        boot = difference[draws].sum(axis=1) / counts[draws].sum(axis=1)
        interval = np.quantile(boot, [0.025, 0.975]).tolist()
        gain = result['mse'][name] - result['mse']['own']
        fraction = gain / result['mse'][name] if result['mse'][name] > 0 else 0.0
        result['comparisons'][name] = dict(difference=gain, difference_ci95=interval,
            relative_gain=fraction, passed=len(active) >= 2 and fraction >= result['min_gain_fraction'] and interval[0] > 0)
    result['passed'] = all(result['comparisons'][name]['passed'] for name in required_controls)
    return result


class TrainedModel:
    def __init__(self, model, cube_ids, train_z, validation, config):
        self.model = model.eval().requires_grad_(False)
        self.cube_ids = cube_ids
        self.train_z = train_z
        self.validation = validation
        self.config = config
        self.prior_mean = np.mean(train_z, axis=0).astype(np.float32)
        variance = np.var(train_z.astype(np.float64), axis=0)
        self.prior_sd = np.sqrt(np.maximum(variance, config.prior_variance_floor)).astype(np.float32)

    @property
    def device(self):
        return self.model.input_mean.device

    def _latent(self, z):
        value = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if value.shape != (LATENT_DIM,) or not torch.isfinite(value).all():
            raise ValueError(f'Expected one finite {LATENT_DIM}-dimensional z')
        return value

    def fit_z(self, rows, initial=None, steps=100):
        """Fit only z on the supplied adaptation episodes for one cube."""
        if steps < 1:
            raise ValueError('At least one adaptation update is required')
        if 'cube_id' in rows and len(np.unique(rows['cube_id'])) != 1:
            raise ValueError('Adaptation episodes must belong to one cube')
        inputs, targets = self.model.normalize(*_transitions(rows, self.device))
        if self.config.inference_prior == 'training_distribution':
            center, scale = self._latent(self.prior_mean), self._latent(self.prior_sd)
        else:
            center, scale = inputs.new_zeros(LATENT_DIM), inputs.new_ones(LATENT_DIM)
        start = center if initial is None else self._latent(initial)
        z = nn.Parameter(start.detach().clone())
        opt = torch.optim.Adam([z], lr=1e-2)
        with torch.enable_grad():
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                prediction = self.model(inputs, z.expand(len(inputs), -1))
                loss = (prediction - targets).square().mean()
                loss = loss + self.config.z_penalty * ((z - center) / scale).square().mean()
                loss.backward()
                opt.step()
        return self._latent(z).detach().cpu().numpy().copy()

    def fit_many_z(self, groups, steps=100, initials=None):
        """Independently adapt one latent per group in a shared forward pass.

        Each group's MSE is averaged over its own rows and state dimensions.
        Summing the group losses preserves the gradient scale of separate Adam
        fits, even when groups contain different numbers of observed steps.
        """
        if steps < 1 or not groups:
            raise ValueError('At least one group and one adaptation update are required')
        for rows in groups:
            if 'cube_id' in rows and len(np.unique(rows['cube_id'])) != 1:
                raise ValueError('Each adaptation group must belong to one cube')
        transitions = [_transitions(rows, self.device) for rows in groups]
        counts = torch.tensor([len(inputs) for inputs, _ in transitions], device=self.device)
        inputs, targets = self.model.normalize(torch.cat([pair[0] for pair in transitions]),
                                              torch.cat([pair[1] for pair in transitions]))
        group_index = torch.arange(len(groups), device=self.device).repeat_interleave(counts)
        if self.config.inference_prior == 'training_distribution':
            center, scale = self._latent(self.prior_mean), self._latent(self.prior_sd)
        else:
            center, scale = inputs.new_zeros(LATENT_DIM), inputs.new_ones(LATENT_DIM)
        start = center.expand(len(groups), -1) if initials is None else torch.as_tensor(
            initials, dtype=torch.float32, device=self.device)
        if start.shape != (len(groups), LATENT_DIM) or not torch.isfinite(start).all():
            raise ValueError('Expected one finite initial latent per adaptation group')
        z = nn.Parameter(start.detach().clone())
        optimizer = torch.optim.Adam([z], lr=1e-2)
        with torch.enable_grad():
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                predictions = self.model(inputs, z[group_index])
                row_errors = (predictions - targets).square().mean(dim=1)
                group_errors = row_errors.new_zeros(len(groups)).scatter_add_(0, group_index, row_errors)
                prior = ((z - center) / scale).square().mean(dim=1)
                loss = (group_errors / counts).sum() + self.config.z_penalty * prior.sum()
                loss.backward()
                optimizer.step()
        if not torch.isfinite(z).all():
            raise ValueError('Nonfinite adapted latent')
        return z.detach().cpu().numpy().copy()

    @torch.no_grad()
    def predict_next_state(self, state, action, z):
        """Advance the entire measured state by one issued action, in physical units."""
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if state.ndim != 1 or state.shape != self.model.delta_mean.shape or action.ndim != 1:
            raise ValueError('Expected one state vector and one action vector')
        if len(state) + len(action) != len(self.model.input_mean):
            raise ValueError('State/action dimensions disagree with the trained model')
        return self.model.advance(state, action, self._latent(z)).cpu().numpy()

    def stopping_error(self, rows, z, *, cube_xy=(19, 20), cube_velocity=(28, 29, 30)):
        """Score the autonomous endpoint of one complete, ordered shove episode."""
        return self.stopping_errors([rows], z, cube_xy=cube_xy, cube_velocity=cube_velocity)[0]

    @torch.no_grad()
    def stopping_errors(self, episodes, z, *, cube_xy=(19, 20), cube_velocity=(28, 29, 30)):
        """Batch full-state rollouts through each episode's observed resting step.

        ``stop_step`` is the inclusive zero-based terminal transition index.
        ``force_start_step`` selects the measured state at pulse onset, defaulting
        to zero for older episodes. Later states supply only the endpoint target.
        Finite endpoints
        retain their full error even when implausibly large or still moving.
        A nonfinite state yields a missing error and its first failing step.
        Callers must count these failures instead of averaging them away.
        """
        state_dim = len(self.model.delta_mean)
        action_dim = len(self.model.input_mean) - state_dim
        xy, velocity = np.asarray(cube_xy, dtype=int), np.asarray(cube_velocity, dtype=int)
        if xy.shape != (2,) or (xy < 0).any() or (xy >= state_dim).any():
            raise ValueError('Expected two valid state columns for the cube XY position')
        if velocity.shape != (3,) or (velocity < 0).any() or (velocity >= state_dim).any():
            raise ValueError('Expected three valid state columns for cube linear velocity')
        result, initial, commands, positions, start_steps = [], [], [], [], []
        for episode in episodes:
            states, actions, next_states = [np.asarray(episode[key]) for key in ('state', 'action', 'next_state')]
            if (states.ndim != 2 or states.shape[1] != state_dim or not len(states)
                    or states.shape != next_states.shape or actions.shape != (len(states), action_dim)):
                raise ValueError('Expected aligned transition arrays from one complete episode')
            if not all(np.isfinite(values).all() for values in (states, actions, next_states)):
                raise ValueError('Nonfinite recorded shove transition')
            if 'step' in episode and not np.array_equal(episode['step'], np.arange(len(states))):
                raise ValueError('Stopping error requires one complete episode in step order')
            stopped = np.asarray(episode['observed_stopped'], dtype=bool)
            terminal = np.asarray(episode['stop_step'])
            if stopped.shape != (len(states),) or terminal.shape != stopped.shape or not np.all(stopped == stopped[0]):
                raise ValueError('Expected a constant observed-stopped flag for the episode')
            if not np.all(terminal == terminal[0]):
                raise ValueError('Expected a constant stop step for the episode')
            record = dict(endpoint_error_m=None, endpoint_status='observed_not_stopped', endpoint_steps=0,
                          predicted_endpoint_xy=None, observed_endpoint_xy=None, divergence_step=None,
                          last_finite_state_max_abs=None, predicted_final_speed_m_s=None, endpoint_valid=False)
            result.append(record)
            if not stopped[0]:
                continue
            stop = int(terminal[0])
            if stop != terminal[0] or not 0 <= stop < len(states):
                raise ValueError('Observed stop step falls outside the recorded episode')
            starts = np.asarray(episode.get('force_start_step', np.zeros(len(states), dtype=int)))
            if starts.shape != (len(states),) or not np.all(starts == starts[0]):
                raise ValueError('Expected a constant force-start step for the episode')
            start = int(starts[0])
            if start != starts[0] or not 0 <= start <= stop:
                raise ValueError('Force-start step must precede or equal the resting step')
            record.update(observed_endpoint_xy=next_states[stop, xy].astype(float).tolist(), endpoint_steps=stop-start+1)
            positions.append(len(result) - 1)
            initial.append(states[start])
            commands.append(actions[start:stop + 1])
            start_steps.append(start)
        if not positions:
            return result
        horizons = torch.tensor([len(c) for c in commands], device=self.device)
        max_steps = max(map(len, commands))
        actions = torch.zeros((max_steps, len(positions), action_dim), device=self.device)
        for column, command in enumerate(commands):
            actions[:len(command), column] = torch.as_tensor(command, device=self.device)
        state = torch.as_tensor(np.stack(initial), dtype=torch.float32, device=self.device)
        latent = self._latent(z).expand(len(state), -1)
        divergence = torch.full((len(state),), -1, dtype=torch.long, device=self.device)
        for step in range(max_steps):
            active = (step < horizons) & (divergence < 0)
            predicted = self.model.advance(state, actions[step], latent)
            finite = torch.isfinite(predicted).all(dim=1)
            divergence = torch.where(active & ~finite, step, divergence)
            # Completed and failed rows retain their last finite state for diagnostics.
            state = torch.where((active & finite)[:, None], predicted, state)
        final_states, failed_steps = state.cpu().numpy(), divergence.cpu().numpy()
        for position, final_state, failed_step, start in zip(positions, final_states, failed_steps, start_steps):
            record = result[position]
            record['last_finite_state_max_abs'] = float(np.max(np.abs(final_state)))
            if failed_step >= 0:
                record.update(endpoint_status='nonfinite_prediction', divergence_step=start + int(failed_step))
                continue
            predicted_xy = final_state[xy].astype(np.float64)
            difference = predicted_xy - np.asarray(record['observed_endpoint_xy'])
            record.update(endpoint_error_m=float(np.hypot(*difference)), endpoint_status='finite',
                          predicted_endpoint_xy=predicted_xy.tolist(), endpoint_valid=True,
                          predicted_final_speed_m_s=float(np.linalg.norm(final_state[velocity].astype(np.float64))))
        return result

    @torch.no_grad()
    def moving_error(self, rows, z):
        """One-step normalized delta MSE, averaged over moving rows and state dimensions."""
        mask = np.asarray(rows['cube_moving'], dtype=bool)
        if not mask.any():
            raise ValueError('No cube-moving steps in the supplied episodes')
        moving = {k: np.asarray(rows[k])[mask] for k in ('state', 'action', 'next_state')}
        inputs, target = self.model.normalize(*_transitions(moving, self.device))
        prediction = self.model(inputs, self._latent(z).expand(len(inputs), -1))
        error = (prediction - target).square().double().mean().item()
        if not np.isfinite(error):
            raise ValueError('Nonfinite moving-step prediction loss')
        return error

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(dict(model=self.model.state_dict(), cube_ids=torch.as_tensor(self.cube_ids),
                        train_z=torch.from_numpy(self.train_z), validation=self.validation,
                        config=asdict(self.config)), temporary)
        temporary.replace(path)


def load_world_model(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    weights = saved['model']
    state_dim = len(weights['delta_mean'])
    model = DeltaModel(state_dim, len(weights['input_mean']) - state_dim).to(device)
    model.load_state_dict(weights)
    config = dict(saved['config'])
    # Checkpoints predating the empirical prior retain their original replay behavior.
    config.setdefault('inference_prior', 'zero')
    return TrainedModel(model, saved['cube_ids'].cpu().numpy(), saved['train_z'].cpu().numpy(),
                        saved['validation'], TrainConfig(**config))


def fit_world_model(train_rows, *, seed, device='cpu', config=TrainConfig(), log=None):
    cube_ids, cube, heldout_mask, held_episodes = _episode_split(train_rows, config)
    torch.manual_seed(seed)
    inputs, targets = _transitions(train_rows, device)
    cube_index = torch.as_tensor(cube, dtype=torch.long, device=device)
    train = torch.as_tensor(np.flatnonzero(~heldout_mask), device=device)
    validation = torch.as_tensor(np.flatnonzero(heldout_mask), device=device)
    model = DeltaModel(targets.shape[1], inputs.shape[1] - targets.shape[1]).to(device)
    _fit_scaling(model, inputs, targets, train)
    inputs, targets = model.normalize(inputs, targets)
    table = nn.Parameter(inputs.new_zeros(len(cube_ids), LATENT_DIM))
    opt = torch.optim.Adam([{'params': model.parameters(), 'lr': 1e-3}, {'params': [table], 'lr': 1e-2}])
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    best_mse = best_significant = float('inf')
    history = []
    clock = monotonic()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = train[torch.randperm(len(train), generator=generator, device=device)]
        for block in order.split(config.batch_size):
            opt.zero_grad(set_to_none=True)
            z = table[cube_index[block]]
            prediction = model(inputs[block], z)
            loss = (prediction - targets[block]).square().mean() + config.z_penalty * z.square().mean()
            loss.backward()
            opt.step()
        model.eval()
        mse = _row_errors(model, inputs, targets, validation, cube_index, table).double().mean().item()
        if not np.isfinite(mse):
            raise ValueError(f'Nonfinite validation MSE at epoch {epoch}')
        if mse < best_mse:
            best_mse, best_epoch = mse, epoch
            best_model, best_z = deepcopy(model.state_dict()), table.detach().clone()
        if mse < best_significant * (1 - config.min_relative_improvement):
            best_significant, last_improvement = mse, epoch
        record = dict(epoch=epoch, validation_mse=mse, elapsed_s=monotonic() - clock)
        history.append(record)
        if log is not None:
            log(record)
        if epoch - last_improvement >= config.patience:
            break
    model.load_state_dict(best_model)
    moving = np.asarray(train_rows['cube_moving'], dtype=bool)
    moving_held = torch.as_tensor(np.flatnonzero(heldout_mask & moving), device=device)
    report = _validation_gate(model, inputs, targets, moving_held, cube_index, cube_ids, best_z, seed + 2,
                              config.inference_prior)
    report.update(seed=seed, best_epoch=best_epoch, best_validation_mse=best_mse, history=history,
                  stopped_for='patience' if epoch - last_improvement >= config.patience else 'max_epochs',
                  train_rows=len(train), validation_rows=len(validation), split_seed=config.split_seed,
                  heldout_episodes=held_episodes)
    return TrainedModel(model, cube_ids, best_z.cpu().numpy(), report, config)
