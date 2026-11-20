"""Isolated, property-free dynamics models and episode-bounded histories.

All inputs to this module are already normalized using training data statistics.
The prediction is a change in normalized state: x_next = x + model(x, a).
Only the explicitly named Oracle accepts true object properties. The two-part
model's embedding row is a training identifier, never a prediction target or a
numerical input feature. Its frozen-weight adaptation uses an explicit ``z``.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


HISTORY_LENGTH = 20
KINDS = ("plain", "history", "oracle", "two_part")


def make_history_windows(
    states: torch.Tensor,
    actions: torch.Tensor,
    episode_starts: torch.Tensor,
    indices: torch.Tensor | None = None,
    max_length: int = HISTORY_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left-aligned observed histories and lengths for selected rows.

    ``states`` and ``actions`` are flattened (N, feature) arrays, in temporal
    order within each episode. ``episode_starts`` is a boolean array of length
    N and must mark row zero and every new episode. No future row is read.
    ``indices`` selects the present rows; omitted means every row. Each returned
    window contains at most ``max_length`` (state, action) pairs ending at its
    selected present row. Padding is zero and is excluded by the GRU's lengths.
    Use selected indices for training batches to avoid a full N x 20 allocation.
    """
    if states.ndim != 2 or actions.ndim != 2 or len(states) != len(actions):
        raise ValueError("states/actions must be aligned two-dimensional arrays")
    if len(states) == 0 or max_length < 1:
        raise ValueError("nonempty data and a positive history length are required")
    if states.device != actions.device:
        raise ValueError("states and actions must share a device")
    if episode_starts.ndim != 1 or len(episode_starts) != len(states):
        raise ValueError("episode_starts must contain one boolean flag per row")
    if episode_starts.dtype != torch.bool:
        raise ValueError("episode_starts must be boolean")
    starts = episode_starts.to(device=states.device)
    if not bool(starts[0]):
        raise ValueError("the first row must start an episode")
    row = torch.arange(len(states), device=states.device)
    start_rows = torch.cummax(torch.where(starts, row, 0), dim=0).values
    if indices is None:
        indices = row
    else:
        indices = indices.to(device=states.device, dtype=torch.long)
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if bool(((indices < 0) | (indices >= len(states))).any()):
        raise IndexError("a selected history row is outside the data")
    lengths = (indices - start_rows[indices] + 1).clamp(max=max_length)
    offsets = torch.arange(max_length, device=states.device)[None, :]
    valid = offsets < lengths[:, None]
    rows = indices[:, None] - lengths[:, None] + 1 + offsets
    # Invalid padding gathers the present row, never any future observation.
    rows = torch.where(valid, rows, indices[:, None])
    windows = torch.cat((states[rows], actions[rows]), dim=-1)
    return windows.masked_fill(~valid[..., None], 0), lengths


def append_history(
    history: torch.Tensor,
    lengths: torch.Tensor,
    state: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one predicted state/action pair during a closed-loop rollout.

    Histories are left-aligned. A full window drops its oldest pair; a shorter
    one appends at its first padded position. The caller supplies predicted
    state and the known next action, so this routine cannot teacher-force a
    future measured state by itself.
    """
    if history.ndim != 3 or state.ndim != 2 or action.ndim != 2:
        raise ValueError("history must be B x K x F and state/action B x F")
    batch, width, features = history.shape
    pair = torch.cat((state, action), dim=-1)
    lengths = lengths.to(device=history.device, dtype=torch.long)
    if pair.shape != (batch, features) or lengths.shape != (batch,):
        raise ValueError("history, lengths, and current pair have incompatible shapes")
    if bool(((lengths < 1) | (lengths > width)).any()):
        raise ValueError("history lengths must be between one and window width")
    full = lengths == width
    shifted = torch.cat((history[:, 1:], history[:, -1:]), dim=1)
    updated = torch.where(full[:, None, None], shifted, history).clone()
    positions = lengths.clamp(max=width - 1)
    updated[torch.arange(batch, device=history.device), positions] = pair
    return updated, (lengths + 1).clamp(max=width)


def _mlp(input_dim: int, output_dim: int, width: int, depth: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    for _ in range(depth):
        modules.extend((nn.Linear(input_dim, width), nn.SiLU()))
        input_dim = width
    modules.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*modules)


class DynamicsModel(nn.Module):
    """Four baselines with a common MLP readout and normalized delta target.

    ``plain``: MLP(state, action).
    ``oracle``: MLP(state, action, normalized mass, normalized friction).
    ``two_part``: MLP(state, action, learned z), with zero-initialized train z.
    ``history``: GRU over at most the last 20 observed state/action pairs,
    including the current pair; MLP(state, action, GRU final hidden state).

    Every history is encoded from a zero hidden state. Histories contain no
    preceding-episode samples, and packed lengths exclude padded timesteps.
    Fitting an unseen cube uses explicit ``z`` after freezing all parameters;
    no lookup or property input is needed. The L2 embedding penalty is computed
    by the training loop, allowing the same loss accounting across candidates.
    """

    def __init__(
        self,
        kind: str,
        state_dim: int,
        action_dim: int,
        width: int = 128,
        depth: int = 2,
        z_dim: int = 2,
        num_cubes: int = 200,
    ):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if min(state_dim, action_dim, width, depth, z_dim, num_cubes) < 1:
            raise ValueError("model dimensions and depth must be positive")
        self.kind = kind
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        input_dim = state_dim + action_dim
        self.embedding: nn.Embedding | None = None
        self.gru: nn.GRU | None = None
        if kind == "two_part":
            self.embedding = nn.Embedding(num_cubes, z_dim)
            nn.init.zeros_(self.embedding.weight)
            input_dim += z_dim
        elif kind == "oracle":
            input_dim += 2
        elif kind == "history":
            self.gru = nn.GRU(state_dim + action_dim, width, batch_first=True)
            input_dim += width
        self.net = _mlp(input_dim, state_dim, width, depth)

    def mean_z(self) -> torch.Tensor:
        """Mean train embedding, detached, for frozen-weight adaptation start."""
        if self.embedding is None:
            raise ValueError("mean_z is defined only for the two-part model")
        return self.embedding.weight.detach().mean(dim=0)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        cube_ids: torch.Tensor | None = None,
        properties: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        history: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.shape[:-1] != action.shape[:-1]:
            raise ValueError("state and action batch dimensions must match")
        if state.shape[-1] != self.state_dim or action.shape[-1] != self.action_dim:
            raise ValueError("state/action feature dimensions do not match the model")
        if self.kind != "oracle" and properties is not None:
            raise ValueError("only the Oracle may receive true properties")
        inputs = [state, action]
        if self.kind == "two_part":
            if z is None:
                if cube_ids is None:
                    raise ValueError("two_part needs training cube_ids or explicit fitted z")
                z = self.embedding(cube_ids)
            if z.shape[-1] != self.z_dim:
                raise ValueError("z has the wrong feature dimension")
            inputs.append(torch.broadcast_to(z, (*state.shape[:-1], self.z_dim)))
        elif self.kind == "oracle":
            if properties is None or properties.shape[-1] != 2:
                raise ValueError("Oracle needs normalized mass and friction")
            inputs.append(torch.broadcast_to(properties, (*state.shape[:-1], 2)))
        elif self.kind == "history":
            if history is None or lengths is None:
                raise ValueError("History needs explicit observed windows and lengths")
            if state.ndim != 2 or history.ndim != 3:
                raise ValueError("History requires batched state and B x K x F windows")
            if history.shape[0] != len(state) or history.shape[2] != self.state_dim + self.action_dim:
                raise ValueError("History input dimensions do not match state/action")
            if history.shape[1] > HISTORY_LENGTH or history.shape[1] < 1:
                raise ValueError("History windows must contain at most 20 timesteps")
            if lengths.shape != (len(state),):
                raise ValueError("History needs one length per batch row")
            if bool(((lengths < 1) | (lengths > history.shape[1])).any()):
                raise ValueError("History lengths must exclude only trailing padding")
            packed = pack_padded_sequence(
                history, lengths.detach().to("cpu"), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
            inputs.append(hidden[-1])
        return self.net(torch.cat(inputs, dim=-1))
