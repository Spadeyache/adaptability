"""A small recurrent clip encoder with accumulated mean evidence.

The model has named, positive outputs; it does not identify causal failure
reasons. Prefix supervision teaches it to infer after 1..K completed attempts.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .data import FEATURE_NAMES, OUTCOME_NAMES


class InteractionEstimator(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.GRU(len(FEATURE_NAMES), hidden, batch_first=True)
        self.outcome_encoder = nn.Sequential(nn.Linear(hidden + len(OUTCOME_NAMES), hidden), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def encode(self, x, lengths, outcomes):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.encoder(packed)
        return self.outcome_encoder(torch.cat((h[-1], outcomes), dim=-1))

    def predict_standard_log(self, evidence):
        return self.head(evidence)

    def forward(self, x, lengths, outcomes, attempts, episode_indices=None):
        embeddings = self.encode(x, lengths, outcomes)
        if episode_indices is not None:
            # Several bouts may belong to one attempt. Match persistent inference:
            # each attempt contributes its mean bout embedding with equal weight.
            episode_count = int(episode_indices.max()) + 1
            sums = torch.zeros(episode_count, self.hidden, dtype=embeddings.dtype, device=x.device)
            sums.index_add_(0, episode_indices, embeddings)
            counts = torch.bincount(episode_indices, minlength=episode_count).to(embeddings.dtype)
            embeddings = sums / counts[:, None]
        embeddings = embeddings.reshape(-1, attempts, self.hidden)
        count = torch.arange(1, attempts + 1, device=x.device).reshape(1, -1, 1)
        return self.head(embeddings.cumsum(dim=1) / count)
