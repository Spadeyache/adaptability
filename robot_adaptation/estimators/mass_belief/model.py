from dataclasses import dataclass

import torch
from torch import nn


class WindowMass(nn.Module):
    """Predict mass in grams from a bounded history ending at this step."""

    def __init__(self, window, low, high):
        super().__init__()
        self.window = window
        self.low = low
        self.span = high - low
        self.register_buffer('center', torch.zeros(25))
        self.register_buffer('scale', torch.ones(25))
        self.temporal = nn.Sequential(nn.Conv1d(26, 32, 5, padding=2), nn.SiLU(),
                                      nn.Conv1d(32, 32, 5, padding=2), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(96, 40), nn.SiLU(), nn.Linear(40, 1))

    def windows(self, sequence):
        # One padded window per post-action measurement, with a validity channel.
        x = ((sequence - self.center) / self.scale).clamp(-10, 10)
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1).transpose(1, 2)
        padded = nn.functional.pad(x, (self.window-1, 0))
        return padded.unfold(-1, self.window, 1).transpose(1, 2)

    def forward(self, windows):
        x = self.temporal(windows)
        pooled = torch.cat([x.mean(-1), x.amax(-1), x[:, :, -1]], dim=1)
        return self.low + self.span * self.head(pooled).sigmoid().flatten()


@dataclass
class MassBelief:
    mean: float
    variance: float

    def update(self, measurement, observation):
        # z = offset + slope * mass + noise. Static object, process noise Q=0.
        slope, offset, noise = observation
        gain = self.variance * slope / (slope*slope*self.variance + noise)
        self.mean += gain * (measurement - offset - slope*self.mean)
        self.variance = (1-gain*slope)**2 * self.variance + gain*gain*noise
