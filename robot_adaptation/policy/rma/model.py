"""RMA² method adapted to ManiSkill3's 44 observable / 2 privileged interface.

Independent implementation: https://github.com/yichao-liang/rma4rma
Depth and object dictionaries are deliberately omitted for this state-based cube baseline.
"""
import torch
from torch import nn
from torch.distributions import Normal

OBSERVABLE = tuple(i for i in range(46) if i not in (42, 45))


def split_observation(obs):
    if obs.shape[-1] != 46:
        raise ValueError('Expected the explicit mass/friction task with 46 entries')
    # Fixed scales, no test-distribution normalization or clipping of OOD values.
    state = obs[..., list(OBSERVABLE)].clone()
    state[..., -2:] /= 40.0
    physical = torch.stack((obs[..., 42] / 9.81, obs[..., 45]), -1)
    return state, physical


def mlp(sizes, output_scale=1.):
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layer = nn.Linear(a, b)
        nn.init.orthogonal_(layer.weight, output_scale if i == len(sizes)-2 else 2**.5)
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        if i < len(sizes)-2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Agent(nn.Module):
    def __init__(self, envs=None, latent_dim=8):
        super().__init__()
        if envs is not None and (envs.single_observation_space.shape != (46,) or envs.single_action_space.shape != (5,)):
            raise ValueError('RMA requires 46 observations and 5 actions')
        self.encoder = mlp([2, 128, 128, latent_dim])
        self.actor = mlp([44+latent_dim, 256, 256, 256, 5], .01)
        self.critic = mlp([44+latent_dim, 256, 256, 256, 1])
        self.logstd = nn.Parameter(torch.full((1, 5), -.5))
        self.register_buffer('physics_center', torch.tensor([.07, .65]))
        self.register_buffer('physics_scale', torch.tensor([.05, .35]))

    def encode(self, physical):
        return self.encoder((physical-self.physics_center)/self.physics_scale)

    def features(self, obs):
        state, physical = split_observation(obs)
        return torch.cat((state, self.encode(physical)), -1)

    def action_from_latent(self, state, latent, deterministic=True):
        mean = self.actor(torch.cat((state, latent), -1))
        return mean if deterministic else Normal(mean, self.logstd.exp()).sample()

    def get_value(self, obs):
        return self.critic(self.features(obs))

    def get_action(self, obs, deterministic=False):
        state, physical = split_observation(obs)
        return self.action_from_latent(state, self.encode(physical), deterministic)

    def get_action_and_value(self, obs, action=None):
        x = self.features(obs)
        distribution = Normal(self.actor(x), self.logstd.exp())
        action = distribution.sample() if action is None else action
        return action, distribution.log_prob(action).sum(-1), distribution.entropy().sum(-1), self.critic(x)


class History:
    """Left-padded (current state, previous APPLIED action, validity) history."""
    def __init__(self, batch, length=50, device='cpu'):
        self.data = torch.zeros(batch, length, 50, device=device)

    def reset(self, mask=None):
        if mask is None:
            self.data.zero_()
        else:
            self.data[mask] = 0

    def append(self, state, previous_action):
        self.data = torch.roll(self.data, -1, dims=1)
        self.data[:, -1] = torch.cat((state, previous_action, torch.ones_like(state[:, :1])), -1)
        return self.data


class Adapter(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.temporal = nn.Sequential(nn.Conv1d(50, 64, 5, padding=2), nn.ELU(),
                                      nn.Conv1d(64, 64, 5, padding=2), nn.ELU())
        self.head = mlp([128, 64, latent_dim])

    def forward(self, history):
        x = self.temporal(history.transpose(1, 2))
        mask = history[..., -1:].transpose(1, 2)
        pooled = (x*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        return self.head(torch.cat((pooled, x[..., -1]), -1))
