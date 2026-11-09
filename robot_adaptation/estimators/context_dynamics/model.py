"""Current-state dynamics with a privileged encoder trained by prediction alone."""
import torch
from torch import nn


def rotation_columns(q):
    """First two rotation-matrix columns, invariant to quaternion sign."""
    w,x,y,z = nn.functional.normalize(q,dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y),
                        2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), dim=-1)


def state_features(raw):
    return torch.cat((raw[...,:21], rotation_columns(raw[...,21:25]), raw[...,25:29],
                      rotation_columns(raw[...,29:33]), raw[...,33:]), dim=-1)


def constants_for_encoder(raw):
    return torch.cat((raw[...,:1].log(), raw[...,1:]),dim=-1)


class ContextDynamics(nn.Module):
    """Predict normalized state increments. The policy is external to this module."""
    def __init__(self, latent=8, width=256):
        super().__init__()
        self.latent = latent
        self.encoder = nn.Sequential(nn.Linear(10,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),
                                     nn.Linear(64,latent))
        self.dynamics = nn.Sequential(nn.Linear(43+5+latent,width), nn.SiLU(),
            nn.Linear(width,width), nn.SiLU(), nn.Linear(width,width), nn.SiLU(),
            nn.Linear(width,width), nn.SiLU(), nn.Linear(width,43))
        for name, size in [('state_mean',43),('state_sd',43),('delta_sd',43),
                           ('physics_mean',10),('physics_sd',10)]:
            self.register_buffer(name,torch.ones(size) if name.endswith('sd') else torch.zeros(size))

    def encode(self, constants):
        return self.encoder((constants_for_encoder(constants)-self.physics_mean)/self.physics_sd)

    def normalize(self, states):
        return (states-self.state_mean)/self.state_sd

    def advance(self, state, action, z):
        # State is normalized. Commands retain their issued [-1,1] units.
        inputs = torch.cat((state.clamp(-15,15), action, z), dim=-1)
        delta = self.dynamics(inputs).float()
        return state + delta*(self.delta_sd/self.state_sd)

    def rollout(self, initial, commands, z):
        state = initial
        future = []
        for action in commands.unbind(dim=-2):
            state = self.advance(state, action, z)
            future.append(state)
        return torch.stack(future,dim=-2)

    def loss_per_window(self, states, commands, z, feature_weights=None):
        predicted = self.rollout(states[...,0,:],commands,z)
        # Express accumulated displacement error in training delta-SD units.
        residual = (predicted-states[...,1:,:])*(self.state_sd/self.delta_sd)
        horizons = torch.arange(1,1+commands.shape[-2],device=commands.device)
        squared = residual.square()
        if feature_weights is not None:
            squared = squared*feature_weights
        return (squared.mean(-1)/horizons).mean(-1)
