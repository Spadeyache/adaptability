"""Episode-indexed training and held-out windows. Gradients never cross an episode."""
import json
from pathlib import Path

import numpy as np
import torch

from robot_adaptation.estimators.context_dynamics.model import ContextDynamics, constants_for_encoder, state_features


class Trajectories:
    def __init__(self, folder, device, deployment=False):
        info = json.loads((folder/'dataset.json').read_text())
        complete = json.loads((folder/'complete.json').read_text())
        if info['deployment'] != deployment:
            raise ValueError('Wrong data split')
        n = complete['episodes']
        h = info['config']['steps']
        self.states = torch.empty((n,h+1,43),device=device)
        self.commands = torch.empty((n,h,5),device=device)
        self.constants = torch.empty((n,10),device=device)
        self.tasks = torch.empty(n,device=device,dtype=torch.long)
        torch.set_num_threads(2)
        offset = 0
        for path in sorted(folder.glob('shard_*.npz')):
            with np.load(path) as s:
                count = len(s['episode_id'])
                np.testing.assert_array_equal(s['episode_id'],np.arange(offset, offset+count))
                sel = slice(offset,offset+count)
                self.states[sel] = state_features(torch.from_numpy(s['state'])).to(device)
                self.commands[sel] = torch.from_numpy(s['command']).to(device)
                self.constants[sel] = torch.from_numpy(s['constants']).to(device)
                self.tasks[sel] = torch.from_numpy(s['task_id']).to(device)
                offset += count
        assert offset == n
        self.device = device
        self.horizon = h
        ids = torch.arange(n,device=device)
        held = (ids//4)%10==9
        self.train_ids = ids[~held] if not deployment else ids[:0]
        # Half of the validation episodes select training checkpoints. Two further
        # quarters are reserved for successive collapse checks on disjoint cases.
        self.select_ids = ids[held & ((ids//40)%2==0)]
        self.check_ids = [ids[held & ((ids//40)%4==i)] for i in (1,3)]

    def add_training_episodes(self, folder):
        extra = Trajectories(folder,self.device)
        if self.horizon != extra.horizon:
            raise ValueError('Mixed control horizons')
        first = len(self.states)
        for name in ('states','commands','constants','tasks'):
            setattr(self,name,torch.cat((getattr(self,name),getattr(extra,name)),dim=0))
        self.train_ids = torch.cat((self.train_ids,first+extra.train_ids))
        # Keep model selection and both diagnostic folds on the original four-task mix.
        # The supplemental data's 10% holdout is never fitted.

    def fit_scaling(self, model):
        # Accumulate train-only moments by episode blocks, keeping GPU memory bounded.
        moments = {k:[] for k in ['state','delta','physics']}
        for indices in self.train_ids.split(512):
            states = self.states[indices]
            features = [states.flatten(0,1), states.diff(dim=1).flatten(0,1),
                        constants_for_encoder(self.constants[indices])]
            for key, values in zip(moments,features):
                v = values.double()
                moments[key].append((len(v),v.sum(0),v.square().sum(0)))
        for key, chunks in moments.items():
            n = sum(c[0] for c in chunks)
            mean = sum(c[1] for c in chunks)/n
            var = (sum(c[2] for c in chunks)/n - mean.square()).clamp_min(0)
            getattr(model,key+'_sd').copy_(var.sqrt().clamp_min(1e-4))
            if key != 'delta':
                getattr(model,key+'_mean').copy_(mean)

    def windows(self, ids, starts, length):
        steps = starts[:,None]+torch.arange(length+1,device=self.device)
        return self.states[ids[:,None],steps], self.commands[ids[:,None],steps[:,:-1]]

    def tiled(self, ids, length):
        starts = torch.arange(0,self.horizon-length+1,length,device=self.device)
        return self.windows(ids.repeat_interleave(len(starts)),starts.repeat(len(ids)),length)


def restore(run, device):
    checkpoint = torch.load(Path(run)/'model.pt',map_location=device,weights_only=True)
    model = ContextDynamics(checkpoint['latent'],checkpoint['width']).to(device)
    model.load_state_dict(checkpoint['model'])
    return model.eval(), checkpoint
