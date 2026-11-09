import unittest

import numpy as np
import torch

from experiments.mass_belief.collect import feature
from robot_adaptation.estimators.mass_belief.model import WindowMass


class MassBeliefTests(unittest.TestCase):
    def test_features_hide_weight_and_alignment_matches_live_history(self):
        rng = np.random.default_rng(12)
        obs = rng.normal(size=(41,45)).astype('float32')
        applied = rng.normal(size=(40,5)).astype('float32')
        offline = feature(obs[1:],applied)
        obs[:,42] = rng.normal(size=41)
        np.testing.assert_array_equal(offline,np.stack([feature(o,a) for o,a in zip(obs[1:],applied)]))
        model = WindowMass(25,40,500).eval()
        torch.set_num_threads(1)
        with torch.no_grad():
            sequence = torch.from_numpy(offline)[None]
            batched = model.windows(sequence)
            for end in [1,16,25,40]:
                live = model.windows(sequence[:,max(0,end-25):end])[:,-1]
                torch.testing.assert_close(batched[:,end-1], live)
                torch.testing.assert_close(model(batched[:,end-1]),model(live))


if __name__=='__main__':
    unittest.main()
