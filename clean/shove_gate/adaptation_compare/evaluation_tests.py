"""Synthetic orchestration checks; never touch experiment data or weights."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import evaluation as ev


class FakeModel(torch.nn.Module):
    def __init__(self, variant):
        super().__init__(); self.variant = variant
        self.weight = torch.nn.Parameter(torch.tensor([0.]))
        self.z_dim = 4 if variant == 'z4' else 0
        self.embedding = torch.nn.Embedding(180, 4) if self.z_dim else None
        if self.embedding is not None:
            torch.nn.init.zeros_(self.embedding.weight)


class OrchestrationTests(unittest.TestCase):
    def test_independent_reset_sequential_carry_and_cached_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'data').mkdir()
            ev.write(root/'data/cubes.json', {'cubes': [dict(cube_id=c, mass_kg=.3, combined_friction=.5)
                                                       for c in list(range(180))+[290,291,292]]})
            cp_path = root/'fake.pt'; cp_path.write_bytes(b'synthetic checkpoint only')
            calls = []
            def load(variant, seed, device):
                return FakeModel(variant), {'normalizer': {'fixed': True}}, cp_path
            def episodes(cubes, eps, **kwargs):
                return [dict(cube_id=c, episode_id=e) for c in cubes for e in eps]
            def fit_z(model, eps, norm, **kwargs):
                self.assertTrue(all(e['episode_id'] < 5 for e in eps))
                cubes = sorted({e['cube_id'] for e in eps})
                return dict(cube_ids=cubes, z_values=np.array([[c-289]*4 for c in cubes], dtype=np.float32),
                            weights_unchanged=True)
            def tune(model, eps, norm, **kwargs):
                self.assertTrue(all(e['episode_id'] < 5 for e in eps))
                cubes = {e['cube_id'] for e in eps}; self.assertEqual(len(cubes), 1)
                c = next(iter(cubes)); before = float(model.weight.item())
                with torch.no_grad(): model.weight.add_(c-289)
                calls.append((c, before, len(eps)))
                return dict(optimizer_reset=True, updates=50)
            def score(model, eps, norm, scales, z_by_cube=None, properties=None, **kwargs):
                self.assertEqual(properties is not None, model.variant == 'oracle')
                self.assertTrue(all(e['episode_id'] >= 5 for e in eps))
                cubes = sorted({e['cube_id'] for e in eps})
                epcounts = {c: sum(e['cube_id']==c for e in eps) for c in cubes}
                rows = [dict(cube_id=c, mean=float(model.weight.item()) +
                             (float(np.mean(z_by_cube[c])) if z_by_cube is not None else 0.),
                             episodes=epcounts[c]) for c in cubes]
                avg = float(np.mean([r['mean'] for r in rows]))
                return dict(scores={m: avg for m in ev.SCORES},
                            aggregates={m: dict(per_cube=rows) for m in ev.SCORES})
            with patch.multiple(ev, TEST_IDS=[290,291,292], SNAPSHOTS=[1,2,3]), \
                 patch.multiple(ev.ad, load_frozen=load, load_episodes=episodes,
                                fit_latents=fit_z, finetune=tune, score=score):
                ev.one_seed(root, 101, {'A': {'z_dim': 4, 'penalty': .1}, 'B': {'learning_rate': 1e-5}}, {}, 'cpu')
            # 2 rates * 5 prefixes * 3 independent cubes; each resets to 0.
            self.assertEqual(len(calls), 36)
            self.assertTrue(all(before == 0. for _, before, _ in calls[:30]))
            self.assertEqual([before for _, before, _ in calls[30:]], [0.,1.,3.,0.,1.,3.])
            for rate in ev.RATES:
                seq = ev.read(root/f'results/sequence/B_lr{ev.rate_name(rate)}_seed101.json')
                self.assertEqual([r['first_cube']['scores']['composite'] for r in seq['snapshots']], [1.,3.,6.])
                self.assertEqual([r['cube_id'] for r in seq['visits']], [290,291,292])
                for i in range(1,3):
                    self.assertEqual(seq['visits'][i]['before_weights_sha256'], seq['visits'][i-1]['after_weights_sha256'])
            a = ev.read(root/'results/sequence/A_seed101.json')
            self.assertEqual([r['first_cube']['scores']['composite'] for r in a['snapshots']], [1.,1.,1.])
            self.assertEqual(len({r['z_first_sha256'] for r in a['snapshots']}), 1)
            self.assertEqual(ev.read(root/'results/prefix/A_seed101_k5.json')['fit']['z_values'][0], [1.,1.,1.,1.])


if __name__ == '__main__': unittest.main()
