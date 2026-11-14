"""Leakage and model/latent isolation regressions for the one-step world model."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from experiments.lift_slide_belief.model import TrainConfig, fit_world_model, load_world_model


class WorldModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(2)
        c, t, e, s = np.indices((6, 2, 10, 8)).reshape(4, -1)
        rng = np.random.default_rng(9)
        state = rng.standard_normal((len(c), 5)).astype('f')
        action = rng.standard_normal((len(c), 2)).astype('f')
        # Only the latent identity can predict this offset across independent episodes.
        delta = (.2 * c[:, None] + .01 * action[:, :1]).astype('f')
        cls.rows = dict(cube_id=c, task=t, episode_id=e, step=s, state=state, action=action,
                        next_state=state+delta, cube_moving=s % 4 != 0)
        cls.model = fit_world_model(cls.rows, seed=1, config=TrainConfig(max_epochs=25, batch_size=256))

    def test_split_scaling_and_shuffled_cube_pairs(self):
        result = self.model.validation
        held = {tuple(record) for record in result['heldout_episodes']}
        self.assertEqual(len(held), 6 * 2 * 2)
        for c in self.model.cube_ids:
            for t in (0, 1):
                self.assertEqual(sum(cube == c and task == t for cube, task, _ in held), 2)
        rows = self.rows
        train = np.asarray([key not in held for key in zip(rows['cube_id'], rows['task'], rows['episode_id'])])
        expected_input = np.column_stack((rows['state'], rows['action']))[train]
        expected_delta = (rows['next_state'] - rows['state'])[train]
        for name, data in (('input', expected_input), ('delta', expected_delta)):
            np.testing.assert_allclose(getattr(self.model.model, name+'_mean').numpy(), data.mean(0, dtype='d'), atol=1e-6)
            np.testing.assert_allclose(getattr(self.model.model, name+'_sd').numpy(), data.std(0, dtype='d'), atol=1e-6)
        self.assertTrue(result['passed'])
        self.assertEqual(result['moving_rows'], int(np.sum((~train) & rows['cube_moving'])))
        donors = np.asarray(result['shuffled_donor_ids'])
        np.testing.assert_array_equal(np.sort(donors), self.model.cube_ids)
        self.assertTrue(np.all(donors != self.model.cube_ids))
        for name in ('shuffled', 'zero'):
            self.assertGreater(result['mse'][name], result['mse']['own'])
            self.assertGreater(result['comparisons'][name]['difference_ci95'][0], 0)

    def test_adaptation_keeps_frozen_state_and_restores_exactly(self):
        rows = {k: v[self.rows['cube_id'] == 0] for k, v in self.rows.items()}
        reference = {k: v.clone() for k,v in self.model.model.state_dict().items()}
        table = self.model.train_z.copy()
        z = self.model.fit_z(rows, steps=30)
        self.assertGreater(np.linalg.norm(z), 0)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'
            self.model.save(path)
            restored = load_world_model(path)
            self.assertEqual(restored.moving_error(rows, z), self.model.moving_error(rows, z))
            np.testing.assert_array_equal(restored.train_z, table)
        for name, weights in self.model.model.state_dict().items():
            self.assertTrue(torch.equal(weights, reference[name]))
        np.testing.assert_array_equal(self.model.train_z, table)
