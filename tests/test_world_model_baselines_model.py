"""Causal history, training-only scaling, and checkpoint regressions."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.world_model_baselines.data import history_indices
from experiments.world_model_baselines.model import (
    BaselineModel, _history_batch, load_checkpoint, train_model,
)


class NoPhysicalLabels(dict):
    def __getitem__(self, key):
        if key.startswith('true_'):
            raise AssertionError('Observation-only baseline accessed a physical label')
        return super().__getitem__(key)


class BaselineTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(2)

    def rows(self):
        episode, step = np.indices((4, 4)).reshape(2, -1)
        rng = np.random.default_rng(16)
        state = rng.normal(size=(16, 3)).astype('f')
        action = rng.normal(size=(16, 2)).astype('f')
        state[8:] += 100
        delta = rng.normal(scale=.01, size=state.shape).astype('f')
        return dict(cube_id=np.zeros(16, dtype=int), task=np.zeros(16, dtype=int),
                    episode_id=episode, step=step, state=state, action=action,
                    next_state=state + delta,
                    true_mass=np.repeat([1., 3., 100., 200.], 4),
                    true_friction=np.repeat([.1, .5, 2., 4.], 4))

    def test_training_scaling_checkpoint_and_no_privileged_reads(self):
        original = self.rows()
        train = original['episode_id'] < 2
        for kind, k in [('B0', 0), ('B1', 3), ('B2', 0)]:
            with self.subTest(kind=kind), TemporaryDirectory() as tmp:
                rows = original if kind == 'B2' else NoPhysicalLabels(original)
                # Only tiny synthetic unit fits; production hyperparameters are unchanged.
                with patch('experiments.world_model_baselines.model.MAX_EPOCHS', 2):
                    model, report = train_model(rows, train, ~train, 8, kind, k, 'cpu', tmp)
                inputs = np.concatenate((rows['state'], rows['action']), axis=1)[train]
                deltas = (rows['next_state'] - rows['state'])[train]
                for name, values in [('input', inputs), ('delta', deltas)]:
                    np.testing.assert_allclose(getattr(model, name + '_mean').numpy(),
                                               values.mean(0, dtype='d'), atol=1e-7)
                    np.testing.assert_allclose(getattr(model, name + '_sd').numpy(),
                                               values.std(0, dtype='d'), atol=1e-7)
                if kind == 'B2':
                    np.testing.assert_allclose(model.oracle_mean.numpy(), [2, .3])
                    np.testing.assert_allclose(model.oracle_sd.numpy(), [1, .2])
                restored, metadata = load_checkpoint(Path(tmp) / 'checkpoint.pt')
                self.assertEqual(metadata, report)
                self.assertEqual(report['train_rows'], 8)
                self.assertEqual(report['validation_rows'], 8)
                self.assertEqual(report['best_validation_mse'], min(x['validation_mse'] for x in report['history']))
                self.assertFalse(any(p.requires_grad for p in restored.parameters()))
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, restored.state_dict()[key]))
                self.assertTrue((Path(tmp) / 'training.json').exists())

    def test_history_is_completed_episode_local_and_padding_is_ignored(self):
        rows = self.rows()
        indices, lengths = history_indices(rows, 3)
        np.testing.assert_array_equal(indices[:4], [[-1, -1, -1], [0, -1, -1], [0, 1, -1], [0, 1, 2]])
        np.testing.assert_array_equal(indices[4], [-1, -1, -1])
        model = BaselineModel(3, 2, 'B1', 3)
        with torch.no_grad():
            model.input_mean.copy_(torch.arange(5))
            model.input_sd.copy_(torch.arange(1, 6))
        s, a, following = [torch.tensor(rows[key]) for key in ('state', 'action', 'next_state')]
        inputs = (torch.cat((s, a), 1) - model.input_mean) / model.input_sd
        following_normalized = (following - model.input_mean[:3]) / model.input_sd[:3]
        block = torch.arange(5)
        history, n = _history_batch(inputs, following_normalized, torch.tensor(indices), torch.tensor(lengths), block)
        expected = model.normalize_history(s, a, following)
        torch.testing.assert_close(history[3], expected[:3])
        changed = history.clone()
        padding = torch.arange(3)[None, :] >= n[:, None]
        changed[padding] = 10000
        torch.testing.assert_close(model(inputs[block], history, n), model(inputs[block], changed, n), atol=0, rtol=0)
        empty_rows = n == 0
        torch.testing.assert_close(model(inputs[block][empty_rows]),
                                   model(inputs[block][empty_rows], changed[empty_rows], n[empty_rows]))

    def test_advance_normalizes_oracle_and_recovers_physical_delta(self):
        model = BaselineModel(3, 2, 'B2')
        with torch.no_grad():
            model.input_mean.fill_(3)
            model.input_sd.fill_(2)
            model.delta_mean.fill_(.1)
            model.delta_sd.fill_(.2)
            model.oracle_mean.copy_(torch.tensor([.4, .5]))
            model.oracle_sd.copy_(torch.tensor([.1, .2]))
        state, action, oracle = torch.randn(4, 3), torch.randn(4, 2), torch.randn(4, 2)
        normalized = (torch.cat((state, action), 1) - model.input_mean) / model.input_sd
        prediction = model(normalized, oracle=model.normalize_oracle(oracle))
        torch.testing.assert_close(model.advance(state, action, oracle=oracle), state + prediction * .2 + .1)
        with self.assertRaises(ValueError):
            BaselineModel(3, 2)(normalized, oracle=oracle)

    def test_rejects_split_inside_episode(self):
        rows = self.rows()
        train = np.arange(16) < 7
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'whole episodes'):
            train_model(rows, train, ~train, 0, 'B0', 0, 'cpu', tmp)


if __name__ == '__main__':
    unittest.main()
