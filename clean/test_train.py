"""Trainer checks use synthetic tensors only, never held-out simulator data."""
import contextlib
import io
import unittest
from types import SimpleNamespace

import torch

from clean.models import DynamicsModel
from clean.train import _check_pack, _prediction_loss, fit_z, parameter_digest, resolve_config


class TrainTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(2)

    def test_training_cannot_accept_held_out_id_or_validation_episode(self):
        pack = SimpleNamespace(x=torch.zeros(2, 3), y=torch.zeros(2, 3),
            a=torch.zeros(2, 2), cube_ids=torch.tensor([0, 200]), episode_ids=torch.tensor([0, 0]))
        with self.assertRaisesRegex(ValueError, "200 training cubes"):
            _check_pack(pack, training=True)
        pack.cube_ids[:] = 0
        pack.episode_ids[1] = 8
        with self.assertRaisesRegex(ValueError, "authorized training/fitting partition"):
            _check_pack(pack, training=True)

    def test_multistep_loss_does_not_teacher_force(self):
        class Increment(torch.nn.Module):
            kind = "plain"

            def forward(self, state, action):
                return torch.ones_like(state)

        # Later measured inputs are deliberately inconsistent with this exact
        # recursive trajectory, detecting any accidental future-state input.
        pack = SimpleNamespace(x=torch.tensor([[0.], [999.], [999.]]),
            y=torch.tensor([[1.], [2.], [3.]]), a=torch.zeros(3, 1),
            cube_ids=torch.zeros(3, dtype=torch.long), moving=torch.ones(3, dtype=torch.bool))
        config = resolve_config({"prediction_horizon": 3}, state_dim=1)
        loss = _prediction_loss(Increment(), pack, torch.tensor([0]), config)
        self.assertEqual(loss.item(), 0.)

    def test_fresh_latents_fit_without_changing_any_model_weight(self):
        model = DynamicsModel("two_part", 3, 2, width=8)
        model.embedding.weight.data.copy_(torch.randn_like(model.embedding.weight) * .1)
        before = parameter_digest(model)
        pack = SimpleNamespace(x=torch.randn(12, 3), a=torch.randn(12, 2),
            y=torch.randn(12, 3), cube_ids=torch.tensor([0] * 6 + [200] * 6),
            episode_ids=torch.tensor([0] * 3 + [1] * 3 + [0] * 3 + [1] * 3),
            remaining=torch.tensor([3, 2, 1] * 4), moving=torch.ones(12, dtype=torch.bool))
        initial = model.mean_z().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            result = fit_z(model, pack, {"fit_steps": 3}, seed=101)
        self.assertEqual(result["cube_ids"].tolist(), [0, 200])
        self.assertEqual(result["z"].shape, (2, 2))
        self.assertFalse(torch.equal(result["z"], initial.expand(2, -1)))
        self.assertEqual(before, parameter_digest(model))
        self.assertEqual(result["log"]["parameter_sha256_before"], result["log"]["parameter_sha256_after"])
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters()))
        pack.episode_ids[0] = 5
        with self.assertRaisesRegex(ValueError, "authorized training/fitting partition"):
            fit_z(model, pack, {"fit_steps": 1}, seed=101)


if __name__ == "__main__":
    unittest.main()
