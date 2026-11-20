"""Scientific safeguards for history boundaries and frozen-weight z adaptation."""
import unittest

import torch

from clean.models import DynamicsModel, append_history, make_history_windows


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)

    def test_history_never_crosses_episode_or_reads_future(self):
        states = torch.arange(30, dtype=torch.float32)[:, None]
        actions = (100 + states).clone()
        starts = torch.zeros(30, dtype=torch.bool)
        starts[[0, 24]] = True
        windows, lengths = make_history_windows(states, actions, starts, torch.tensor([23, 24, 26]))
        self.assertEqual(lengths.tolist(), [20, 1, 3])
        self.assertEqual(windows[0, :, 0].tolist(), list(range(4, 24)))
        self.assertEqual(windows[1, 0].tolist(), [24, 124])
        self.assertEqual(windows[2, :3, 0].tolist(), [24, 25, 26])
        self.assertTrue(torch.equal(windows[1, 1:], torch.zeros_like(windows[1, 1:])))
        # Replacing all future measurements cannot change the selected window.
        future_changed = states.clone()
        future_changed[27:] = 1e9
        again, _ = make_history_windows(future_changed, actions, starts, torch.tensor([26]))
        self.assertTrue(torch.equal(windows[2], again[0]))

    def test_padding_does_not_advance_hidden_state(self):
        model = DynamicsModel("history", 3, 2, width=8)
        state, action = torch.randn(2, 3), torch.randn(2, 2)
        history = torch.randn(2, 20, 5)
        lengths = torch.tensor([1, 4])
        first = model(state, action, history=history, lengths=lengths)
        for row, valid in enumerate(lengths):
            history[row, valid:] = 1e6
        second = model(state, action, history=history, lengths=lengths)
        self.assertTrue(torch.equal(first, second))
        # A batch neighbor from a different episode cannot change this output.
        single = model(state[:1], action[:1], history=history[:1, :1], lengths=torch.tensor([1]))
        torch.testing.assert_close(first[:1], single)

    def test_append_history_preserves_last_twenty_and_short_lengths(self):
        history = torch.zeros(2, 20, 2)
        history[0, :, 0] = torch.arange(20)
        history[1, :2, 0] = torch.tensor([90, 91])
        updated, lengths = append_history(history, torch.tensor([20, 2]), torch.tensor([[20.], [92.]]), torch.tensor([[3.], [4.]]))
        self.assertEqual(updated[0, :, 0].tolist(), list(range(1, 21)))
        self.assertEqual(updated[1, :3, 0].tolist(), [90, 91, 92])
        self.assertEqual(lengths.tolist(), [20, 3])
        self.assertEqual(history[0, 0, 0].item(), 0)  # no mutation of observed context

    def test_only_fitted_z_changes_when_weights_are_frozen(self):
        model = DynamicsModel("two_part", 3, 2, width=8, z_dim=2)
        self.assertTrue(torch.equal(model.embedding.weight, torch.zeros(200, 2)))
        model.requires_grad_(False)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        fitted_z = torch.nn.Parameter(model.mean_z().clone())
        optimizer = torch.optim.Adam([fitted_z], lr=.01)
        state, action = torch.randn(12, 3), torch.randn(12, 2)
        target = torch.randn(12, 3)
        optimizer.zero_grad()
        (model(state, action, z=fitted_z) - target).square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(fitted_z.detach(), torch.zeros(2)))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_context_interfaces_and_property_exclusion(self):
        state, action = torch.randn(4, 3), torch.randn(4, 2)
        contexts = {
            "plain": {},
            "oracle": {"properties": torch.randn(4, 2)},
            "two_part": {"cube_ids": torch.tensor([0, 1, 198, 199])},
            "history": {"history": torch.randn(4, 20, 5), "lengths": torch.tensor([1, 5, 20, 2])},
        }
        for kind, context in contexts.items():
            model = DynamicsModel(kind, 3, 2, width=8)
            self.assertEqual(model(state, action, **context).shape, (4, 3))
            if kind != "oracle":
                with self.assertRaisesRegex(ValueError, "only the Oracle"):
                    model(state, action, properties=torch.randn(4, 2), **context)


if __name__ == "__main__":
    unittest.main()
