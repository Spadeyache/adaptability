"""Synthetic, label-free checks of scientific metric and rollout semantics."""
import json
import math
import unittest

import numpy as np
import torch
from torch import nn

from clean.evaluate import (aggregate_seed_metrics, derangement, evaluate,
                            json_safe, success_criteria)
from clean.models import make_history_windows


class Pack:
    def __init__(self, lengths=(11, 21), velocities=(1., 2.)):
        self.x = torch.cat([torch.arange(n, dtype=torch.float32)[:, None] * v
                            for n, v in zip(lengths, velocities)])
        self.y = torch.cat([(torch.arange(n, dtype=torch.float32)[:, None] + 1) * v
                            for n, v in zip(lengths, velocities)])
        self.a = torch.zeros((sum(lengths), 1))
        self.cube_ids = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lengths)])
        self.episode_ids = torch.zeros(sum(lengths), dtype=torch.long)
        self.step = torch.cat([torch.arange(n) for n in lengths])
        self.remaining = torch.cat([torch.arange(n, 0, -1) for n in lengths])
        self.moving = torch.ones(sum(lengths), dtype=torch.bool)
        self.starts = self.step == 0
        self.history_calls = 0

    def history(self, indices):
        self.history_calls += 1
        return make_history_windows(self.x, self.a, self.starts, indices)


class ConstantModel(nn.Module):
    def __init__(self, value=0., kind="plain"):
        super().__init__()
        self.kind, self.value = kind, value

    def forward(self, state, action, history=None, lengths=None, **kwargs):
        if self.kind == "history":
            latest = history[torch.arange(len(state)), lengths - 1, :state.shape[1]]
            torch.testing.assert_close(latest, state)
        return torch.full_like(state, self.value)


class EvaluationTests(unittest.TestCase):
    def test_equal_cube_weight_pooled_r2_and_recursive_error(self):
        pack = Pack()
        model = ConstantModel()
        result = evaluate(model, pack, {}, batch_size=7)
        self.assertAlmostEqual(result["one_step_nmse"], 2.5)
        self.assertAlmostEqual(result["rollout_error"], 2.5 * 38.5)
        residual = ((pack.x - pack.y) ** 2).sum().item()
        denominator = ((pack.y.double() - pack.y.double().mean(dim=0)) ** 2).sum().item()
        self.assertAlmostEqual(result["r2"], 1 - residual / denominator)
        self.assertEqual(result["rollout_starts"], (11 - 9) + (21 - 9))
        self.assertEqual(result["moving_steps"], 32)
        self.assertTrue(model.training, "evaluate must restore caller's training mode")

    def test_history_rollout_uses_predicted_states_and_known_actions(self):
        pack = Pack((13,), (2.,))
        result = evaluate(ConstantModel(1., "history"), pack, {})
        self.assertAlmostEqual(result["one_step_nmse"], 1.)
        self.assertAlmostEqual(result["rollout_error"], 38.5)
        self.assertEqual(pack.history_calls, 2, "future histories must never be loaded")

    def test_only_moving_rows_start_rollouts(self):
        pack = Pack((14,), (2.,))
        pack.moving[:] = False
        pack.moving[[0, 4, 13]] = True
        result = evaluate(ConstantModel(1.), pack, {})
        self.assertEqual(result["moving_steps"], 3)
        self.assertEqual(result["rollout_starts"], 2)
        self.assertAlmostEqual(result["rollout_error"], 38.5)

    def test_invalid_remaining_cannot_cross_episode_boundary(self):
        pack = Pack()
        pack.remaining[8] = 10
        with self.assertRaisesRegex(ValueError, "episode boundary"):
            evaluate(ConstantModel(), pack, {})

    def test_missing_cube_motion_is_explicit_and_not_silently_dropped(self):
        pack = Pack()
        pack.moving[pack.cube_ids == 1] = False
        result = evaluate(ConstantModel(), pack, {})
        self.assertTrue(math.isnan(result["one_step_nmse"]))
        self.assertTrue(math.isnan(result["rollout_error"]))
        self.assertEqual(result["missing_moving_cubes"], [1])

    def test_nonfinite_scores_remain_invalid_and_json_null_is_explained(self):
        result = evaluate(ConstantModel(math.inf), Pack(), {})
        self.assertTrue(math.isinf(result["one_step_nmse"]))
        clean = json_safe(result)
        self.assertIsNone(clean["one_step_nmse"])
        self.assertIn("one_step_nmse", clean["nonfinite_paths"])
        self.assertIn("one_step_nmse", json_safe(clean)["nonfinite_paths"])
        json.dumps(clean, allow_nan=False)

    def test_five_seeds_sample_std_and_locked_success_rules(self):
        rows = [dict(one_step_nmse=float(i), r2=float(i), rollout_error=float(i)) for i in range(5)]
        result = aggregate_seed_metrics(rows)
        self.assertEqual(result["one_step_nmse"]["mean"], 2.)
        self.assertAlmostEqual(result["one_step_nmse"]["std"], np.std(np.arange(5), ddof=1))
        with self.assertRaises(ValueError):
            aggregate_seed_metrics(rows[:4])
        def summary(mean, std=.01):
            return {"one_step_nmse": {"mean": mean, "std": std}}
        good = success_criteria(summary(2), summary(1.4), summary(1), summary(1.4), summary(1.6),
                                {f"F{i}": True for i in range(1, 10)})
        self.assertTrue(all(good[f"S{i}"] for i in range(1, 5)))
        self.assertAlmostEqual(good["gap_recovery"], .6)
        bad = success_criteria(summary(1), summary(1), summary(2), summary(.5), summary(1))
        self.assertFalse(bad["S1"], "a nonpositive Oracle gap cannot pass")
        self.assertFalse(bad["S4"], "missing failure checks cannot pass")

    def test_derangement_has_no_self_assignment_and_is_reproducible(self):
        for n in (2, 3, 30, 200):
            p = derangement(101, n)
            np.testing.assert_array_equal(np.sort(p), np.arange(n))
            np.testing.assert_array_equal(p, derangement(101, n))
            self.assertTrue(np.all(p != np.arange(n)))
        with self.assertRaises(ValueError):
            derangement(1, 1)


if __name__ == "__main__":
    unittest.main()
