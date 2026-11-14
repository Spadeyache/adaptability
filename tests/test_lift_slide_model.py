"""Leakage and model/latent isolation regressions for the one-step world model."""
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
import unittest

import numpy as np
import torch

from experiments.lift_slide_belief.model import (
    DeltaModel, TrainedModel, TrainConfig, fit_world_model, load_world_model,
)
from experiments.lift_slide_belief.planner import choose_force


class WorldModelTest(unittest.TestCase):
    def test_updated_gate_requires_shuffled_comparison_and_keeps_mean_diagnostic(self):
        from experiments.lift_slide_belief.model import _validation_gate

        class LatentPrediction(torch.nn.Module):
            def forward(self, inputs, z):
                return z[:, :1]

        table = torch.tensor([[-1., 0., 0., 0.], [1., 0., 0., 0.]])
        cube_index = torch.tensor([0, 0, 1, 1])
        result = _validation_gate(
            LatentPrediction(), torch.zeros(4, 2), .2 * table[cube_index, :1],
            torch.arange(4), cube_index, np.array([0, 1]), table, 7, 'training_distribution')
        self.assertTrue(result['passed'])
        self.assertEqual(result['required_controls'], ['shuffled'])
        self.assertTrue(result['comparisons']['shuffled']['passed'])
        self.assertFalse(result['comparisons']['mean']['passed'])

    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(2)
        c, t, e, s = np.indices((6, 3, 10, 8)).reshape(4, -1)
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
        self.assertEqual(len(held), 6 * 3 * 2)
        for c in self.model.cube_ids:
            for t in (0, 1, 2):
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
        for name in ('shuffled', 'mean'):
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

    def test_empirical_prior_and_legacy_checkpoint_behavior(self):
        model = DeltaModel(5, 2)
        for parameter in model.parameters():
            parameter.data.zero_()
        table = np.array([[1, 2, 3, 4], [3, 2, 5, 4]], dtype=np.float32)
        fitted = TrainedModel(model, np.arange(2), table, {}, TrainConfig())
        np.testing.assert_array_equal(fitted.prior_mean, [2, 2, 4, 4])
        np.testing.assert_allclose(fitted.prior_sd, [1, .01, 1, .01])
        rows = dict(state=np.zeros((3, 5)), next_state=np.zeros((3, 5)), action=np.zeros((3, 2)))
        np.testing.assert_array_equal(fitted.fit_z(rows), fitted.prior_mean)
        initial = fitted.prior_mean + 1
        z = fitted.fit_z(rows, initial=initial, steps=20)
        self.assertTrue(np.all(np.abs(z-fitted.prior_mean) < 1))
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'legacy.pt'
            fitted.save(path)
            saved = torch.load(path, weights_only=True)
            saved['config'].pop('inference_prior')
            saved['config'].pop('prior_variance_floor')
            torch.save(saved, path)
            restored = load_world_model(path)
            self.assertEqual(restored.config.inference_prior, 'zero')
            np.testing.assert_array_equal(restored.fit_z(rows), np.zeros(4))

    def test_old_two_task_inputs_still_train(self):
        rows = {k: v[self.rows['task'] < 2] for k, v in self.rows.items()}
        model = fit_world_model(rows, seed=2, config=TrainConfig(max_epochs=1, batch_size=256))
        self.assertEqual(len(model.validation['heldout_episodes']), 6 * 2 * 2)

    def test_batched_adaptation_matches_individual_fits_with_unequal_groups(self):
        groups = []
        for cube, count in ((0, 19), (2, 43), (4, 71)):
            selected = np.flatnonzero(self.rows['cube_id'] == cube)[:count]
            groups.append({key: value[selected] for key, value in self.rows.items()})
        for prior in ('training_distribution', 'zero'):
            model = TrainedModel(self.model.model, self.model.cube_ids, self.model.train_z,
                                 self.model.validation, replace(self.model.config, inference_prior=prior))
            for initials in (None, np.arange(12, dtype=np.float32).reshape(3, 4) * .01):
                expected = np.stack([model.fit_z(group, steps=30,
                                                initial=None if initials is None else initials[i])
                                     for i, group in enumerate(groups)])
                actual = model.fit_many_z(groups, steps=30, initials=initials)
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


class EndpointTest(unittest.TestCase):
    def setUp(self):
        self.network = DeltaModel(5, 1)
        for parameter in self.network.parameters():
            parameter.data.zero_()
        # Cube X doubles at each free-running step. One-step prediction uses the
        # action through Y, ensuring recorded controls reach the rollout.
        layers = self.network.network
        layers[0].weight.data[0, 0] = 1
        layers[0].weight.data[1, 5] = 1
        layers[2].weight.data[0, 0] = layers[2].weight.data[1, 1] = 1
        layers[4].weight.data[0, 0] = layers[4].weight.data[1, 1] = 1
        layers[6].weight.data[0, 0] = layers[6].weight.data[1, 1] = 1
        self.model = TrainedModel(self.network, np.arange(2), np.zeros((2, 4)), {}, TrainConfig())

    @staticmethod
    def episode(count):
        states = np.zeros((count, 5), dtype=np.float32)
        states[0, 0] = 1
        # Later observations stay at zero, so any use of measured intermediate
        # states produces an endpoint different from the autonomous prediction.
        return dict(state=states, next_state=np.zeros_like(states), action=np.ones((count, 1)),
                    step=np.arange(count), observed_stopped=np.ones(count, dtype=bool),
                    stop_step=np.full(count, count-1))

    def test_full_state_feedback_and_recorded_commands(self):
        result = self.model.stopping_errors([self.episode(3), self.episode(2)], np.zeros(4),
                                            cube_xy=(0, 1), cube_velocity=(2, 3, 4))
        for r, expected in zip(result, ([8, 3], [4, 2])):
            self.assertEqual(r['predicted_endpoint_xy'], expected)
            self.assertAlmostEqual(r['endpoint_error_m'], np.hypot(*expected))
            self.assertEqual(r['endpoint_status'], 'finite')
        predicted = self.model.predict_next_state(np.array([1, 0, 0, 0, 0]), np.array([1]), np.zeros(4))
        np.testing.assert_array_equal(predicted, [2, 1, 0, 0, 0])
        pulse = self.episode(4)
        pulse['force_start_step'] = np.full(4, 2)
        pulse['state'][2, 0] = 3
        result = self.model.stopping_error(pulse, np.zeros(4), cube_xy=(0, 1), cube_velocity=(2, 3, 4))
        self.assertEqual(result['predicted_endpoint_xy'], [12, 2])
        self.assertEqual(result['endpoint_steps'], 2)

    def test_huge_finite_error_is_kept_and_overflow_is_reported(self):
        episodes = [self.episode(100), self.episode(140), self.episode(2)]
        episodes[2]['observed_stopped'][:] = False
        finite, overflow, unrested = self.model.stopping_errors(episodes, np.zeros(4),
                                                               cube_xy=(0, 1), cube_velocity=(2, 3, 4))
        self.assertGreater(finite['endpoint_error_m'], 1e29)
        self.assertTrue(finite['endpoint_valid'])
        self.assertEqual(overflow['endpoint_status'], 'nonfinite_prediction')
        self.assertIsNone(overflow['endpoint_error_m'])
        self.assertEqual(overflow['divergence_step'], 127)
        self.assertEqual(unrested['endpoint_status'], 'observed_not_stopped')
        self.assertIsNone(unrested['endpoint_error_m'])

    def test_planner_chooses_force_using_frozen_belief(self):
        layers = self.network.network
        layers[0].weight.data[2, 6] = 1  # first latent coordinate
        layers[2].weight.data[2, 2] = layers[4].weight.data[2, 2] = 1
        layers[6].weight.data[1, 2] = 1
        forces = np.array([1, 2.5, 4])
        commands = np.broadcast_to(forces[:, None, None], (3, 2, 1)).copy()
        args = (self.model, np.zeros(5), [0, 1], 5, forces, commands)
        zero = choose_force(*args, z=np.zeros(4), cube_xy=(0, 1), cube_velocity=(2, 3, 4))
        adapted = choose_force(*args, z=np.array([1, 0, 0, 0]), cube_xy=(0, 1), cube_velocity=(2, 3, 4))
        self.assertEqual(zero['selected_force_n'], 2.5)
        self.assertEqual(zero['predicted_endpoint_xy'], [0, 5])
        self.assertEqual(adapted['selected_force_n'], 1)
        self.assertEqual(adapted['predicted_endpoint_xy'], [0, 4])
        self.assertEqual(zero['candidate_valid_fraction'], 1)

    def test_planner_flags_all_nonfinite_and_executes_declared_fallback(self):
        forces = np.array([1, 2.5, 4])
        commands = np.broadcast_to(forces[:, None, None], (3, 140, 1)).copy()
        result = choose_force(self.model, [1, 0, 0, 0, 0], [0, 1], 5, forces, commands,
                              z=np.zeros(4), cube_xy=(0, 1), cube_velocity=(2, 3, 4))
        self.assertEqual(result['selected_force_n'], 2.5)
        self.assertEqual(result['planner_status'], 'all_nonfinite_midpoint_fallback')
        self.assertIsNone(result['predicted_target_error_m'])
        self.assertEqual(result['valid_candidates'], 0)
        self.assertEqual(result['candidate_divergence_steps'], [127, 127, 127])
